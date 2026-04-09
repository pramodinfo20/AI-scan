import logging
import os
from openai import OpenAI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
import pyheif
import cv2
import pytesseract
import pdfplumber
import imageio
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Middleware to increase the request size limit
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        # Check if the request body is too large
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_upload_size:
                raise HTTPException(status_code=413, detail="File size exceeds the limit.")
        return await call_next(request)

# Function to load datenmodell.txt as "wissen"
def load_wissen(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            wissen = file.read()
            logger.info("Datenmodell successfully loaded.")
            return wissen
    except FileNotFoundError:
        logger.error("Datenmodell file not found.")
        raise HTTPException(status_code=500, detail="Datenmodell file not found.")
    except Exception as e:
        logger.error(f"Error loading datenmodell: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading datenmodell: {e}")

# Convertiert GIF in JPG
def convert_gif_to_jpg(gif_path, output_path):
    # Read the GIF using imageio, taking only the first frame
    gif = imageio.mimread(gif_path, memtest=False)  # memtest=False to avoid memory warnings on large GIFs
    
    # Convert the first frame to an Image and save as JPG
    img = Image.fromarray(gif[0])  # Only take the first frame
    img = img.convert("RGB")  # Convert to RGB to remove transparency
    img.save(output_path, "JPEG")    
# Process Files for OCR and let openai correct the output
def process_image_for_ocr(file, temp_file_path):
  try:
    # Convert HEIC/Apple Images and Open the image to check format
    file_extension = temp_file_path.lower().split('.')[-1]
    if file_extension == "heic":
            heif_file = pyheif.read(temp_file_path)
            img_load = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            temp_png_path = temp_file_path.replace(".heic", ".png")
            img_load.save(temp_png_path, "PNG")
            temp_file_path = temp_png_path
    with Image.open(temp_file_path) as img_load:
        if img_load.format not in ["JPEG", "JPG", "PNG", "BMP", "GIF", "TIFF"]:
            raise HTTPException(status_code=400, detail=f"Unsupported image format/type {img_load.format}")
    logger.info(f"Image format/type: {img_load.format}")
    # GIF HANDLING
    if img_load.format == "GIF":
      temp_jpg_path = temp_file_path.replace(".gif", ".jpg")
      convert_gif_to_jpg(temp_file_path, temp_jpg_path)
      temp_file_path = temp_jpg_path  # Update path for further processing

    # Read the image with OpenCV
    img = cv2.imread(temp_file_path)
    if img is None:
      raise HTTPException(status_code=400, detail=f"Failed to load image for file {file.filename}")
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Resize the image by a factor of 2
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # Perform OCR on the processed image
    ocr_text = pytesseract.image_to_string(resized)
    ai_message = [
           {"role": "system", "content": "Du bist ein Steuerberater in Deutschland und versuchst Privatpersonen bei der Erstellung der Steuererklaerung zu unterstützen"},
           {"role": "user", "content": f"Correct this text, that was provided by ocr. Return only the corrected version without comments: {ocr_text}"}
    ]
    response = dochatgpt(ai_message, use_mini=True)
    return response.choices[0].message.content.strip()
  except UnidentifiedImageError:
    raise HTTPException(status_code=400, detail=f"Unsupported or unrecognized image format for file {file.filename}")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error processing image file {file.filename}: {e}")
  
def process_pdf (file, temp_file_path):
  try:
    with pdfplumber.open(temp_file_path) as pdf:
       content = ""
       for page in pdf.pages:
         text = page.extract_text()
         if text:
           content += text
           # logger.error(f"PDF Content: {content}")
         else:
        # If no text is found on the page, convert the page to image and perform OCR
        # Use convert_from_path to turn the PDF page into an image
            page_number = page.page_number
            images = convert_from_path(temp_file_path, first_page=page.page_number, last_page=page.page_number)
            for image in images:
              ocr_text = pytesseract.image_to_string(image)
              content += ocr_text
            logger.error(f"Before autocorrect: {content}")
            ai_message = [
              {"role": "system", "content": "Du bist ein Steuerberater in Deutschland und versuchst Privatpersonen bei der Erstellung der Steuererklaerung zu unterstützen"},
              {"role": "user", "content": f"Correct this text, that was provided by ocr. Return only the corrected version without comments: {content}"}
            ]
            response = dochatgpt(ai_message, use_mini=True)
            content = response.choices[0].message.content.strip()
            logger.error(f"After autocorrect: {content}")
    return content
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error processing PDF file {file.filename}: {e}")

def dochatgpt(ai_message, use_mini=False):
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_ORGANIZATION = os.getenv('OPENAI_ORGANIZATION')
    OPENAI_MODEL = 'gpt-4o-mini' if use_mini else os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORGANIZATION)
    if not OPENAI_API_KEY:
        raise ValueError("No OPENAI_API_KEY provided")

    # Set max_tokens based on model
    max_tokens = 4096 if OPENAI_MODEL == 'gpt-4-turbo' else 10000
    
    response = client.chat.completions.create(
            messages=ai_message,
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=max_tokens
    )
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    logger.error(f"Model: {OPENAI_MODEL}")
    logger.error(f"Input Tokens: {input_tokens}")
    logger.error(f"Output Tokens: {output_tokens}")
    logger.error(f"Total Tokens: {total_tokens}")
    return response
