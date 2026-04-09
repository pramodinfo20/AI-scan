import os
from utils import load_wissen, process_image_for_ocr, process_pdf, dochatgpt, LimitUploadSizeMiddleware
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from typing import List, Union
import logging

# Initialize the FastAPI app
app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()

# Add the middleware to limit upload size to 10 MB
app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=10 * 1024 * 1024)  # 10 MB

# Load FileTypes
filetypes = load_wissen("wissen/filetypes.txt")

# Function to call GPT with extracted data and knowledge base
def call_gpt_with_wissen(filetypes, extracted_data):
    content = f"{filetypes}. Find out, which DocumentType was processed. If the  DokumentType is not there, please write what you think it might be\n\nHere is the extracted data from the uploads:\n\n{extracted_data}"
    
    messages = [
           {"role": "system", "content": "Du bist ein Steuerberater in Deutschland und versuchst Privatpersonen bei der Erstellung der Steuererklaerung zu unterstützen"},
           {"role": "user", "content": f"{content}"}
           
    ]
    #logger.info({content})
    response = dochatgpt(messages)
    answer = response.choices[0].message.content.strip()
    token_usage = response.usage
    chat_id = response.id

    return answer, token_usage, chat_id

# Process the uploaded file, extract data, and match it against the data model
@router.post("/")
async def upload_files(files: List[UploadFile] = File(..., allow_multiple=True)):
# async def upload_files(files: Union[UploadFile, List[UploadFile]] = File(...)):
    # Ensure files is always treated as a list
    if isinstance(files, UploadFile):
        files = [files]

    combined_extracted_data = ""
    results = []

    for file in files:
        # Save the uploaded file temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Extract text from the file based on its type
        extracted_data = ""
        try:
            if file.filename.endswith(".pdf"):
              extracted_data = process_pdf(file, temp_file_path)
            elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
              extracted_data = process_image_for_ocr(file, temp_file_path)
            else:
              raise HTTPException(status_code=400, detail="Unsupported file type")
            
            extracted_data_with_filename = f"Filename: {file.filename}\n\n{extracted_data}"
            combined_extracted_data += extracted_data_with_filename + "\n\n"  # Separate each file's data with newlines for clarity
            # logger.info({extracted_data})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {e}")
        
        # Clean up the temporary file
        os.remove(temp_file_path)

        # Store the result for each file in case you want to return individual file details
        results.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "extracted_data": extracted_data_with_filename
        })

    # Send the combined extracted data to GPT with the knowledge base
    try:
        analysis, token_usage, chat_id = call_gpt_with_wissen(filetypes,combined_extracted_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling GPT: {e}")

    # Return the combined analysis result and details of each file
    return {
        "files": results,
        "combined_analysis": analysis,
        "chat_id": chat_id,
        "token_usage": {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens
        }
    }

app.include_router(router)