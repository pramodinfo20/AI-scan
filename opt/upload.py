import os
from utils import load_wissen, process_image_for_ocr, process_pdf,dochatgpt, LimitUploadSizeMiddleware
from fastapi import FastAPI, APIRouter, File, Form, UploadFile, HTTPException
from typing import List, Union, Optional
import logging

# Initialize the FastAPI app
app = FastAPI()
# Add the middleware to limit upload size to 10 MB
app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=10 * 1024 * 1024)  # 10 MB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()

# Path to datenmodell.txt and load it
xmlstruktur = load_wissen("wissen/xmlstruktur.txt")
uploads = load_wissen("wissen/filetypes.txt")
max_words = 25000

# Function to call GPT with extracted data and knowledge base
def call_gpt_with_wissen(xmlstruktur, uploads, extracted_data, pre_data):
    content = f"In the List are possible IDs and DocumentTypes seperated by ':': {uploads}. Find out, which DocumentType was processed return the ID.\n\nHere is the extracted data from the uploads:\n\n{extracted_data}\n\nRespond ONLY the ID or NONE, if you cannot determine the DocumentType"
    message_init = [
           {"role": "system", "content": "Du bist ein Steuerberater in Deutschland und versuchst Privatpersonen bei der Erstellung der Steuererklaerung zu unterstützen"},
           {"role": "user", "content": f"{content}"}
           
   ]
    # logger.info({content})
    
    response1 = dochatgpt(message_init, use_mini=True)
    answer1 = response1.choices[0].message.content.strip()
    logger.info({answer1})

    # Verarbeite das relevante Datenmodell
    try:
      # Spezifisches Datenmodell
      if (answer1=="ste"):
         datenmodell = load_wissen(f"wissen/filetypes/jahresgehalt.txt")
         datenmodell += load_wissen(f"wissen/filetypes/phv.txt")
         datenmodell += load_wissen(f"wissen/filetypes/bank.txt")
      else:
        datenmodell = load_wissen(f"wissen/filetypes/{answer1}.txt")
    except Exception as e:
      # Allgemeines Datenmodell
      datenmodell = load_wissen("wissen/filetypes/NONE.txt")
    
    stammdaten = load_wissen("wissen/stammdaten.txt")  
    content2 = f"Using the following data models ('User Stammdaten' and 'User Zusatzdaten') to analyse the following data. 'User Stammdaten': {stammdaten} and 'User Zusatzdaten': {datenmodell}\n\nUse already exiting preData: {pre_data} to termine, whether the Data is for Steuerpflichtiger or Ehefrau. If datapoints from the model are missing, leave them out. Respond ONLY a valid XML in the following format (<f00> is only an example. Do not use it):\n\n{xmlstruktur}\n\nHere is the extracted data from the uploads:\n\n{extracted_data}"
    messages2 = [
           {"role": "system", "content": "Du bist ein Steuerberater in Deutschland und versuchst Privatpersonen bei der Erstellung der Steuererklaerung zu unterstützen"},
           {"role": "user", "content": f"{content2}"}
    ]
    # logger.info({content2})
    
    response2 = dochatgpt(messages2, use_mini=False)
    answer2 = response2.choices[0].message.content.strip()
    logger.info({answer2})
    token_usage = total_tokens = response1.usage.total_tokens + response2.usage.total_tokens
    chat_id = response2.id

    return answer2, token_usage, chat_id

# Process the uploaded file, extract data, and match it against the data model
@router.post("/")
async def upload_files(
   files: List[UploadFile] = File(..., allow_multiple=True),
   predata: Optional[str] = Form(None, description="Optional predata field")
):
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
            
            # Append this data to the combined data
            combined_extracted_data += extracted_data_with_filename + "\n\n"  # Separate each file's data with newlines for clarity
            # logger.info(f"Extracted Data: {extracted_data}")
            # logger.info(f"PreData: {predata}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {e}")
        
        os.remove(temp_file_path)

        # Limit Words
        words = combined_extracted_data.split()
        if len(words) > max_words:
          words = words[:max_words]
          combined_extracted_data = " ".join(words)
          logger.info(f"combined_extracted_data worde auf {max_words} limitiert")
          logger.info(f"Neue Daten {combined_extracted_data}")

        # Store the result for each file in case you want to return individual file details
        results.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "extracted_data": extracted_data_with_filename
        })

    # Send the combined extracted data to GPT with the knowledge base
    try:
        analysis, token_usage, chat_id = call_gpt_with_wissen(xmlstruktur, uploads,combined_extracted_data, predata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling GPT: {e}")

    # Return the combined analysis result and details of each file
    return {
        "files": results,
        "combined_analysis": analysis,
        "chat_id": chat_id,
        "token_usage": token_usage
    
    }
app.include_router(router)
