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
            extracted_data = process_image_for_ocr(file, temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {e}")
        
        # Clean up the temporary file
        os.remove(temp_file_path)

        # Store the result for each file in case you want to return individual file details
        results.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "ocr_data": extracted_data
        })

    return {
        "files": results
    }

app.include_router(router)