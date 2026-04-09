import os
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from importlib import import_module

app = FastAPI()

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ORGANIZATION = os.getenv('OPENAI_ORGANIZATION')

# Ensure API key is provided
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY provided")

# Dynamically import all modules named 'embedding*.py' and include their routers
module_names = ['filetype','upload','ocr']  # List all your modules here
for module_name in module_names:
    module = import_module(module_name)
    if hasattr(module, 'router'):
      app.include_router(module.router, prefix=f"/{module_name}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
