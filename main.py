from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import os
from dotenv import load_dotenv
from agents import DataAnalystOrchestrator

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.post("/api/")
async def analyze_task(file: UploadFile = File(...)):
    try:
        # Read the uploaded .txt file
        task_text = await file.read()
        task_description = task_text.decode('utf-8')
        
        # Get Groq API key from environment variable
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        # Initialize the orchestrator
        orchestrator = DataAnalystOrchestrator(groq_api_key)
        
        # Process the task
        result = await orchestrator.process_task(task_description)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
