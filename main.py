from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import os
from typing import List, Optional
from dotenv import load_dotenv
from agents import DataAnalystOrchestrator

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.get("/api/")
async def analyze_task(files: List[UploadFile] = File(...)):
    try:
        # Find the questions.txt file (required)
        questions_file = None
        additional_files = []
        
        for file in files:
            if file.filename and file.filename.endswith('questions.txt'):
                questions_file = file
            else:
                additional_files.append(file)
        
        if not questions_file:
            raise HTTPException(status_code=400, detail="questions.txt file is required")
        
        # Read the questions file
        questions_content = await questions_file.read()
        task_description = questions_content.decode('utf-8')
        
        # Process additional files if any
        additional_data = {}
        for file in additional_files:
            if file.filename:
                file_content = await file.read()
                # Store file content with filename as key
                additional_data[file.filename] = {
                    'content': file_content,
                    'content_type': file.content_type or 'application/octet-stream'
                }
        
        # Get Groq API key from environment variable
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        # Initialize the orchestrator
        orchestrator = DataAnalystOrchestrator(groq_api_key)
        
        # Process the task with additional files - with timeout
        try:
            result = await asyncio.wait_for(
                orchestrator.process_task(task_description, additional_data),
                timeout=180.0  # 3 minutes
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout - analysis took longer than 3 minutes")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
