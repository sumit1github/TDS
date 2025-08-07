# Data Analyst Agent API

A FastAPI-based service that uses LLMs to source, prepare, analyze, and visualize data based on natural language task descriptions.

## Features

- **Data Sourcing**: Scrapes Wikipedia tables, queries DuckDB/S3 data
- **Data Preparation**: Cleans and transforms data automatically
- **Analysis**: Performs statistical analysis, correlations, filtering
- **Visualization**: Generates plots and returns as base64-encoded images
- **LLM Integration**: Uses Groq's fast inference for natural language task interpretation

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Set Groq API key for enhanced task interpretation:
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   ```

## Usage

1. Start the API server:
   ```bash
   python main.py
   ```
   The server will run on `http://localhost:8000`

2. Send POST requests with task descriptions:
   ```bash
   curl -X POST "http://localhost:8000/api/" -F "file=@question.txt"
   ```

3. Test with the included sample:
   ```bash
   python test_api.py
   ```

## Supported Task Types

### Wikipedia Data Analysis
- Scrapes tables from Wikipedia URLs
- Analyzes movie data, financial data, etc.
- Returns JSON arrays with answers

### DuckDB/S3 Data Analysis  
- Executes SQL queries on cloud data
- Analyzes court judgments, large datasets
- Returns JSON objects with answers

### Visualizations
- Creates scatterplots with regression lines
- Returns base64-encoded PNG images
- Keeps file sizes under 100KB

## Example Responses

**Movie Analysis:**
```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

**Court Data Analysis:**
```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "Delhi High Court",
  "What's the regression slope...": 0.5,
  "Plot the year and # of days...": "data:image/png;base64,..."
}
```

## API Endpoints

- `POST /api/` - Main analysis endpoint
  - Accepts: multipart/form-data with a text file
  - Returns: JSON with analysis results

## Project Structure

- `main.py` - FastAPI application
- `agents.py` - Core analysis agents
- `requirements.txt` - Dependencies
- `test_question.txt` - Sample task
- `test_api.py` - API test script


# I am usign uv

## to install all pachanges
uv sync

# to start server

source .venv/bin/activate