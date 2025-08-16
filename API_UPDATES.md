# API Updates for Multiple File Upload Support

## Changes Made

### 1. Updated FastAPI Endpoint (`main.py`)

**Key Changes:**
- Modified the endpoint to accept `List[UploadFile]` instead of single file
- Added logic to identify the required `questions.txt` file
- Added processing for additional optional files
- Implemented 3-minute timeout using `asyncio.wait_for()`
- Added proper error handling for missing `questions.txt`

**New API Format:**
```bash
curl "https://app.example.com/api/" -F "files=@questions.txt" -F "files=@data.csv" -F "files=@image.png"
```

### 2. Updated Data Processing (`agents.py`)

**Key Changes:**
- Modified `DataAnalystOrchestrator.process_task()` to accept `additional_files` parameter
- Added file type detection and processing:
  - CSV files: Parsed into pandas DataFrames
  - Text files: Decoded as UTF-8 strings
  - Images: Converted to base64 data URIs
  - Other files: Stored as raw binary data
- Prioritized CSV files from uploads over web scraping when available
- Maintained backward compatibility with existing functionality

### 3. Requirements Compliance

✅ **Endpoint accepts POST requests**
✅ **Handles multiple files via form data**
✅ **questions.txt is always required**
✅ **Additional files are optional**
✅ **3-minute timeout implemented**
✅ **Maintains existing data analysis capabilities**

## File Processing Logic

1. **questions.txt** (Required): Contains the analysis questions/tasks
2. **CSV files**: Automatically parsed and used as primary data source
3. **Image files**: Converted to base64 for potential analysis
4. **Text files**: Decoded and stored for reference
5. **Other files**: Stored as binary data

## Testing

Use the provided test scripts:
- `test_api_curl.sh`: Bash script with curl examples
- `test_multiple_files.py`: Python test script

## Backward Compatibility

The changes maintain full backward compatibility:
- Single file uploads still work
- Existing analysis logic unchanged
- All previous features preserved

## Error Handling

- Returns 400 if `questions.txt` is missing
- Returns 408 if processing takes longer than 3 minutes
- Returns 500 for other processing errors
- Graceful fallbacks for file parsing errors
