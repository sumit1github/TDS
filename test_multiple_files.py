#!/usr/bin/env python3
"""
Test script to verify the multiple file upload functionality
"""
import requests
import tempfile
import os

# Create test files
def create_test_files():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create questions.txt
    questions_file = os.path.join(temp_dir, "questions.txt")
    with open(questions_file, "w") as f:
        f.write("1. What is the average value in the dataset?\n")
        f.write("2. What is the maximum value?\n")
        f.write("3. Create a simple visualization of the data.\n")
    
    # Create a sample CSV file
    data_file = os.path.join(temp_dir, "data.csv")
    with open(data_file, "w") as f:
        f.write("name,value,category\n")
        f.write("Item1,100,A\n")
        f.write("Item2,200,B\n")
        f.write("Item3,150,A\n")
        f.write("Item4,300,B\n")
        f.write("Item5,250,A\n")
    
    return questions_file, data_file, temp_dir

def test_api():
    """Test the API with multiple files"""
    questions_file, data_file, temp_dir = create_test_files()
    
    try:
        # Prepare files for upload
        files = []
        
        # Add questions.txt
        with open(questions_file, 'rb') as f:
            files.append(('files', ('questions.txt', f.read(), 'text/plain')))
        
        # Add data.csv
        with open(data_file, 'rb') as f:
            files.append(('files', ('data.csv', f.read(), 'text/csv')))
        
        # Make the request
        url = "http://localhost:8000/api/"
        
        print("Testing multiple file upload...")
        print(f"Files to upload: questions.txt, data.csv")
        
        # Note: This is just for testing the file structure
        # You would need to run your FastAPI server first
        print("To test this properly, run:")
        print("1. Start your FastAPI server: python main.py")
        print("2. Use curl to test:")
        print(f'curl "http://localhost:8000/api/" -F "files=@{questions_file}" -F "files=@{data_file}"')
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        print("Test files cleaned up.")

if __name__ == "__main__":
    test_api()
