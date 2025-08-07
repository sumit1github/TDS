#!/usr/bin/env python3
"""
Test script for the Data Analyst Agent API
"""
import requests
import json
import sys

def test_api(file_path):
    url = "http://localhost:8000/api/"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response for {file_path}:")
            print(json.dumps(result, indent=2))
            print("\n" + "="*50 + "\n")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running.")
    except FileNotFoundError:
        print(f"Error: {file_path} file not found.")

def main():
    # Test both sample files
    test_files = [
        'test_question.txt',      # Wikipedia movie data
        'test_court_data.txt'     # Indian court judgment data
    ]
    
    if len(sys.argv) > 1:
        # Test specific file if provided
        test_api(sys.argv[1])
    else:
        # Test all sample files
        for test_file in test_files:
            test_api(test_file)

if __name__ == "__main__":
    main()
