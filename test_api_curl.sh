#!/bin/bash

# Test script for the updated API with multiple file uploads
# This script demonstrates the new API format as specified in the requirements

echo "Creating test files..."

# Create questions.txt (always required)
cat > questions.txt << 'EOF'
1. What is the total number of rows in the dataset?
2. What are the unique categories in the data?
3. What is the average value by category?
4. Create a visualization showing the distribution of values by category.
EOF

# Create a sample CSV data file
cat > data.csv << 'EOF'
name,value,category
Product A,100,Electronics
Product B,200,Clothing
Product C,150,Electronics
Product D,300,Clothing
Product E,250,Electronics
Product F,180,Clothing
Product G,220,Electronics
Product H,400,Clothing
EOF

# Create a sample image file (placeholder)
echo "Creating placeholder image..."
echo "This would be a real image file" > image.png

echo "Files created successfully!"
echo ""
echo "To test the API, first start your FastAPI server:"
echo "  python main.py"
echo ""
echo "Then run one of these curl commands:"
echo ""
echo "1. Test with questions.txt only:"
echo "curl \"http://localhost:8000/api/\" -F \"files=@questions.txt\""
echo ""
echo "2. Test with questions.txt and data.csv:"
echo "curl \"http://localhost:8000/api/\" -F \"files=@questions.txt\" -F \"files=@data.csv\""
echo ""
echo "3. Test with all files (as per requirements example):"
echo "curl \"http://localhost:8000/api/\" -F \"files=@questions.txt\" -F \"files=@data.csv\" -F \"files=@image.png\""
echo ""
echo "Note: The API now expects:"
echo "- questions.txt is ALWAYS required"
echo "- Additional files are optional"
echo "- All responses must be returned within 3 minutes"
echo ""
echo "Cleanup:"
echo "To remove test files, run: rm questions.txt data.csv image.png"
