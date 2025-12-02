#!/bin/bash

# Define the output directory
OUTPUT_DIR="courseware"

# Create the output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Loop through all PDF files in the current directory
for pdf_file in *.pdf; do
    # Check if any PDF files exist
    if [ ! -e "$pdf_file" ]; then
        echo "No PDF files found in the current directory."
        exit 0
    fi

    # Extract the filename without extension
    filename=$(basename -- "$pdf_file")
    filename_no_ext="${filename%.*}"
    
    # Define output text file path
    output_txt="$OUTPUT_DIR/${filename_no_ext}.txt"

    echo "Converting '$pdf_file' to '$output_txt'..."

    # Convert PDF to text using pdftotext
    # -layout: Maintain original physical layout (good for RAG context)
    pdftotext -layout "$pdf_file" "$output_txt"

    if [ $? -eq 0 ]; then
        echo "Successfully converted: $filename"
    else
        echo "Error converting: $filename"
    fi
done

echo "Conversion complete."
