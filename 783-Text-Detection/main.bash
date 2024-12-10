#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <photo_path> <output_directory> <typescript_program> <python_image_processor_program> <python_analyzer_program>"
    exit 1
fi

# Assign arguments to variables
photo_path="$1"
output_directory="$2"
typescript_program="$3"
python_program="$4"
analyzer_program="$5"

# Ensure the output directory exists
if [ ! -d "$output_directory" ]; then
    echo "Error: Output directory does not exist."
    exit 1
fi

# Run the Python program
python3 $python_program "$photo_path"

# Check if Python program succeeded
if [ $? -ne 0 ]; then
    echo "Error: Python program failed to process the photo."
    exit 1
fi

# Create or clear the output file
output_file="$output_directory/output.txt"
: > "$output_file" # Truncate the file if it exists or create it

# Iterate over the files in the output directory, excluding .txt files
for file in "$output_directory"/*; do
    # Skip if the file is a .txt file
    if [[ "$file" == *.txt ]]; then
        continue
    fi

    # Check if the file is valid
    if [ -f "$file" ]; then
        # Execute the TypeScript program with the file as argv[1]
        output=$(npx tsx "$typescript_program" "$file")
        # Check if the TypeScript program succeeded
        if [ $? -ne 0 ]; then
            echo "Error: TypeScript program failed to send file $file."
        else
            # echo "Successfully processed file: $file"
            # Append the output to the output file
            echo "$output" >> "$output_file"
			echo "---" >> "$output_file"
        fi
    fi
done

# Run analysis on the output file
python3 $analyzer_program "$output_file"

