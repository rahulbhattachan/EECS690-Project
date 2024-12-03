
import sys
import re
from collections import Counter
def clean_line(line):
    # Remove everything except alphanumeric characters and spaces
    return re.sub(r'[^a-zA-Z0-9\s]', '', line).strip()
def find_most_common(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Normalize lines (strip whitespace, ignore empty lines)
        cleaned_lines = [clean_line(line) for line in lines]
        print(cleaned_lines)
        # Count occurrences of each unique output
        line_counts = Counter(cleaned_lines)
        most_common = line_counts.most_common(5)  # Top 5 most common outputs
        # Find the most common lines
        print(most_common)
        for line, count in most_common:
            if count > 1:
                print(f"Possible Content:\n{line}\n")
            

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_outputs.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    find_most_common(output_file)

