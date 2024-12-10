
import sys
import re
from collections import Counter
def is_uppercase_and_numeric(line):
    return bool(re.fullmatch(r'[A-Z0-9]+', line))

def clean_line(line):
    # Remove everything except alphanumeric characters and spaces
    return re.sub(r'[^a-zA-Z0-9]', '', line).strip()
def find_most_common(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Normalize lines (strip whitespace, ignore empty lines)
        cleaned_lines = [clean_line(line) for line in lines]
        # Count occurrences of each unique output
        line_counts = Counter(cleaned_lines)
        most_common = line_counts.most_common(30)  # Top 5 most common outputs
        # Find the most common lines
        return most_common 

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_outputs.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    most_common_dict = find_most_common(output_file)
    strings_detected = []
    for obj in most_common_dict:
        key = obj[0] 
        value = int(obj[1])
        if key ==  None or len(key) == 0:
            pass 
        elif is_uppercase_and_numeric(key) == True and value > 0 and key not in strings_detected:
            strings_detected.append(key)
        
    print(strings_detected) 


