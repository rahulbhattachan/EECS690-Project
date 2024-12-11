
import sys
import re
from collections import Counter
def is_uppercase_and_numeric(line):
    return bool(re.fullmatch(r'[A-Z0-9]+', line))
def find_word_with_most_longest_substrings(words):
    """
    Finds the word that contains the most of the longest substrings from the input list.

    Args:
        words (list): List of input strings.

    Returns:
        dict: Dictionary with 'longest_substrings' and 'matching_words' keys.
    """
    substrings = Counter()
    
    # Generate all substrings and count their occurrences
    for word in words:
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                substrings[word[i:j]] += 1
    
    # Find the longest substrings
    max_length = max(len(sub) for sub in substrings.keys())
    longest_substrings = [sub for sub in substrings if len(sub) == max_length]
    
    # Determine which words contain the most of the longest substrings
    word_counts = Counter()
    for word in words:
        count = sum(1 for sub in longest_substrings if sub in word)
        word_counts[word] = count
    
    # Get words containing the most longest substrings with count > 2
    matching_words = [word for word, count in word_counts.items() if count > 1]
    
    return {
        "longest_substrings": longest_substrings,
        "matching_words": matching_words
    }

def clean_line(line):
    # Remove everything except alphanumeric characters and spaces
    return "".join(re.findall(r'\b[A-Z0-9]+\b', line))
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
        if value > 0 and key not in strings_detected and key != '' and key != None:
            strings_detected.append(key)            
    # outlist = []
    # out = find_word_with_most_longest_substrings(strings_detected)
    # print(out)
    # outlist.append(out)
    # print(outlist)
    while "IC" in strings_detected or "I" in strings_detected:
        try:
            strings_detected.remove("IC")
            strings_detected.remove("I")
            strings_detected.remove("ICIC")
        except:
            pass
    for index, string in enumerate(strings_detected):
        if len(string) <= 2:
            strings_detected.remove(string)
        elif ("II" in string or "IC" in string or "CI" in string or "MD" in string) and len(string) <= 4:
            del strings_detected[index]
        elif "HTML" in string:
            del strings_detected[index]
            
    print(strings_detected)


