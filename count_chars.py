import json

def count_characters_and_spaces(file_path):
    """Count total characters and spaces in a JSON file"""
    try:
        # Read the entire file as text (not parsing as JSON)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count total characters
        total_chars = len(content)
        
        # Count spaces (including all whitespace characters)
        space_count = content.count(' ')
        
        # Count all whitespace characters (spaces, tabs, newlines)
        all_whitespace = sum(1 for char in content if char.isspace())
        
        return {
            'total_characters': total_chars,
            'spaces_only': space_count,
            'all_whitespace': all_whitespace,
            'non_whitespace': total_chars - all_whitespace
        }
    
    except Exception as e:
        return {'error': str(e)}

# Count characters in the JSON file
file_path = r"c:\Users\UK-PC\Desktop\sweden_legal_rag\titles_only.json"
result = count_characters_and_spaces(file_path)

print("Character Count Analysis for titles_only.json:")
print("=" * 50)
if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print(f"Total characters: {result['total_characters']:,}")
    print(f"Spaces only: {result['spaces_only']:,}")
    print(f"All whitespace (spaces, tabs, newlines): {result['all_whitespace']:,}")
    print(f"Non-whitespace characters: {result['non_whitespace']:,}")