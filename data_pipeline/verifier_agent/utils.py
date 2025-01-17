import json
import re
COLOR_BOOK = {
    "tool_use_input": 96,
    "tool_use_output": 92,
    "agent_response": 95,
    "tool_use_command":90
}
def print_colored(text, color_code):
    """
    Prints the given text in the specified color.

    Parameters:
    text (str): The string to be printed.
    color_code (str): The ANSI color code.

    Example color codes:
    '30' - Black
    '31' - Red
    '32' - Green
    '33' - Yellow
    '34' - Blue
    '35' - Magenta
    '36' - Cyan
    '37' - White
    '90' - Bright Black (Gray)
    '91' - Bright Red
    '92' - Bright Green
    '93' - Bright Yellow
    '94' - Bright Blue
    '95' - Bright Magenta
    '96' - Bright Cyan
    '97' - Bright White
    """
    print(f"\033[{color_code}m{text}\033[0m")

def extract_json(text):
    """
    Extracts a JSON dictionary from a text string that might contain other content.
    
    Args:
        text (str): Input text containing a JSON dictionary
        
    Returns:
        dict: Extracted JSON dictionary
        
    Raises:
        ValueError: If no valid JSON dictionary is found in the text
    """
    # Find text that looks like a JSON object starting and ending with curly braces
    # This simpler pattern will work for most cases where JSON is well-formed
    matches = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    
    # Try each potential JSON match
    for match in matches:
        try:
            json_str = match.group()
            json_dict = json.loads(json_str)
            # If we successfully parsed it and it's a dictionary, return it
            if isinstance(json_dict, dict):
                return json_dict
        except json.JSONDecodeError:
            continue
    
    raise ValueError("No valid JSON dictionary found in the text")

def process_text_with_json(text):
    """
    Example function showing how to use the JSON extractor with error handling
    """
    try:
        json_data = extract_json(text)
        return json_data
    except ValueError as e:
        print(f"Error: {e}")
        return None