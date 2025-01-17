def extract_code(text, extension="cpp"):
    """
    Extract code between ```cpp and ``` markers from a string.
    
    Args:
        text (str): Input string containing code block
        
    Returns:
        str: Extracted code content, or empty string if no matching block found
    """
    start_marker = f"```{extension}"
    end_marker = "```"
    
    # Find start and end positions
    start_pos = text.find(start_marker)
    if start_pos == -1:
        # model didn't generate wrapper
        return text
        
    # Add length of start marker to get to start of code
    start_pos += len(start_marker)
    
    # Find end marker after the start position
    end_pos = text.find(end_marker, start_pos)
    if end_pos == -1:
        end_pos = len(text)
    
    # Extract the code between markers
    code = text[start_pos:end_pos].strip()
    return code