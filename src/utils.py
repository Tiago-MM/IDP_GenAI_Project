import json
import re

import json
import re 

def clean_json_output(raw_text):

    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    match = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return raw_text.strip()
