import json
import re

def extract_json(raw: str):
    """Robust JSON extractor — handles markdown code fences and prefix text."""
    if not isinstance(raw, str):
        return None
        
    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback: search for the first { and last }
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None
