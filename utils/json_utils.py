import json

def safe_json_parse(response):
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        # Strip markdown if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}
    return {}
