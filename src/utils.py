import json
import re

def clean_json_output(raw_text):
    """
    Nettoie la réponse du LLM pour ne garder que le JSON valide.
    Retire les balises markdown ```json ... ```
    """
    try:
        # Si le modèle renvoie du markdown
        if "```" in raw_text:
            pattern = r"```json(.*?)```"
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                raw_text = match.group(1)
            else:
                # Essayer de nettoyer les balises génériques
                raw_text = raw_text.replace("```json", "").replace("```", "")
        
        return json.loads(raw_text.strip())
    except Exception as e:
        return {"error": "Parsing failed", "raw_text": raw_text}
