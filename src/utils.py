import json
import re

import json
import re # <-- NÉCESSAIRE POUR LES EXPRESSIONS RÉGULIÈRES

def clean_json_output(raw_text):
    """
    Nettoie la réponse du LLM pour ne garder que le JSON valide.
    Retire les balises markdown ```json ... ``` ou ``` ... ```
    """
    # 1. Recherche du motif ```json ... ``` (non-greedy)
    # re.DOTALL permet de capturer les sauts de ligne (\s*)
    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    
    if match:
        # Retourne le contenu capturé dans le groupe 1, nettoyé des espaces
        return match.group(1).strip()
    
    # 2. Si le format est juste ``` ... ``` (sans 'json')
    match = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # 3. Si aucune balise n'est trouvée, on suppose que le texte est déjà du JSON pur
    return raw_text.strip()
