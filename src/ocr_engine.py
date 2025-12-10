import easyocr
import numpy as np
from PIL import Image
import io

# On charge le Reader une seule fois (Variable globale ou via st.cache_resource)
# 'fr' pour français, 'en' pour anglais
reader = easyocr.Reader(['fr', 'en'], gpu=False) # Mettez gpu=True si vous en avez un

def process_with_easyocr(image_bytes):
    """
    Extrait le texte et les positions avec EasyOCR.
    Retourne une liste de dictionnaires compatible JSON.
    """
    try:
        # EasyOCR accepte directement les bytes dans readtext
        # Mais passer par PIL -> bytes est plus sûr pour les formats exotiques
        results = reader.readtext(image_bytes)
        
        json_output = []
        full_text = ""

        for bbox, text, conf in results:
            # Nettoyage des types Numpy (int32/int64) qui font planter json.dumps
            # bbox est une liste de 4 points [ [x,y], [x,y], ... ]
            clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
            
            json_output.append({
                "text": text,
                "confidence": float(conf),
                "bbox": clean_bbox
            })
            full_text += text + " "

        return full_text, json_output

    except Exception as e:
        return f"Erreur EasyOCR : {str(e)}", []

# Ajoutez cet import si besoin
import json

import json

def parse_ocr_with_llm(ocr_text, client_groq=None, model="llama-3.1-8b-instant"):
    """
    Prend le texte brut d'EasyOCR et le transforme en JSON strict via un LLM.
    """
    prompt = f"""
    Tu es un expert en extraction de données. Voici des données brutes issues d'un OCR :

    --- DONNÉES OCR ---
    {ocr_text}
    -------------------
    
    Ta mission :
    1. Analyse ce texte pour identifier les informations clés.
    2. Corrige les erreurs évidentes de l'OCR.
    3. Génère un objet JSON structuré contenant ces informations.
    
    IMPORTANT : Réponds UNIQUEMENT avec le code JSON. Pas de phrases d'introduction, pas de markdown (```json). Juste le JSON brut.
    """
    
    try:
        # On utilise Groq avec le paramètre response_format pour garantir le JSON
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data extractor. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=0,
            stream=False,
            response_format={"type": "json_object"} # <--- C'est ça qui force le JSON pur
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # En cas d'erreur, on renvoie un JSON d'erreur pour ne pas casser l'app
        return json.dumps({"error": f"Erreur Parsing LLM : {str(e)}"})