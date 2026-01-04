import easyocr
import numpy as np
from PIL import Image
import io
import json


reader = easyocr.Reader(['fr', 'en'], gpu=False)

# process image with EasyOCR
# image_bytes : bytes
# returns: full_text (str), json_output (list of dict)
def process_with_easyocr(image_bytes):

    try:
        results = reader.readtext(image_bytes)
        
        json_output = []
        full_text = ""

        for bbox, text, conf in results:
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



# parse OCR text with LLM to structured JSON
def parse_ocr_with_llm(ocr_text, client_groq=None, model="llama-3.1-8b-instant", schema_json=None):
    if schema_json :
        prompt = f"""
        Tu es un expert en extraction de données. Voici des données brutes issues d'un OCR :

        --- DONNÉES OCR ---
        {ocr_text}
        -------------------
        
        Ta mission :
        1. Analyse ce texte pour identifier les informations clés.
        2. Corrige les erreurs évidentes de l'OCR.
        3. Génère un objet JSON structuré contenant ces informations selon ce schéma : {schema_json}
        
        IMPORTANT : Réponds UNIQUEMENT avec le code JSON. Pas de phrases d'introduction, pas de markdown (```json). Juste le JSON brut.
        """
    else :
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
        # create json chat completion
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data extractor. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=0,
            stream=False,
            response_format={"type": "json_object"} # Ensure the response is JSON
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # if error in parsing
        return json.dumps({"error": f"Erreur Parsing LLM : {str(e)}"})