from groq import Groq
import os
import base64
import time

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

# --- FONCTION POUR ENCODER L'IMAGE ---
def encode_image(image_path):
    """Encodes a local file to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Erreur: Le fichier n'a pas été trouvé à l'emplacement: {image_path}")
        return None
    except Exception as e:
        print(f"Erreur lors de l'encodage de l'image: {e}")
        return None

# --- VÉRIFICATION ET ENCODAGE ---
def check_file_exists(file_path):
    """Vérifie si le fichier existe à l'emplacement donné."""
    return os.path.isfile(file_path)

def generate_base64_from_local(file_path):
    """Génère une chaîne Base64 à partir d'un fichier local."""
    if not check_file_exists(file_path):
        print(f"Erreur: Le fichier n'existe pas à l'emplacement: {file_path}")
        return None
    return encode_image(file_path)

def analyse_image(image, model=VISION_MODEL, GROQ_API_KEY=None,schema_json=None):
    print(f"Modèle utilisé pour l'analyse: {model}")
    base64_image = image

    if not base64_image:
        exit()

    # Format Base64 obligatoire pour l'API
    base64_url = f"data:image/jpeg;base64,{base64_image}"

    # --- APPEL À L'API GROQ ---
    client = Groq(api_key=GROQ_API_KEY)

    # Note: Je remplace le modèle par le modèle Vision Groq supporté (Llama 3.2 Vision)
    # Si vous avez un accès spécial, vous pouvez remettre votre nom de modèle.
    
    start_time = time.time()
    print(f"Envoi du fichier local à Groq...")

    if schema_json:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"I want you to generate a Json file of the document with no other text. Just a json file according to this schema: {schema_json} . Json file : "
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # Utilisation de la chaîne Base64 encodée
                                "url": base64_url
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

    else :
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I want you to generate a Json file of the document with no other text. Just a json file. Json file : "
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # Utilisation de la chaîne Base64 encodée
                                "url": base64_url
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

    end_time = time.time()
    print(f"\n\nTemps d'exécution total: {end_time - start_time:.2f} secondes")
    
    return completion
