from groq import Groq
import os
import base64

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

#Encodes a local file to a Base64 string
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Erreur: Le fichier n'a pas été trouvé à l'emplacement: {image_path}")
        return None
    except Exception as e:
        print(f"Erreur lors de l'encodage de l'image: {e}")
        return None

# check if file exists
def check_file_exists(file_path):
    return os.path.isfile(file_path)

# generate base64 from local file
def generate_base64_from_local(file_path):
    if not check_file_exists(file_path):
        print(f"Erreur: Le fichier n'existe pas à l'emplacement: {file_path}")
        return None
    return encode_image(file_path)

def analyse_image(image, model=VISION_MODEL, GROQ_API_KEY=None,schema_json=None):
    print(f"Modèle utilisé pour l'analyse: {model}")
    base64_image = image

    if not base64_image:
        exit()

    base64_url = f"data:image/jpeg;base64,{base64_image}"

    #groq client
    client = Groq(api_key=GROQ_API_KEY)

    
    print(f"Envoi du fichier local à Groq...")

    # case with schema
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

    # case without schema
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
    
    return completion
