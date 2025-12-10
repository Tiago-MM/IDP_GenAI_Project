import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv() # Charge le fichier .env en mémoire
API_KEY = os.getenv("GROQ_API_KEY")


def analyze_image(image_bytes, prompt, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    client = Groq(api_key=API_KEY)
    
    # Groq demande l'image en base64 (petit changement technique)
    import base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{encoded_image}"

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            model=model,
            temperature=0,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Erreur API Groq : {str(e)}"