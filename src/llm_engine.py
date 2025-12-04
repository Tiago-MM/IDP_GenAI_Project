import ollama

def analyze_image(image_bytes, prompt, model="llama3.2-vision"):
    """
    Envoie l'image et le prompt au modèle local via Ollama.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"
