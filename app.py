import streamlit as st
import os
import json
from PIL import Image
import io

# Import local (nécessite que le dossier src contienne __init__.py)
from src.llm_engine import analyze_image
from src.utils import clean_json_output

st.set_page_config(layout="wide", page_title="IDP GenAI Project")

st.title("📄 Intelligent Document Processing")
st.markdown("Extraction structurée avec **Llama 3.2 Vision**")

# --- Sidebar : Chargement des schémas ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Modèle", ["llama3.2-vision", "llava"])

# Lecture automatique des schémas disponibles
schema_dir = "schemas"
schema_files = [f for f in os.listdir(schema_dir) if f.endswith('.json')]
selected_schema_file = st.sidebar.selectbox("Type de document", schema_files)

# Chargement du contenu du schéma sélectionné
if selected_schema_file:
    with open(os.path.join(schema_dir, selected_schema_file), 'r') as f:
        schema_content = f.read()
    
    st.sidebar.subheader("Schéma cible")
    target_schema = st.sidebar.text_area("JSON Schema", schema_content, height=200)

# --- Main Area ---
uploaded_file = st.file_uploader("Image du document", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Document scanné", use_column_width=True)
    
    with col2:
        if st.button("Extraire les données", type="primary"):
            with st.spinner("Analyse visuelle en cours..."):
                # Conversion image pour l'API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_bytes = img_byte_arr.getvalue()
                
                # Construction du prompt
                prompt = f"Analyze this image. Extract data strictly following this JSON schema: {target_schema}. Return ONLY JSON."
                
                # Appel Backend
                raw_result = analyze_image(img_bytes, prompt, model=model_choice)
                
                # Parsing
                final_data = clean_json_output(raw_result)
                
                if "error" in final_data:
                    st.error("Erreur de parsing")
                    st.code(raw_result)
                else:
                    st.success("Succès !")
                    st.json(final_data)
