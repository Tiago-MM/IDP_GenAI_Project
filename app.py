import base64
import streamlit as st
import os
import json
from PIL import Image
from dotenv import load_dotenv
from groq import Groq# Import local
from src.llm_engine import analyse_image
from src.utils import clean_json_output
from src.ocr_engine import process_with_easyocr, parse_ocr_with_llm

# --- INITIALISATION ET CONFIGURATION GLOBALE ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    layout="wide",
    # J'ai déplacé le titre dans le header pour la cohérence
    page_title="IDP GenAI Project", 
    initial_sidebar_state="expanded",
    menu_items={"About": "Extraction de documents avancée par IA"}
)

# --- INITIALISATION DU SESSION STATE ---
if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "active_view" not in st.session_state: 
    st.session_state.active_view = 'Image'
if "fullscreen_mode" not in st.session_state:
    st.session_state.fullscreen_mode = False

# --- VARIABLES ET CSS CUSTOM ---
PRIMARY_COLOR = "#333333"
SECONDARY_COLOR = "#999999"
SUCCESS_COLOR = "#609966"

st.markdown(f"""
    <style>
        /* 1. Rendre le HEADER plus fin */
        h1, h2 {{
            margin-top: 5px; /* Réduire l'espace en haut */
            padding-top: 0;
            margin-bottom: 5px; /* Réduire l'espace en bas */
        }}
        .logo-container {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 5px;
        }}
        .section-title {{
            color: {PRIMARY_COLOR};
            font-size: 18px;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 5px;
            border-left: 4px solid {SECONDARY_COLOR};
            padding-left: 10px;
        }}
        .info-box {{
            background-color: #F8F8F8;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {SECONDARY_COLOR};
            margin: 10px 0;
        }}
        .success-box {{
            background-color: #e6f7e6;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {SUCCESS_COLOR};
        }}
    </style>
""", unsafe_allow_html=True)

# --- HEADER (LOGO + TITRE) ---

# Remplacement du header par un conteneur logo (Exemple : Utiliser un titre simple)
st.markdown("""
    <div class="logo-container">
        <h2>IDP Project</h2>
    </div>
""", unsafe_allow_html=True)


# --- BARRE DE NAVIGATION (IMMÉDIATEMENT SOUS LE LOGO) ---
col_btn_img, col_btn_json, col_spacer = st.columns([1, 1, 6])

with col_btn_img:
    btn_style_img = "secondary" if st.session_state.active_view != 'Image' else "primary"
    if st.button("🖼️ Visualiser l'Image", type=btn_style_img, use_container_width=True, key='view_img'):
        st.session_state.active_view = 'Image'

with col_btn_json:
    btn_style_json = "secondary"
    if st.session_state.extraction_result:
        btn_style_json = "secondary" if st.session_state.active_view != 'JSON' else "primary"
        if st.button("🌲 Visualiser le JSON", type=btn_style_json, use_container_width=True, key='view_json'):
            st.session_state.active_view = 'JSON'
    else:
        st.button("🌲 Visualiser le JSON", disabled=True, use_container_width=True)

st.markdown("---")
# Fin de la barre de navigation

# --- UPLOADER ---
st.markdown("<div class='section-title'>📤 Télécharger votre document</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Sélectionnez une image de document",
    type=['png', 'jpg', 'jpeg'],
    help="Formats acceptés: PNG, JPG, JPEG (max 200MB)"
)

# Réinitialiser le résultat si on change de fichier
if uploaded_file and 'last_uploaded_file' in st.session_state:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.extraction_result = None
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.active_view = 'Image'
elif uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file.name

# --- SIDEBAR (Configuration + Contrôles) ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration du Modèle")
    model_choice = st.selectbox(
        "🤖 Modèle IA",
        ["meta-llama/llama-4-scout-17b-16e-instruct", "easyocr"],
        help="Sélectionnez le modèle à utiliser pour l'analyse"
    )
    st.info("ℹ️ L'extraction se fait en mode Auto-Schema.")
    
    st.divider()
    st.markdown("### 🖼️ Contrôles de Visualisation")
    zoom_level = st.slider("🔍 Zoom (%)", 50, 200, 100, 10, key="sidebar_zoom")

    st.divider()
    
    # --- LOGIQUE D'EXTRACTION ---
    if uploaded_file:
        if st.button("🚀 Extraire les données", type="primary", use_container_width=True, key="extract_btn"):
            st.session_state.extraction_result = None 
            st.session_state.active_view = 'JSON'

            st.info("⏳ Analyse en cours... Voir les résultats dans la vue 'JSON'.")

            try:
                uploaded_file.seek(0) 
                img_bytes = uploaded_file.read()
                encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                final_data='test ----'
                if model_choice == "easyocr":
                    with st.spinner("Lecture avec EasyOCR..."):
                        # Appel de la fonction
                        raw_text, structured_data = process_with_easyocr(img_bytes)
                        final_data = parse_ocr_with_llm(raw_text, client_groq=client)
                    
                else :
                    raw_result_generator = analyse_image(
                        image=encoded_image,
                        model=model_choice
                    )

                    full_raw_result = "".join([chunk.choices[0].delta.content or "" for chunk in raw_result_generator])
                    
                    # Parsing final
                    if isinstance(full_raw_result, str):
                        cleaned_str = clean_json_output(full_raw_result) 
                        try:
                            final_data = json.loads(cleaned_str)
                        except json.JSONDecodeError:
                            final_data = {"error": "JSON invalide", "raw": full_raw_result}
                    else:
                        final_data = full_raw_result

                # Sauvegarde et mise à jour de l'UI
                st.session_state.extraction_result = final_data
                st.rerun() 

            except Exception as e:
                st.error(f"❌ Erreur critique lors de l'extraction: {str(e)}")
    else:
        st.warning("⚠️ Téléchargez un fichier pour activer le bouton.")


# --- CONTENU PRINCIPAL : AFFICHAGE DÉTAILLÉ ---

if uploaded_file:
    uploaded_file.seek(0)
    image_pil = Image.open(uploaded_file)
    
    # 2. Affichage du contenu basé sur l'état
    
    # --- VUE IMAGE ---
    if st.session_state.active_view == 'Image':
        st.markdown("### Aperçu du Document (Vue Image)", unsafe_allow_html=True)
        st.markdown(f"**Fichier:** `{uploaded_file.name}`")
        st.markdown("---")
        st.image(image_pil, width=int(image_pil.width * zoom_level / 100), use_container_width=True) 
        
    # --- VUE JSON ---
    elif st.session_state.active_view == 'JSON' and st.session_state.extraction_result:
        result_data = st.session_state.extraction_result
        
        st.markdown("### Données Structurées (Vue JSON)", unsafe_allow_html=True)
        st.markdown("---")

        if "error" in result_data and "raw" in result_data:
            st.error("❌ Le modèle n'a pas renvoyé un JSON valide.")
            with st.expander("Réponse brute du modèle", expanded=True):
                st.code(result_data["raw"], language="json")
        else:
            st.markdown("""
                <div class="success-box">
                    <h4>✅ Extraction réussie!</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Exploration des Nœuds JSON")
            st.json(result_data, expanded=True) 

            # Bouton de téléchargement
            json_str = json.dumps(result_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Télécharger JSON",
                data=json_str,
                file_name=f"extract_{uploaded_file.name.split('.')[0]}.json",
                mime="application/json",
                use_container_width=True
            )