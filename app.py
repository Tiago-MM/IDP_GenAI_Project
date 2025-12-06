import base64
import streamlit as st
import os
import json
from PIL import Image
from dotenv import load_dotenv

# Import local
from src.llm_engine import analyse_image
from src.utils import clean_json_output

# --- INITIALISATION ET CONFIGURATION GLOBALE ---
load_dotenv()

st.set_page_config(
    layout="wide",
    page_title="IDP GenAI Project",
    initial_sidebar_state="expanded",
    menu_items={"About": "Extraction de documents avancée par IA"}
)

# --- INITIALISATION DU SESSION STATE ---
if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "active_view" not in st.session_state: # Nouvel état pour la vue active
    st.session_state.active_view = 'Image'
if "fullscreen_mode" not in st.session_state:
    st.session_state.fullscreen_mode = False

# --- VARIABLES ET CSS CUSTOM (Nettoyage des classes non utilisées) ---
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"
SUCCESS_COLOR = "#28a745"

st.markdown(f"""
    <style>
        /* La classe .main-header a été retirée */
        .section-title {{
            color: {PRIMARY_COLOR};
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
            border-left: 4px solid {PRIMARY_COLOR};
            padding-left: 10px;
        }}
        .info-box {{
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {PRIMARY_COLOR};
            margin: 10px 0;
        }}
        .success-box {{
            background-color: #d4edda;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {SUCCESS_COLOR};
        }}
        /* Marge pour le titre principal */
        h1 {{
            margin-top: 0;
        }}
    </style>
""", unsafe_allow_html=True)

# Header simplifié (pas de banner)
st.title("📄 Intelligent Document Processing")

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
        st.session_state.active_view = 'Image' # Revenir à la vue Image
elif uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file.name

# --- SIDEBAR (Configuration + Contrôles) ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration du Modèle")
    model_choice = st.selectbox(
        "🤖 Modèle IA",
        ["meta-llama/llama-4-scout-17b-16e-instruct"],
        help="Sélectionnez le modèle à utiliser pour l'analyse"
    )
    st.info("ℹ️ L'extraction se fait en mode Auto-Schema.")
    
    st.divider()
    st.markdown("### 🖼️ Contrôles de Visualisation")
    zoom_level = st.slider("🔍 Zoom (%)", 50, 200, 100, 10, key="sidebar_zoom")

    st.divider()
    
    # --- LOGIQUE D'EXTRACTION (Déplacée dans la sidebar) ---
    if uploaded_file:
        if st.button("🚀 Extraire les données", type="primary", use_container_width=True, key="extract_btn"):
            st.session_state.extraction_result = None 
            st.session_state.active_view = 'JSON' # Passer à la vue JSON après l'extraction

            st.info("⏳ Analyse en cours... Voir les résultats dans l'onglet 'JSON'.")

            try:
                uploaded_file.seek(0) 
                img_bytes = uploaded_file.read()
                encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                
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


# --- CONTENU PRINCIPAL : NAVIGATION PAR BOUTONS ---

if uploaded_file:
    uploaded_file.seek(0)
    image_pil = Image.open(uploaded_file)
    
    st.divider()
    st.markdown("## Vue Détaillée")

    # 1. Barre de Boutons (Header de Navigation)
    col_btn_img, col_btn_json, col_spacer = st.columns([1, 1, 6])

    with col_btn_img:
        # Style pour le bouton actif
        btn_style_img = "secondary" if st.session_state.active_view != 'Image' else "primary"
        if st.button("🖼️ Visualiser l'Image", type=btn_style_img, use_container_width=True, key='view_img'):
            st.session_state.active_view = 'Image'

    with col_btn_json:
        # Le bouton JSON ne peut être cliqué que si le résultat est là
        btn_style_json = "secondary" if st.session_state.active_view != 'JSON' and st.session_state.extraction_result else "primary"
        
        if st.session_state.extraction_result:
            if st.button("🌲 Visualiser le JSON", type=btn_style_json, use_container_width=True, key='view_json'):
                st.session_state.active_view = 'JSON'
        else:
            st.button("🌲 Visualiser le JSON", disabled=True, use_container_width=True) # Bouton désactivé
    
    st.markdown("---")


    # 2. Affichage du contenu basé sur l'état
    
    # --- VUE IMAGE ---
    if st.session_state.active_view == 'Image':
        st.markdown("### Aperçu du Document (Zoomable)", unsafe_allow_html=True)
        # Utilisation du zoom de la sidebar
        st.image(image_pil, width=int(image_pil.width * zoom_level / 100), use_container_width=True) 
        
    # --- VUE JSON ---
    elif st.session_state.active_view == 'JSON' and st.session_state.extraction_result:
        result_data = st.session_state.extraction_result
        
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