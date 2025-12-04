import streamlit as st
import os
import json
from PIL import Image
import io

# Import local (nécessite que le dossier src contienne __init__.py)
from src.llm_engine import analyze_image
from src.utils import clean_json_output

# Configuration page
st.set_page_config(
    layout="wide",
    page_title="IDP GenAI Project",
    initial_sidebar_state="expanded",
    menu_items={"About": "Intelligent Document Processing with Llama 3.2 Vision"}
)

# Custom CSS pour plus de style
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
        .section-title {
            color: #667eea;
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
            padding-left: 10px;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>📄 Intelligent Document Processing</h1>
        <p>Extraction structurée de données avec <b>Llama 3.2 Vision</b></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_choice = st.selectbox(
        "🤖 Modèle IA",
        ["llama3.2-vision", "llava"],
        help="Sélectionnez le modèle à utiliser pour l'analyse"
    )
    
    st.divider()
    
    st.markdown("### 📋 Schéma du document")
    
    # Lecture automatique des schémas disponibles
    schema_dir = "schemas"
    schema_files = [f for f in os.listdir(schema_dir) if f.endswith('.json')]
    
    selected_schema_file = st.selectbox(
        "Type de document",
        schema_files,
        help="Choisissez le schéma JSON correspondant à votre document"
    )
    
    # Chargement du contenu du schéma sélectionné
    target_schema = None
    if selected_schema_file:
        with open(os.path.join(schema_dir, selected_schema_file), 'r') as f:
            schema_content = f.read()
        
        target_schema = st.text_area(
            "Schéma JSON",
            schema_content,
            height=200,
            key="schema_input",
            help="Le schéma utilisé pour extraire et structurer les données"
        )

# Main content area
st.markdown("<div class='section-title'>📤 Télécharger votre document</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Sélectionnez une image de document",
    type=['png', 'jpg', 'jpeg'],
    help="Formats acceptés: PNG, JPG, JPEG (max 200MB)"
)

if uploaded_file and target_schema:
    # Affichage en deux colonnes
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='section-title'>👁️ Aperçu du document</div>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        
        # Options de zoom et plein écran
        zoom_cols = st.columns([1, 1, 2])
        with zoom_cols[0]:
            zoom_level = st.slider("🔍 Zoom", 50, 200, 100, 10, help="Ajustez le niveau de zoom")
        with zoom_cols[1]:
            if st.button("⛶ Plein écran", use_container_width=True):
                st.session_state.fullscreen_mode = True
        
        # Mode plein écran
        if st.session_state.get('fullscreen_mode', False):
            st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px;">
            """, unsafe_allow_html=True)
            
            col_back, col_zoom_fs = st.columns([2, 1])
            with col_back:
                if st.button("← Retour", use_container_width=True):
                    st.session_state.fullscreen_mode = False
                    st.rerun()
            with col_zoom_fs:
                zoom_fs = st.slider("Zoom plein écran", 50, 300, 150, 10)
            
            st.image(image, width=int(image.width * zoom_fs / 100), use_column_width=False)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Affichage normal avec zoom
            st.image(image, width=int(image.width * zoom_level / 100), use_column_width=False)
            st.markdown(f"**Fichier:** `{uploaded_file.name}`")
    
    with col2:
        st.markdown("<div class='section-title'>⚡ Traitement</div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <p><b>Cliquez sur le bouton ci-dessous</b> pour lancer l'extraction des données du document.</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Extraire les données", type="primary", use_container_width=True):
            with st.spinner("⏳ Analyse visuelle en cours... Veuillez patienter"):
                try:
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
                        st.error("❌ Erreur lors du parsing des données")
                        st.markdown("**Réponse brute du modèle:**")
                        st.code(raw_result, language="json")
                    else:
                        st.markdown("""
                            <div class="success-box">
                                <h4>✅ Extraction réussie!</h4>
                                <p>Les données ont été correctement extraites et structurées.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<div class='section-title'>📊 Données extraites</div>", unsafe_allow_html=True)
                        st.json(final_data)
                        
                        # Option de téléchargement
                        col_download, col_copy = st.columns(2)
                        with col_download:
                            st.download_button(
                                label="⬇️ Télécharger JSON",
                                data=json.dumps(final_data, ensure_ascii=False, indent=2),
                                file_name=f"extraction_{uploaded_file.name.split('.')[0]}.json",
                                mime="application/json"
                            )
                except Exception as e:
                    st.error(f"❌ Une erreur est survenue: {str(e)}")

elif uploaded_file and not target_schema:
    st.warning("⚠️ Veuillez sélectionner un schéma dans la barre latérale avant de continuer.")
