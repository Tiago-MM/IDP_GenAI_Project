import base64
import streamlit as st
import os
import json
import time
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

# Import local
from src.llm_engine import analyse_image
from src.utils import clean_json_output
from src.ocr_engine import process_with_easyocr, parse_ocr_with_llm

# --- INITIALISATION ET CONFIGURATION GLOBALE ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check de sécurité
if not GROQ_API_KEY:
    st.error("⚠️ Clé GROQ_API_KEY manquante dans le fichier .env")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    layout="wide",
    page_title="IDP GenAI Project", 
    initial_sidebar_state="expanded",
    menu_items={"About": "Extraction de documents avancée par IA"}
)

# --- INITIALISATION DU SESSION STATE ---
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "active_view" not in st.session_state: 
    st.session_state.active_view = 'JSON'

# --- VARIABLES ET CSS CUSTOM ---
PRIMARY_COLOR = "#333333"
SECONDARY_COLOR = "#999999"
SUCCESS_COLOR = "#609966"

st.markdown(f"""
    <style>
        h1, h2 {{ margin-top: 5px; padding-top: 0; margin-bottom: 5px; }}
        .logo-container {{ display: flex; align-items: center; gap: 10px; padding-bottom: 5px; }}
        .section-title {{
            color: {PRIMARY_COLOR}; font-size: 18px; font-weight: bold;
            margin-top: 15px; margin-bottom: 5px; border-left: 4px solid {SECONDARY_COLOR}; padding-left: 10px;
        }}
        .info-box {{ background-color: #F8F8F8; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .success-box {{ background-color: #e6f7e6; padding: 15px; border-radius: 8px; border-left: 4px solid {SUCCESS_COLOR}; }}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="logo-container">
        <h2>📄 IDP Project : Intelligent Document Processing</h2>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR (Configuration) ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_choice = st.selectbox(
        "🤖 Modèle IA",
        ["meta-llama/llama-4-scout-17b-16e-instruct", "easyocr"],
        help="Sélectionnez le modèle à utiliser pour l'analyse"
    )
    st.info("ℹ️ Mode Batch activé : Traitement de plusieurs fichiers.")
    
    st.divider()
    st.markdown("### 🖼️ Contrôles")
    zoom_level = st.slider("🔍 Zoom (%)", 50, 200, 100, 10, key="sidebar_zoom")

# --- UPLOADER (BATCH) ---
st.markdown("<div class='section-title'>📤 Télécharger vos documents</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Sélectionnez une ou plusieurs images",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,  # <--- IMPORTANT POUR LE BATCH
    help="Formats acceptés: PNG, JPG, JPEG"
)

# --- LOGIQUE D'EXTRACTION ---
if uploaded_files:
    # Bouton d'action
    if st.button(f"🚀 Lancer l'extraction ({len(uploaded_files)} fichiers)", type="primary", use_container_width=True):
        
        st.session_state.batch_results = [] # Reset des résultats
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- BOUCLE SUR CHAQUE FICHIER ---
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Traitement de {filename} ({i+1}/{len(uploaded_files)})...")
            
            try:
                # Lecture du fichier
                uploaded_file.seek(0)
                img_bytes = uploaded_file.read()
                
                final_data = None
                
                # --- CAS 1 : EASYOCR + LLM ---
                if model_choice == "easyocr":
                    # 1. OCR Brut
                    raw_text, _ = process_with_easyocr(img_bytes)
                    # 2. Structuration via LLM
                    json_str = parse_ocr_with_llm(raw_text, client_groq=client)
                    # 3. Nettoyage
                    final_data = json.loads(clean_json_output(json_str))
                    
                # --- CAS 2 : VLM (LLAMA VISION / GROQ) ---
                else:
                    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                    raw_result_generator = analyse_image(
                        image=encoded_image,
                        model=model_choice,
                        GROQ_API_KEY=GROQ_API_KEY
                    )
                    # Reconstitution du stream
                    full_raw_result = "".join([chunk.choices[0].delta.content or "" for chunk in raw_result_generator])
                    
                    # Parsing
                    if isinstance(full_raw_result, str):
                        cleaned_str = clean_json_output(full_raw_result)
                        try:
                            if isinstance(cleaned_str, str):
                                final_data = json.loads(cleaned_str)
                            else:
                                final_data = cleaned_str
                        except json.JSONDecodeError:
                            final_data = {"error": "JSON invalide", "raw": full_raw_result}
                    else:
                        final_data = full_raw_result

                # --- AGREGATION DES RÉSULTATS ---
                # On ajoute le nom du fichier source dans le JSON pour s'y retrouver
                if isinstance(final_data, dict):
                    final_data["_Source_File"] = filename
                    st.session_state.batch_results.append(final_data)
                elif isinstance(final_data, list):
                     for item in final_data:
                         if isinstance(item, dict):
                             item["_Source_File"] = filename
                     st.session_state.batch_results.extend(final_data)
                
            except Exception as e:
                st.error(f"❌ Erreur sur {filename}: {str(e)}")
            
            # Mise à jour barre de progression
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.success("✅ Traitement terminé !")
        time.sleep(1) # Petit temps pour voir le message
        st.rerun() # Rafraîchir pour afficher les résultats

# --- AFFICHAGE DES RÉSULTATS ---

if uploaded_files:
    
    # S'il y a des résultats en mémoire
    if st.session_state.batch_results:
        
        st.divider()
        
        # --- MODE BATCH (JSON GLOBAL) ---
        if len(uploaded_files) > 1:
            st.markdown("### 📦 Résultat Global (Tous les fichiers)")
            
            # Affichage JSON interactif
            st.json(st.session_state.batch_results, expanded=False)
            
            # Bouton de téléchargement GLOBAL
            json_str_all = json.dumps(st.session_state.batch_results, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Télécharger le JSON Global",
                data=json_str_all,
                file_name="batch_extraction_results.json",
                mime="application/json",
                type="primary",
                use_container_width=True
            )

        # --- MODE SINGLE (VUE DÉTAILLÉE) ---
        # Si on n'a qu'un seul fichier, on garde votre belle interface Split View
        elif len(uploaded_files) == 1:
            st.markdown(f"### 🔎 Vue Détaillée : `{uploaded_files[0].name}`")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Document Source")
                uploaded_files[0].seek(0)
                image_pil = Image.open(uploaded_files[0])
                st.image(image_pil, use_container_width=True)
            
            with col2:
                st.info("Données Extraites")
                # On prend le premier élément de la liste
                result_unit = st.session_state.batch_results[0]
                st.json(result_unit, expanded=True)
                
                # Téléchargement unitaire
                json_str_unit = json.dumps(result_unit, indent=2, ensure_ascii=False)
                st.download_button(
                    label="⬇️ Télécharger JSON",
                    data=json_str_unit,
                    file_name=f"extract_{uploaded_files[0].name}.json",
                    mime="application/json",
                    use_container_width=True
                )

    else:
        # Message d'attente si pas encore traité
        if len(uploaded_files) == 1:
            st.info("👆 Cliquez sur le bouton 'Extraire' pour analyser ce document.")
        else:
            st.info(f"👆 Cliquez sur 'Lancer l'extraction' pour traiter vos {len(uploaded_files)} documents.")