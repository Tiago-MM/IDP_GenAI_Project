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

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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

# initialize session state variables
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "active_view" not in st.session_state: 
    st.session_state.active_view = 'JSON'

# css styles
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

# header
st.markdown("""
    <div class="logo-container">
        <h2>📄 IDP Project : Intelligent Document Processing</h2>
    </div>
""", unsafe_allow_html=True)

# sidebar for configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_choice = st.selectbox(
        "🤖 Modèle IA",
        ["meta-llama/llama-4-scout-17b-16e-instruct", "easyocr"],
        help="Sélectionnez le modèle à utiliser pour l'analyse"
    )
    st.markdown("### 📋 Schéma de Données")
    schema_dir = "schemas"
    available_schemas = [f for f in os.listdir(schema_dir) if f.endswith('.json')]
    # auto detection option
    selected_schema_name = st.selectbox("Format de sortie", ["Auto-détection"] + available_schemas)

    target_schema = None
    if selected_schema_name != "Auto-détection":
        with open(os.path.join(schema_dir, selected_schema_name), "r") as f:
            target_schema = f.read()
        with st.expander("Voir le schéma cible"):
            st.code(target_schema, language="json")


    st.info("ℹ️ Mode Batch activé : Traitement de plusieurs fichiers.")
    
    st.divider()
    st.markdown("### 🖼️ Contrôles")
    zoom_level = st.slider("🔍 Zoom (%)", 50, 200, 100, 10, key="sidebar_zoom")

# batch upload
st.markdown("<div class='section-title'>📤 Télécharger vos documents</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Sélectionnez une ou plusieurs images",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,  # true for batch upload
    help="Formats acceptés: PNG, JPG, JPEG"
)


if uploaded_files:
    st.markdown("<div class='section-title'>2. Aperçu des documents</div>", unsafe_allow_html=True)
    
    preview_map = {f.name: f for f in uploaded_files}
    
    col_sel, col_view = st.columns([1, 2])
    
    with col_sel:
        selected_preview_name = st.selectbox(
            "Choisir une image à vérifier :",
            list(preview_map.keys()),
            key="preview_selector"
        )
        if selected_preview_name:
            file_info = preview_map[selected_preview_name]
            st.info(f"📄 **Fichier :** {file_info.name}\n\n💾 **Taille :** {file_info.size / 1024:.1f} KB")

    with col_view:
        if selected_preview_name:
            file_to_show = preview_map[selected_preview_name]
            file_to_show.seek(0)
            image_preview = Image.open(file_to_show)
            st.image(image_preview, caption=f"Aperçu : {selected_preview_name}", use_container_width=True)
            file_to_show.seek(0)

# process button
if uploaded_files:
    if st.button(f"🚀 Lancer l'extraction ({len(uploaded_files)} fichiers)", type="primary", use_container_width=True):
        
        st.session_state.batch_results = [] # Reset des résultats
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # processing each file
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Traitement de {filename} ({i+1}/{len(uploaded_files)})...")
            
            try:
                # read file bytes
                uploaded_file.seek(0)
                img_bytes = uploaded_file.read()
                
                final_data = None
                
                # case easyocr
                if model_choice == "easyocr":
                    # brut ocr
                    raw_text, _ = process_with_easyocr(img_bytes)
                    # transform with llm
                    json_str = parse_ocr_with_llm(raw_text, client_groq=client,schema_json=target_schema)
                    # clean and parse json
                    final_data = json.loads(clean_json_output(json_str))
                    
                # case llm vision
                else:
                    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                    raw_result_generator = analyse_image(
                        image=encoded_image,
                        model=model_choice,
                        GROQ_API_KEY=GROQ_API_KEY,
                        schema_json=target_schema
                    )
                    # streaming aggregation
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

                # agregate results
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
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.success("✅ Traitement terminé !")
        time.sleep(1) 
        st.rerun() 

# print results if any
if uploaded_files:
    
    if st.session_state.batch_results:
        
        st.divider()
        
        tab_global, tab_detail = st.tabs(["📦 Vue Globale (JSON)", "🔍 Explorateur par Image"])
        
        # gloval view
        with tab_global:
            st.markdown(f"### Résultat consolidé ({len(st.session_state.batch_results)} documents)")
            
            # print json
            st.json(st.session_state.batch_results, expanded=False)
            
            # download button
            json_str_all = json.dumps(st.session_state.batch_results, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Télécharger le JSON Global",
                data=json_str_all,
                file_name="batch_extraction_results.json",
                mime="application/json",
                type="primary",
                use_container_width=True
            )

        # detailed view per image
        with tab_detail:
            #dict creation for easy access
            file_map = {f.name: f for f in uploaded_files}
            
            # files list 
            processed_files = [res.get("_Source_File", "Inconnu") for res in st.session_state.batch_results]
            
            if processed_files:
                selected_filename = st.selectbox("Choisir un document à inspecter :", processed_files)
                
                
                selected_result = next((item for item in st.session_state.batch_results if item["_Source_File"] == selected_filename), None)
                selected_image_file = file_map.get(selected_filename)

                # Affichage Split View
                if selected_image_file and selected_result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"🖼️ Image : {selected_filename}")
                        selected_image_file.seek(0)
                        img = Image.open(selected_image_file)
                        st.image(img, use_container_width=True)
                    
                    with col2:
                        st.info("🧠 Données Extraites")
                        st.json(selected_result, expanded=True)
                else:
                    st.warning("Impossible d'associer l'image au résultat.")
            else:
                st.info("Aucun résultat à afficher pour le moment.")

    else:
        if len(uploaded_files) > 0:
            st.info(f"👆 Cliquez sur 'Lancer l'extraction' pour traiter vos {len(uploaded_files)} documents.")