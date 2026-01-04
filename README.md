# IDP GenAI Project: Intelligent Document Processing

This project is an **Intelligent Document Processing (IDP)** solution designed to extract structured data from various documents (such as ID cards and invoices) using Multimodal Large Language Models (LLMs) and OCR technology.

## 🚀 Key Features

* **Multimodal Extraction**: Uses advanced vision models like `llama-4-scout` via the Groq API to analyze images directly.
* **Hybrid OCR + LLM Approach**: Supports extraction through **EasyOCR**, combined with an LLM to structure and correct raw text.
* **Schema-Based Validation**: Ensures data extraction follows predefined JSON format (prefefined json format is not mandatory), such as `id_card_schema.json` or `invoice_schema.json`.
* **Streamlit Interface**: Features a user-friendly web interface for batch uploading, document previewing, and JSON results export.
* **Batch Processing**: Capable of processing multiple files simultaneously and consolidating the results into a single output.

## 🛠️ Project Architecture

```text
├── app.py                # Main Streamlit application and UI logic
├── src/
│   ├── llm_engine.py     # Logic for interacting with Groq Vision LLMs
│   ├── ocr_engine.py     # EasyOCR integration and LLM parsing logic
│   └── utils.py          # Helper functions for cleaning JSON outputs
├── schemas/              # JSON schemas defining the target data structure
├── notebooks/            # Jupyter notebooks for testing OCR and Llama models
└── requirements.txt      # List of Python dependencies

```

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd IDP_GenAI_Project

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Set up environment variables**:
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here

```


## 🖥️ Usage

Start the Streamlit application with the following command:

```bash
streamlit run app.py

```

1. **Configuration**: Select your preferred AI model (Llama 4 Vision or EasyOCR) in the sidebar.
2. **Schema**: Choose the appropriate output format for your document type.
3. **Upload**: Drag and drop your images (JPG, PNG) into the uploader.
4. **Extract**: Click "Run Extraction" to process the documents and view the structured JSON data.

## 📊 Example Output

The system can transform an image of an ID card into a structured JSON object:

```json
{
  "type": "id_card",
  "first_name": "Audrey",
  "last_name": "Chevallier",
  "id_number": "T7X62TZ79",
  "birth_date": "1995-04-01",
  "expiry_date": "2031-01-27"
}

```

*(Result based on Llama 4 Vision inference)*