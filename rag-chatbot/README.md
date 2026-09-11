# RAG Chatbot over Custom Documents

A complete Retrieval-Augmented Generation (RAG) chatbot using LangChain, Streamlit, OpenRouter, and ChromaDB.

## Features
- Ingests PDF, DOCX, TXT, and MD files.
- Uses free, local HuggingFace embeddings (`all-MiniLM-L6-v2`).
- Stores documents persistently in a local ChromaDB instance.
- Chat interface via Streamlit.
- Uses OpenRouter API to access free LLMs like `meta-llama/llama-3.1-8b-instruct:free`.

## Setup Instructions

1. **Install Requirements**
   Make sure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   - Copy `.env.example` to `.env`.
   - Obtain an API key from [OpenRouter](https://openrouter.ai/).
   - Add your API key to the `.env` file:
     ```env
     OPENROUTER_API_KEY=your_actual_api_key
     ```

3. **Add Documents**
   Place your PDF, DOCX, TXT, or MD files inside the `data/documents/` folder.

4. **Ingest Documents**
   Run the ingestion script to process your documents and create the local vector database.
   ```bash
   python ingest.py
   ```

5. **Run the Streamlit App**
   Start the web interface.
   ```bash
   streamlit run app.py
   ```
