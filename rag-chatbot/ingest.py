import os
import shutil
from config import DOCUMENTS_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from utils.loaders import load_documents
from utils.splitter import split_documents
from utils.vectorstore import create_vector_store

def run_ingestion():
    """
    Runs the full document ingestion pipeline:
    1. Loads documents from the data/documents/ folder
    2. Splits them into chunks
    3. Embeds and stores them in ChromaDB
    """
    print("Starting ingestion pipeline...")
    
    # 1. Load documents
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print("No documents found to ingest. Please add files to data/documents/")
        return False
        
    print(f"Loaded {len(documents)} documents.")
    
    # 2. Split documents
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print("Failed to split documents into chunks.")
        return False
        
    # 3. Create Vector Store (clear existing DB if needed)
    if os.path.exists(CHROMA_DB_DIR):
        print("Clearing existing vector store...")
        shutil.rmtree(CHROMA_DB_DIR)
        
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    # Store in ChromaDB
    create_vector_store(chunks, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME)
    
    print("Ingestion complete!")
    return True

if __name__ == "__main__":
    run_ingestion()
