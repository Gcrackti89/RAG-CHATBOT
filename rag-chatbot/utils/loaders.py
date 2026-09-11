import os
import glob
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

def load_documents(docs_dir: str):
    """
    Scans the given directory and loads supported documents.
    Supported formats: .pdf, .docx, .txt, .md
    """
    documents = []
    
    # Supported file extensions and their corresponding loaders
    loaders_map = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    # Iterate through all files in the directory
    for root, _, files in os.walk(docs_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in loaders_map:
                file_path = os.path.join(root, file)
                print(f"Loading {file_path}...")
                try:
                    loader = loaders_map[ext](file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
    return documents
