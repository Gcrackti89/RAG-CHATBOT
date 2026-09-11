from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits a list of documents into smaller chunks for embedding.
    """
    if not documents:
        return []
        
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks
