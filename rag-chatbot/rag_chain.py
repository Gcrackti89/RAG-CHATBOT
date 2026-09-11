from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL_NAME, TOP_K, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME
from utils.vectorstore import load_vector_store

def get_llm():
    """
    Initializes the ChatOpenAI model pointing to OpenRouter API.
    """
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise ValueError("OPENROUTER_API_KEY is not set. Please set it in your .env file.")
        
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        streaming=True
    )

def format_docs(docs):
    """
    Formats the retrieved documents into a single string for context,
    along with their source filenames and page numbers (if available).
    """
    formatted_docs = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "")
        page_info = f" (Page {page})" if page else ""
        content = d.page_content
        formatted_docs.append(f"[Source: {source}{page_info}]\n{content}\n")
    return "\n".join(formatted_docs)

def get_rag_chain():
    """
    Builds and returns the complete RAG chain.
    """
    # Load vector store and set up retriever
    vectorstore = load_vector_store(CHROMA_DB_DIR, EMBEDDING_MODEL_NAME)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Setup LLM
    llm = get_llm()
    
    # Define Prompt Template
    template = """You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Always cite the source if possible based on the context provided.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain
    # The chain first retrieves documents, then formats them, passes them to the prompt, and finally to the LLM.
    def build_chain():
        # Using a custom wrapper to return both answer and source docs
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
    return build_chain(), retriever
