import os
import streamlit as st
from ingest import run_ingestion
from rag_chain import get_rag_chain
from config import DOCUMENTS_DIR

# Set Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("📚 RAG Chatbot over Custom Documents")

# Sidebar for file upload and ingestion
with st.sidebar:
    st.header("Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, MD)", 
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md"]
    )
    
    if uploaded_files:
        if st.button("Save Uploaded Files"):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s).")
            
    # Ingestion button
    if st.button("Run Document Ingestion", type="primary"):
        with st.spinner("Processing documents and creating vector store..."):
            success = run_ingestion()
            if success:
                st.success("Ingestion complete! You can now chat with your documents.")
            else:
                st.error("Ingestion failed or no documents found.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source_docs" in message and message["source_docs"]:
            with st.expander("View Source Documents"):
                for idx, doc in enumerate(message["source_docs"]):
                    st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.text(doc.page_content)

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Initialize chain and retriever
            chain, retriever = get_rag_chain()
            
            # Fetch source docs for display
            source_docs = retriever.invoke(prompt)
            
            # Stream the response
            full_response = ""
            for chunk in chain.stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                
            # Finalize response
            message_placeholder.markdown(full_response)
            
            # Display source documents in expander
            if source_docs:
                with st.expander("View Source Documents"):
                    for idx, doc in enumerate(source_docs):
                        st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.text(doc.page_content)
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "source_docs": source_docs
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Have you set your OPENROUTER_API_KEY in the .env file?")
