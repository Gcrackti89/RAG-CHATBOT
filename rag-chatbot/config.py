import os
from dotenv import load_dotenv

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)
# API Keys
OPENROUTER_API_KEY = os.getenv("UnoRouter") or os.getenv("UnoRouter1") or os.getenv("OPENROUTER_API_KEY")

# OpenRouter Settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct:free"

# Embedding Settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Text Splitting Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
TOP_K = 4

# Paths
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure documents directory exists
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
