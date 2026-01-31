import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings for the ANSYS RAG system"""
    
    # Directory paths
    PDF_DIR = Path("./Ansys_docs")
    CHROMA_DIR = Path("./chroma_ansys")
    
    # Embedding settings
    EMBEDDING_MODEL = "intfloat/e5-base-v2"
    DEVICE = "cpu"  # or "cuda" for GPU
    
    # LLM settings
    LLM_MODEL = "gpt-4o"  # llama3.1:8b or "gpt-3.5-turbo" and "gpt-4o"
    LLM_TYPE = "openai" # "ollama" or "openai"
    
    # API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your API key as environment variable
    
    # Text processing settings
    CHUNK_SIZE = 600  # tokens per chunk
    CHUNK_OVERLAP = 120  # 20% overlap
    
    # LLM generation settings
    TEMPERATURE = 0.1
    MAX_TOKENS = 1000
