# ANSYS RAG System

A Retrieval-Augmented Generation (RAG) system for querying ANSYS documentation using OpenAI GPT models or local LLMs via Ollama.

## Overview

This project implements a sophisticated question-answering system that processes ANSYS PDF documentation and enables natural language queries using:

- **Docling** for advanced PDF processing and conversion to markdown
- **E5 embeddings** for semantic text representation
- **ChromaDB** for vector storage and similarity search
- **OpenAI GPT models** (GPT-3.5, GPT-4, GPT-4o) or **Ollama** with local models
- **LangChain** for RAG pipeline orchestration

## Features

- 🔄 **PDF Processing**: Converts ANSYS PDFs to structured markdown with table and layout recognition
- 🧠 **Semantic Search**: Uses E5 embeddings for accurate document retrieval
- 💾 **Persistent Storage**: ChromaDB vector store for fast query responses
- 🤖 **Flexible LLM Support**: OpenAI GPT models (recommended) or local Ollama models
- 💬 **Multiple Interfaces**: Command-line, single query, and interactive modes
- 🔍 **Debug Information**: Detailed logging and retrieval diagnostics
- 🔐 **Environment Variables**: Secure API key management with .env files

## Prerequisites

### System Requirements
- Python 3.11+
- Windows/macOS/Linux
- 8GB+ RAM (recommended for Llama 3.1 8B)
- 10GB+ free disk space

### Required Software
1. **OpenAI API Key** (recommended): Get from [platform.openai.com](https://platform.openai.com)
2. **OR Ollama** (for local models): Download from [ollama.ai](https://ollama.ai)
3. **Llama 3.1 8B model** (if using Ollama): Install via `ollama pull llama3.1:8b`

## Installation

### 1. Clone and Setup Environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv/bin/activate     # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages** (if no requirements.txt):
```bash
pip install docling langchain langchain-chroma langchain-ollama langchain-text-splitters sentence-transformers chromadb openai python-dotenv
```

### 3. Setup API Key (for OpenAI)
Create a `.env` file in the project directory:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important**: Add `.env` to your `.gitignore` to keep your API key secure!

### 4. Setup Ollama (Optional - for local models)
```bash
# Install Ollama from https://ollama.ai
ollama serve  # Start Ollama service
ollama pull llama3.1:8b  # Download the model
```

## Project Structure

```
AI_LLM_RAG/
├── rag_ansys.py              # Main RAG system
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── .gitignore               # Git ignore file (includes .env)
├── Ansys_docs/              # PDF documents directory
│   └── *.pdf                # ANSYS documentation files
├── chroma_ansys/            # Vector database
    └── chroma.sqlite3       # Persistent vector storage      
```

## Usage

### 1. Initial Setup (Required)
Process PDF documents and create vector store:
```bash
python rag_ansys.py --setup
```

### 2. Ask Single Questions
```bash
python rag_ansys.py --ask "How to set boundary conditions?"
```

### 3. Interactive Mode
```bash
python rag_ansys.py --interactive
```

### 4. Switch Between LLM Types
```bash
# Use OpenAI (default)
python rag_ansys.py --llm openai --model gpt-4o --ask "your question"

# Use local Ollama
python rag_ansys.py --llm ollama --model llama3.1:8b --ask "your question"
```

### 5. Get Help
```bash
python rag_ansys.py --help
```

## Configuration

Edit the `Config` class in `rag_ansys.py` to customize:

```python
class Config:
    PDF_DIR = Path("./Ansys_docs")           # PDF source directory
    CHROMA_DIR = Path("./chroma_ansys")      # Vector store location
    EMBEDDING_MODEL = "intfloat/e5-base-v2"  # Embedding model
    LLM_MODEL = "gpt-4o"                    # OpenAI model or Ollama model
    LLM_TYPE = "openai"                     # "openai" or "ollama"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Loaded from .env
    CHUNK_SIZE = 1000                       # Text chunk size
    CHUNK_OVERLAP = 100                     # Overlap between chunks
```

### Available Models

**OpenAI Models** (requires API key):
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4o` - Latest optimized model

**Ollama Models** (local, free):
- `llama3.1:8b` - Good general performance

## Technical Details

- **Embeddings**: E5-base-v2 model for semantic text representation with asymmetric query/passage prefixes
- **Chunking**: 600 tokens (≈2400 characters) with 120 token overlap, header-aware splitting (H1-H3)
- **Vector Store**: ChromaDB with persistent storage
- **Hybrid Retrieval**: 
  - Top 10 chunks from semantic (E5) search
  - Top 3 chunks from BM25 lexical search
  - Combined without duplicates (semantic prioritized)
- **LLM Integration**: OpenAI GPT-4o (temperature=0.1, max_tokens=1000) with strict grounding prompt
- **PDF Processing**: Docling for PDF→Markdown conversion with table recognition


