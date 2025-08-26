import os
from pathlib import Path
from typing import List

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# PDF processing
from docling.document_converter import DocumentConverter

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings

# Embeddings and LLM
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------- Simple Configuration ----------
class Config:
    PDF_DIR = Path("./Ansys_docs")
    CHROMA_DIR = Path("./chroma_ansys")
    EMBEDDING_MODEL = "intfloat/e5-base-v2"
    #LLM_MODEL = "llama3.1:8b"
    LLM_MODEL = "gpt-4o"  # "gpt-3.5-turbo" or "gpt-4o"
    #LLM_TYPE = "ollama"  # "ollama "or "openai"
    LLM_TYPE = "openai"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your API key as environment variable
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

# ---------- OpenAI LLM Wrapper ----------
class OpenAILLM:
    def __init__(self, model="gpt-3.5-turbo"):
        if OpenAI is None:
            raise ImportError("Install OpenAI: pip install openai")
        
        if not Config.OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000, # "max_tokens" or "max_completions"
            temperature=0.1
        )
        return response.choices[0].message.content

# ---------- Simple E5 Embeddings ----------
class SimpleE5Embeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, device='cpu')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {text}" for text in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}"
        return self.model.encode([prefixed], normalize_embeddings=True)[0].tolist()

# ---------- PDF Processing ----------
def process_pdfs() -> List[Document]:
    print(f"Processing PDFs from {Config.PDF_DIR}")
    
    if not Config.PDF_DIR.exists():
        print(f"Directory {Config.PDF_DIR} not found!")
        return []
    
    converter = DocumentConverter()
    all_chunks = []
    
    for pdf_path in Config.PDF_DIR.glob("*.pdf"):
        print(f"Converting: {pdf_path.name}")
        try:
            # Convert PDF to markdown
            result = converter.convert(str(pdf_path))
            markdown = result.document.export_to_markdown()
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            texts = splitter.split_text(markdown)
            
            # Create documents
            for text in texts:
                if text.strip():
                    all_chunks.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path.name}
                    ))
            
            print(f"✓ Processed {pdf_path.name}")
            
        except Exception as e:
            print(f"✗ Failed to process {pdf_path.name}: {e}")
    
    print(f"Created {len(all_chunks)} chunks from {len(list(Config.PDF_DIR.glob('*.pdf')))} PDFs")
    return all_chunks

# ---------- Vector Store ----------
def get_vector_store(documents: List[Document] = None) -> Chroma:
    embeddings = SimpleE5Embeddings()
    
    # Load existing store if available
    if Config.CHROMA_DIR.exists() and any(Config.CHROMA_DIR.iterdir()):
        print("Loading existing vector store...")
        return Chroma(persist_directory=str(Config.CHROMA_DIR), embedding_function=embeddings)
    
    # Create new store
    if not documents:
        raise ValueError("No documents provided and no existing store found")
    
    print("Creating new vector store...")
    Config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=str(Config.CHROMA_DIR)
    )

# ---------- Question Answering ----------
def ask_question(question: str, vector_store: Chroma, top_k: int = 10) -> str:
    print(f"\nQuery: {question}\n",)
    
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    
    
    if not docs:
        return "No relevant information found."
    
    # Show retrieved sources
    print(f"Found {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  {i}. {source}: {preview}")
    
    # Create context
    context = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    
    # Generate answer
    if Config.LLM_TYPE == "openai":
        llm = OpenAILLM(model=Config.LLM_MODEL)
    else:
        llm = OllamaLLM(model=Config.LLM_MODEL)
        
    prompt = f"""You are an ANSYS technical expert. Use only the provided context to answer the user’s question. 

Instructions:
- Provide a clear, step-by-step, and technically accurate answer. 
- When possible, cite relevant sections or terminology from the provided context. 
- If multiple solutions exist, list them and explain trade-offs. 
- If the context does not contain the answer, say: "The provided documentation does not cover this." Do not guess.

Question: {question}

Context:
{context}

Answer:"""
    
    return llm.invoke(prompt)

# ---------- Main Functions ----------
def setup():
    print("Setting up ANSYS RAG system...")
    
    # Process PDFs
    documents = process_pdfs()
    if not documents:
        print("No documents created. Check your PDF directory.")
        return None
    
    # Create vector store
    vector_store = get_vector_store(documents)
    print("✓ Setup complete!")
    return vector_store

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ANSYS RAG System")
    parser.add_argument("--setup", action="store_true", help="Process PDFs and create vector store")
    parser.add_argument("--ask", type=str, help="Ask a question")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    args = parser.parse_args()
    
    # Setup mode
    if args.setup:
        setup()
        return
    
    # Load existing vector store
    try:
        vector_store = get_vector_store()
    except ValueError as e:
        print(f"Error: {e}")
        print("Run with --setup first to process your PDFs.")
        return
    
    # Single question mode
    if args.ask:
        answer = ask_question(args.ask, vector_store)
        print(f"\nANSWER:\n{answer}")
        return
    
    # Interactive mode
    if args.interactive:
        print("\nInteractive ANSYS Q&A (type 'quit' to exit)")
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                answer = ask_question(question, vector_store)
                print(f"\nANSWER:\n\n{answer}")
        return
    
    # No arguments - show help
    parser.print_help()

if __name__ == "__main__":
    main()
