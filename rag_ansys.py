from typing import List

# Configuration
from config import Config

# PDF processing
from docling.document_converter import DocumentConverter

# BM25 for lexical search
from rank_bm25 import BM25Okapi

# LangChain components
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain_ollama import OllamaLLM

# Embeddings
from sentence_transformers import SentenceTransformer

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------- OpenAI LLM Wrapper ----------
class OpenAILLM:
    """Wrapper for OpenAI GPT models."""
    def __init__(self, model: str):
        if OpenAI is None:
            raise ImportError("Install OpenAI: pip install openai")
        
        if not Config.OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model

    def invoke(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE,
        )
        return response.choices[0].message.content


# ---------- E5 Embeddings ----------
class SimpleE5Embeddings(Embeddings):
    """E5 embeddings for semantic text representation."""
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks with 'passage:' prefix."""
        prefixed = [f"passage: {text}" for text in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed user query with 'query:' prefix."""
        prefixed = f"query: {text}"
        return self.model.encode([prefixed], normalize_embeddings=True)[0].tolist()


# ---------- PDF Processing ----------
def process_pdfs() -> List[Document]:
    """Process all PDFs in the configured directory and return document chunks."""
    
    print(f"Processing PDFs from {Config.PDF_DIR}")
    
    if not Config.PDF_DIR.exists():
        print(f"Directory {Config.PDF_DIR} not found!")
        return []
    
    converter = DocumentConverter()
    all_chunks = []
    pdf_files = list(Config.PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in directory.")
        return []
    
    for pdf_path in pdf_files:
        print(f"Converting: {pdf_path.name}")
        try:
            # Convert PDF to structured text
            result = converter.convert(str(pdf_path))
            # Structured text to markdown format
            markdown = result.document.export_to_markdown()
            
            # Split by headers to preserve document structure
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            header_splits = markdown_splitter.split_text(markdown)
            
            # Check if document has headers
            if len(header_splits) == 1 and not header_splits[0].metadata:
                print(f"  ⚠ No headers found in {pdf_path.name}, using paragraph-based splitting")
            else:
                print(f"  ✓ Found {len(header_splits)} sections in {pdf_path.name}")
            
            # Split by tokens, respecting section boundaries (1 token = ~4 characters)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE * 4,  # Convert tokens to approximate characters
                chunk_overlap=Config.CHUNK_OVERLAP * 4,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            texts = text_splitter.split_documents(header_splits)
            
            # Create documents with enriched metadata
            for doc in texts:
                if doc.page_content.strip():
                    # Merge filename metadata with headers metadata
                    metadata = {"source": pdf_path.name}
                    metadata.update(doc.metadata)
                    all_chunks.append(Document(
                        page_content=doc.page_content,
                        metadata=metadata
                    ))
            
            print(f"✓ Processed {pdf_path.name}")
            
        except Exception as e:
            print(f"✗ Failed to process {pdf_path.name}: {e}")
    
    print(f"\nCreated {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    return all_chunks


# ---------- Vector Store ----------
def get_vector_store(documents: List[Document] = None) -> Chroma:
    """Load existing vector store or create a new one from documents."""
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

# ---------- Hybrid Search ----------
def hybrid_search(question: str, vector_store: Chroma, top_k: int = 10) -> List[Document]:
    """Combine semantic (E5) and lexical (BM25) search: top 10 semantic + top 3 BM25."""
    # Get all documents
    store_data = vector_store.get()
    all_docs = store_data['documents']
    all_metadata = store_data['metadatas']
    all_ids = store_data['ids']
    
    # Semantic search - get top 10
    semantic_results = vector_store.similarity_search(question, k=10)
    
    # BM25 lexical search - get top 3
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
    
    # Combine results: semantic first, then BM25 (avoiding duplicates)
    seen_ids = set()
    combined_docs = []
    
    # Add semantic results
    for doc in semantic_results:
        doc_id = all_ids[all_docs.index(doc.page_content)]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined_docs.append(doc)
    
    # Add BM25 results
    for idx in bm25_top_indices:
        doc_id = all_ids[idx]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined_docs.append(Document(
                page_content=all_docs[idx],
                metadata=all_metadata[idx]
            ))
    
    return combined_docs

# ---------- Question Answering. Core RAG function that ties everything together ----------
def ask_question(question: str, vector_store: Chroma, top_k: int = 10) -> str:
    """Retrieve relevant documents and generate an answer using LLM."""
    
    print(f"\nQuery: {question}\n")
    
    # Hybrid retrieval: semantic + BM25
    docs = hybrid_search(question, vector_store, top_k)
    
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
    
    # Debug: show full context being sent to LLM
    print(f"\n{'='*80}")
    print("FULL CONTEXT SENT TO LLM:")
    print(f"{'='*80}")
    print(context)
    print(f"{'='*80}\n")
    
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
    """Process PDFs and create vector store for the first time."""
    
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
    """Main entry point for the ANSYS RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ANSYS RAG System - Query ANSYS documentation using AI"
    )
    parser.add_argument("--setup", action="store_true", help="Process PDFs and create vector store")
    parser.add_argument("--ask", type=str, help="Ask a question")
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
    
    # No arguments - show help
    parser.print_help()

if __name__ == "__main__":
    main()
