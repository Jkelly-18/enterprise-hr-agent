"""
# ingest_docs.py

## What this script does:
This script processes all PDF documents in the company_docs/ folder and
loads them into ChromaDB as vector embeddings for use by the RAG pipeline.
It does the following in order:

1. Scans the company_docs/ folder for all PDF files
2. Extracts raw text from each PDF using pypdf
3. Splits each document into overlapping chunks using LangChain's text splitter
   so that context is preserved at chunk boundaries
4. Sends each chunk to OpenAI's embedding model to generate a vector
   representation of its meaning
5. Stores all chunks + their vectors + metadata in ChromaDB

Run this script once after seed_database.py and before starting the backend:
    python scripts/ingest_docs.py

The ChromaDB vector store will be saved to: internal_data/chroma/
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration

DOCS_DIR        = Path("company_docs")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./internal_data/chroma")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "velo_knowledge_base"

# Chunking settings
CHUNK_SIZE      = 800   # characters per chunk
CHUNK_OVERLAP   = 150   # overlap between chunks to preserve context


# Imports 
if not OPENAI_API_KEY:
    print(" OPENAI_API_KEY not found in .env file. Exiting.")
    sys.exit(1)

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Setup
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs("internal_data", exist_ok=True)

embedding_fn = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Helper Functions
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def get_doc_category(filename: str) -> str:
    """Infer document category from filename prefix."""
    categories = {
        "01": "general_onboarding",
        "02": "role_onboarding_engineering",
        "03": "role_onboarding_sales",
        "04": "role_onboarding_cs",
        "05": "hr_policy",
        "06": "hr_policy",
        "07": "hr_policy",
        "08": "it_security",
        "09": "it_security",
        "10": "internal_process_engineering",
        "11": "internal_process_sales",
        "12": "internal_process_cs",
        "13": "company",
    }
    prefix = filename[:2]
    return categories.get(prefix, "general")


def get_relevant_roles(filename: str) -> str:
    """Tag which roles this document is most relevant to."""
    role_map = {
        "01": "all",
        "02": "engineering",
        "03": "sales",
        "04": "customer_success",
        "05": "all",
        "06": "all",
        "07": "all",
        "08": "all",
        "09": "all",
        "10": "engineering",
        "11": "sales",
        "12": "customer_success",
        "13": "all",
    }
    prefix = filename[:2]
    return role_map.get(prefix, "all")


# Ingestion 
def ingest_documents():
    """Main ingestion pipeline — reads PDFs, chunks, embeds, stores."""

    # Check company_docs folder exists and has PDFs
    if not DOCS_DIR.exists():
        print(" company_docs/ folder not found")
        sys.exit(1)

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {DOCS_DIR}/")
        sys.exit(1)

    print(f"  📄 Found {len(pdf_files)} PDF files in {DOCS_DIR}/")

    # Delete and recreate collection for clean ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    total_chunks  = 0
    doc_summaries = []

    for pdf_path in pdf_files:
        filename = pdf_path.name
        doc_name = filename.replace(".pdf", "").replace("_", " ").title()

        print(f"\n  📃 Processing: {filename}")

        # Extract text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(" No text extracted — skipping")
            continue

        # Chunk the text
        chunks = text_splitter.split_text(raw_text)

        # Build metadata for each chunk
        category     = get_doc_category(filename)
        relevant_to  = get_relevant_roles(filename)

        ids       = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}__chunk_{i:04d}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source":       filename,
                "doc_name":     doc_name,
                "category":     category,
                "relevant_to":  relevant_to,
                "chunk_index":  i,
                "total_chunks": len(chunks),
            })

        # Add to ChromaDB in batches of 50 to avoid API rate limits
        batch_size = 50
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        total_chunks += len(chunks)
        doc_summaries.append({
            "filename":     filename,
            "chunks":       len(chunks),
            "category":     category,
            "relevant_to":  relevant_to,
            "char_count":   len(raw_text),
        })

    return collection, total_chunks, doc_summaries


##  Summary Report
def print_summary(total_chunks, doc_summaries):
    print("\nINGESTION SUMMARY")
    print("="*40)
    for doc in doc_summaries:
        print(f"  {doc['filename']:<45} {doc['chunks']} chunks")
    print(f"  {'TOTAL':<45} {total_chunks} chunks")
    print("="*40)


# Main 
def main():
    print("\nVELO — Document Ingestion Script")
    print(f"Scanning {DOCS_DIR}/ for PDF documents...")

    collection, total_chunks, doc_summaries = ingest_documents()

    print_summary(total_chunks, doc_summaries)
    print("\n Documents ingested successfully!")

if __name__ == "__main__":
    main()