import os
import sys
import pandas as pd

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from glob import glob



# ==========================
# CONFIG: paths
# ==========================



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Files INSIDE data/
CSV_PATH = os.path.join(DATA_DIR, "ai_overall_pred_output.csv")

# Chroma persistent directory (also inside data/)
CHROMA_DIR = os.path.join(DATA_DIR, "vectorstore_canada_ai")

# SentenceTransformer model name
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# ==========================
# Embeddings wrapper
# ==========================

class SBERTEmbeddings(Embeddings):
    """
    Wrap SentenceTransformer so it can be used with LangChain vector stores.
    """

    def __init__(self, model_name: str = SBERT_MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # texts: List[str]
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        # text: str
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


# ==========================
# PDF → Documents
# ==========================

def load_pdf_docs(pdf_path: str):
    """
    Load PDF and split into smaller chunks as LangChain Documents.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()   # list[Document], each page

    print("Splitting PDF into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)

    # Add a simple source tag in metadata
    for i, doc in enumerate(chunks):
        doc.metadata = doc.metadata or {}
        doc.metadata.setdefault("source", "pdf")
        doc.metadata["chunk_id"] = f"pdf_chunk_{i}"

    print(f"PDF chunks created: {len(chunks)}")
    return chunks

def load_csv_docs(csv_path: str):
    """
    Convert each CSV row into a text Document so it can be embedded.
    Adds:
      - metadata["country"] = country name
      - simple aliases for the US and China (so 'American' / 'USA' / 'PRC' etc. still match)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Try to automatically detect the country column
    country_col = None
    for col in df.columns:
        if str(col).lower().startswith("country"):
            country_col = col
            break
    if country_col is None:
        raise ValueError(
            f"Could not find a 'Country' column in CSV. Columns are: {list(df.columns)}"
        )

    docs = []
    print(f"CSV rows: {len(df)} — converting to documents...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        country_name = str(row.get(country_col, "")).strip()

        # --- Build base text from all columns ---
        text_lines = [f"{col}: {row[col]}" for col in df.columns]
        base_text = "\n".join(text_lines)

        # --- Add country aliases to help retrieval for comparison questions ---
        aliases = ""
        cname_lower = country_name.lower()
        if cname_lower in ["united states", "united states of america", "usa", "us"]:
            aliases = "\nAliases: United States, USA, US, America, American"
        elif cname_lower == "canada":
            aliases = "\nAliases: Canada, Canadian"
        elif cname_lower == "china":
            aliases = "\nAliases: China, PRC, Chinese"
        # you can extend this pattern for other countries if needed

        full_text = base_text + aliases

        doc = Document(
            page_content=full_text,
            metadata={
                "source": "csv",
                "row_index": int(idx),
                "country": country_name,
            }
        )
        docs.append(doc)

    print(f"CSV Documents created: {len(docs)}")
    return docs



# ==========================
# Chroma build / update
# ==========================

def build_new_chroma(docs, persist_dir: str):
    """
    Create a new Chroma vector store from Documents and persist it.
    """
    os.makedirs(persist_dir, exist_ok=True)

    print("Initializing SBERT embeddings...")
    embeddings = SBERTEmbeddings()

    print("Building new Chroma DB...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"New Chroma DB saved to: {persist_dir}")
    return vectordb


def load_existing_chroma(persist_dir: str):
    """
    Load an existing Chroma vector store.
    """
    print(f"Loading existing Chroma DB from: {persist_dir}")
    embeddings = SBERTEmbeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    return vectordb


def add_docs_to_chroma(vectordb, docs):
    """
    Add new documents to an existing Chroma DB.
    """
    if not docs:
        print("No new documents to add.")
        return vectordb

    print(f"Adding {len(docs)} documents to existing Chroma DB...")
    vectordb.add_documents(docs)
    vectordb.persist()
    print("Chroma DB updated and persisted.")
    return vectordb


def load_all_pdf_docs(pdf_root: str):
    """
    Search recursively inside `pdf_root` for all PDF files,
    load and chunk them, and return a combined list of Documents.
    """
    pattern = os.path.join(pdf_root, "**/*.pdf")
    pdf_files = sorted(glob(pattern, recursive=True))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_root}")

    all_chunks = []
    print(f"Found {len(pdf_files)} PDF files under {pdf_root}:")
    for path in pdf_files:
        print(f"  - {os.path.relpath(path, pdf_root)}")
        chunks = load_pdf_docs(path)   # uses your existing function
        all_chunks.extend(chunks)

    print(f"Total PDF chunks created: {len(all_chunks)}")
    return all_chunks




# ==========================
# main()
# ==========================

def main():
    """
    Main entry:
    - Load PDF and CSV from DATA_DIR
    - Turn into Documents
    - Build or update Chroma vector store inside DATA_DIR
    """
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR:     {DATA_DIR}")
    print(f"CHROMA_DIR:   {CHROMA_DIR}")

    pdf_docs = load_all_pdf_docs(DATA_DIR)
    csv_docs = load_csv_docs(CSV_PATH)
    all_docs = pdf_docs + csv_docs
    print(f"Total docs (PDF + CSV): {len(all_docs)}")

    # Build or update vector store in data/vectorstore_canada_ai
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        print("\nNo existing Chroma DB found. Creating a new one...")
        _ = build_new_chroma(all_docs, CHROMA_DIR)
    else:
        print("\nExisting Chroma DB detected. Loading and updating...")
        vectordb = load_existing_chroma(CHROMA_DIR)
        _ = add_docs_to_chroma(vectordb, all_docs)

    print("\nDone. Your vectors are ready in:", CHROMA_DIR)


if __name__ == "__main__":
    main()


