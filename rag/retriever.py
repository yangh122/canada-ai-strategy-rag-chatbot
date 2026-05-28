# rag/retriever.py
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "vectorstore_canada_ai")

# Must match SBERT_MODEL_NAME in ingest.py
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class STEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def get_retriever(k: int = 20, model_name: str = EMBED_MODEL_NAME):
    """
    Return a retriever over the Canada AI vector store.
    We use a larger k (20) so multi-country questions are more likely to
    retrieve both countries.
    """
    embeddings = STEmbeddings(model_name=model_name)

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Larger k improves recall for multi-country comparisons
    return vectordb.as_retriever(search_kwargs={"k": k})

# rag/retriever.py (add this)

def get_vectordb(model_name: str = EMBED_MODEL_NAME):
    """
    Return the underlying Chroma vector store (not a retriever),
    so tools can run country-filtered searches.
    """
    embeddings = STEmbeddings(model_name=model_name)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb



