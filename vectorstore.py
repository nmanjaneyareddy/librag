# vectorstore.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FAISS_DIR = "faiss_index"
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")

def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_store(docs):
    """
    Create FAISS index from documents and save it locally.
    """
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_DIR)
    return vectorstore

def load_vector_store():
    """
    Load FAISS index if it exists, else raise a clear error.
    """
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            "FAISS index not found. Please run create_vector_store(docs) first "
            "to generate faiss_index/index.faiss"
        )

    embeddings = _get_embeddings()
    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
