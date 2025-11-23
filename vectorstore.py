# /mnt/data/vectorstore.py
# Drop-in vectorstore loader for Streamlit + LangChain (FAISS + HuggingFace embeddings)
# - Exposes load_vector_store()
# - Exports `vectorstore` at module level (may be None until loaded)

import os
from typing import Optional

INDEX_DIR = "faiss_index"       # where FAISS index will be saved/loaded
DOCS_DIR = "data/documents"     # default documents folder (adjust if needed)

vectorstore = None  # module-level export for convenience


def _ensure_packages():
    try:
        import langchain
        # import kept lazy
    except Exception as e:
        raise ImportError(
            "Required packages are missing. Make sure requirements.txt includes langchain, langchain-community, sentence-transformers, faiss-cpu, etc."
        ) from e


def _load_existing_index(embeddings):
    """
    Try to load a saved FAISS index from INDEX_DIR.
    Returns the vectorstore or raises if not present.
    """
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"Index folder '{INDEX_DIR}' not found.")
    # FAISS.load_local(index_path, embeddings) signature for langchain-community
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs


def _build_index_from_docs(embeddings):
    """
    Load files from DOCS_DIR, split into documents and build FAISS index.
    Saves index to INDEX_DIR for reuse.
    """
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(DOCS_DIR):
        raise FileNotFoundError(f"Documents folder '{DOCS_DIR}' not found. Put PDFs / .txt files there or change DOCS_DIR.")

    docs = []
    # walk DOCS_DIR and load PDFs and txts
    for root, _, files in os.walk(DOCS_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            lower = fname.lower()
            try:
                if lower.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    loaded = loader.load()
                elif lower.endswith(".txt") or lower.endswith(".md"):
                    loader = TextLoader(path, encoding="utf-8")
                    loaded = loader.load()
                else:
                    # skip unknown extensions
                    continue
                docs.extend(loaded)
            except Exception:
                # don't fail the whole run for one bad file
                continue

    if not docs:
        raise RuntimeError(f"No documents found in {DOCS_DIR} (supported: .pdf, .txt, .md).")

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_out = splitter.split_documents(docs)

    # build FAISS vectorstore
    vs = FAISS.from_documents(docs_out, embeddings)
    # save to disk for future loads
    try:
        vs.save_local(INDEX_DIR)
    except Exception:
        # saving may fail depending on versions — ignore but warn
        pass
    return vs


def load_vector_store() -> object:
    """
    Public loader function used by app.py.
    Returns a vectorstore instance (FAISS-like) that supports .as_retriever() or .similarity_search().
    """
    global vectorstore
    _ensure_packages()

    # Import here to avoid import-time failures when packages missing
    try:
        # embeddings: HuggingFace wrapper (works well with sentence-transformers)
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        raise ImportError("langchain_community.embeddings.HuggingFaceEmbeddings not available. Install langchain-community and sentence-transformers.")

    # create embeddings object
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # 1) try loading existing index
    try:
        vs = _load_existing_index(embeddings)
        vectorstore = vs
        return vs
    except Exception:
        # 2) try to build from docs if index missing
        vs = _build_index_from_docs(embeddings)
        vectorstore = vs
        return vs


# Optional: attempt to load at import time (comment out if you prefer lazy loading)
# try:
#     vectorstore = load_vector_store()
# except Exception:
#     vectorstore = None
