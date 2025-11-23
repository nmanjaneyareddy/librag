# Patched vectorstore.py — searches BOTH data/documents and /mnt/data
# Works with FAISS, sentence-transformers, and your existing app.

import os
from typing import List, Dict, Any
import numpy as np


# Search locations for PDF/TXT/MD files
DOCS_DIRS = [
    "data/documents",                 # Repo folder
    "/mnt/data",                      # Streamlit uploaded files
    "/mount/src/librag/data"          # REAL location of your PDF
]

# Basic chunker
def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return chunks

def _load_all_docs() -> List[Dict[str, Any]]:
    """Walk all DOCS_DIRS and return a list of chunked text documents."""
    items = []
    idx = 0
    from pypdf import PdfReader

    for docs_dir in DOCS_DIRS:
        if not os.path.exists(docs_dir):
            continue

        for root, _, files in os.walk(docs_dir):
            for fname in files:
                path = os.path.join(root, fname)
                lower = fname.lower()

                try:
                    # PDF
                    if lower.endswith(".pdf"):
                        text = []
                        reader = PdfReader(path)
                        for p in reader.pages:
                            try:
                                text.append(p.extract_text() or "")
                            except:
                                continue
                        full_text = "\n".join(text)

                    # Text files
                    elif lower.endswith(".txt") or lower.endswith(".md"):
                        with open(path, "r", encoding="utf-8") as f:
                            full_text = f.read()

                    else:
                        continue

                    # Split into chunks
                    chunks = _chunk_text(full_text)
                    for i, c in enumerate(chunks):
                        items.append({
                            "id": f"{idx}",
                            "text": c,
                            "meta": {"source": path, "chunk": i}
                        })
                        idx += 1

                except Exception as e:
                    print(f"⚠️ Failed loading {path}: {e}")
                    continue

    return items


# --- FAISS vectorstore wrapper ------------------------------------------------

class SimpleFAISSVectorStore:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.metas = []

    def build(self, docs):
        import faiss

        texts = [d["text"] for d in docs]
        if not texts:
            raise ValueError("No documents to build index from.")

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        self.index = index
        self.texts = texts
        self.metas = [d["meta"] for d in docs]

    def similarity_search(self, query, k=4):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)

        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.texts):
                results.append({
                    "page_content": self.texts[idx],
                    "metadata": self.metas[idx],
                    "score": float(score)
                })
        return results

    def as_retriever(self, search_kwargs=None):
        parent = self
        k = 4
        if search_kwargs:
            k = search_kwargs.get("k", 4)

        class Retriever:
            def get_relevant_documents(self, query):
                docs = parent.similarity_search(query, k)
                class DocObj:
                    def __init__(self, text, meta): 
                        self.page_content = text
                        self.metadata = meta
                return [DocObj(d["page_content"], d["metadata"]) for d in docs]

        return Retriever()


# --- Public API ---------------------------------------------------------------

def load_vector_store() -> SimpleFAISSVectorStore:
    docs = _load_all_docs()

    if not docs:
        raise RuntimeError(
            "No documents found in data/documents or /mnt/data.\n"
            "Make sure igidr_library_details.pdf exists in one of these folders."
        )

    store = SimpleFAISSVectorStore()
    store.build(docs)
    return store
