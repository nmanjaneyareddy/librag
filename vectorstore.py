# FINAL vectorstore.py (FAISS + SentenceTransformer only, NO LANGCHAIN)

import os
from typing import List, Dict, Any
import numpy as np

# Paths where documents can exist
DOC_PATHS = [
    "data/documents",               # Repo folder
    "/mnt/data",                    # Streamlit uploaded files
    "/mount/src/librag/data"        # Your actual PDF location
]

def _chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return chunks


def _load_all_docs() -> List[Dict[str, Any]]:
    """Load PDFs/TXT/MD from all known paths (NO langchain)."""
    from pypdf import PdfReader
    docs = []
    idx = 0

    for base in DOC_PATHS:
        if not os.path.exists(base):
            continue

        for root, _, files in os.walk(base):
            for fname in files:
                path = os.path.join(root, fname)
                lower = fname.lower()

                try:
                    if lower.endswith(".pdf"):
                        reader = PdfReader(path)
                        text = "\n".join([(p.extract_text() or "") for p in reader.pages])

                    elif lower.endswith(".txt") or lower.endswith(".md"):
                        with open(path, "r", encoding="utf-8") as f:
                            text = f.read()

                    else:
                        continue

                    chunks = _chunk_text(text)
                    for i, c in enumerate(chunks):
                        docs.append({
                            "id": f"{idx}",
                            "text": c,
                            "meta": {"source": path, "chunk": i}
                        })
                        idx += 1

                except Exception:
                    continue

    return docs


class SimpleFAISSVectorStore:
    """FAISS + Sentence-Transformer wrapper compatible with LangChain retriever API."""

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.metas = []

    def build(self, docs):
        import faiss

        texts = [d["text"] for d in docs]
        if not texts:
            raise ValueError("No text to build index.")

        embeddings = self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        self.index = index
        self.texts = texts
        self.metas = [d["meta"] for d in docs]

    def similarity_search(self, query, k=4):
        q = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(q, k)
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
        k = (search_kwargs or {}).get("k", 4)

        class Retriever:
            def get_relevant_documents(self, query):
                docs = parent.similarity_search(query, k)
                class DocObj:
                    def __init__(self, t, m): self.page_content = t; self.metadata = m
                return [DocObj(d["page_content"], d["metadata"]) for d in docs]

        return Retriever()


def load_vector_store():
    docs = _load_all_docs()

    if not docs:
        raise RuntimeError(
            "No documents found in:\n" + "\n".join(DOC_PATHS) +
            "\nMake sure igidr_library_details.pdf exists."
        )

    store = SimpleFAISSVectorStore()
    store.build(docs)
    return store
