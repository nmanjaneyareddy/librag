# /mnt/data/vectorstore.py
# Minimal FAISS vectorstore loader WITHOUT langchain.
# Exposes load_vector_store() and returns an object with:
#   - similarity_search(query, k)
#   - as_retriever() -> object with get_relevant_documents(query)
#
# Requirements (add to requirements.txt): pypdf, sentence-transformers, faiss-cpu, numpy

import os
from typing import List, Dict, Any
import numpy as np

# OPTIONAL: change where your documents live
DOCS_DIR = "data"
INDEX_DIR = "faiss_index_np"  # folder where we save index + metadata (optional)

# Basic chunker: split text into approx chunk_size characters with overlap
def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(0, end - overlap)
    return chunks

def _load_texts_from_docs(docs_dir: str) -> List[Dict[str, Any]]:
    """Walk docs_dir and return list of items {'id': , 'text': , 'meta': }"""
    items = []
    if not os.path.exists(docs_dir):
        return items
    idx = 0
    from pypdf import PdfReader
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            path = os.path.join(root, fname)
            lower = fname.lower()
            try:
                if lower.endswith(".pdf"):
                    text = []
                    reader = PdfReader(path)
                    for p in reader.pages:
                        try:
                            text.append(p.extract_text() or "")
                        except Exception:
                            # skip extract errors for that page
                            pass
                    text = "\n".join(text)
                elif lower.endswith(".txt") or lower.endswith(".md"):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    # skip other file types
                    continue
            except Exception:
                # skip unreadable files
                continue

            chunks = _chunk_text(text)
            for i, c in enumerate(chunks):
                items.append({"id": f"{idx}", "text": c, "meta": {"source": path, "chunk": i}})
                idx += 1
    return items

# Simple wrapper store that holds embeddings, texts and provides similarity_search
class SimpleFAISSVectorStore:
    def __init__(self, embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        # lazy imports to surface missing package errors only when needed
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers not installed. Add 'sentence-transformers' to requirements.") from e

        try:
            import faiss
        except Exception as e:
            raise ImportError("faiss (faiss-cpu) not installed. Add 'faiss-cpu' to requirements.") from e

        self._model_name = embeddings_model_name
        self._sbert = SentenceTransformer(embeddings_model_name)
        self._index = None  # faiss index
        self._embeddings = None  # numpy array (N x D)
        self._metadatas = []  # list of dicts for each vector
        self._texts = []

    def build(self, docs: List[Dict[str, Any]]):
        """Build FAISS index from docs (each doc = {'text':..., 'meta':...})"""
        import faiss

        texts = [d["text"] for d in docs]
        if len(texts) == 0:
            raise ValueError("No documents to build index from.")
        emb = self._sbert.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # ensure float32
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        d = emb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(emb)
        self._index = index
        self._embeddings = emb
        self._texts = texts
        self._metadatas = [d.get("meta", {}) for d in docs]

    def similarity_search(self, query: str, k: int = 4):
        """Return list of simple document-like dicts: {'page_content': text, 'metadata': meta}"""
        q_emb = self._sbert.encode([query], convert_to_numpy=True)
        if q_emb.dtype != np.float32:
            q_emb = q_emb.astype(np.float32)
        if self._index is None:
            raise RuntimeError("Index not built yet.")
        D, I = self._index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._texts):
                continue
            results.append({"page_content": self._texts[idx], "metadata": self._metadatas[idx], "score": float(score)})
        return results

    def as_retriever(self, search_kwargs: dict = None):
        """Return an object exposing get_relevant_documents(query) compatible with some code."""
        parent = self
        class Retriever:
            def __init__(self, parent, k):
                self.parent = parent
                self.k = k or 4
            def get_relevant_documents(self, query):
                docs = parent.similarity_search(query, k=self.k)
                # For compatibility with some flows, return objects with .page_content attribute
                class DocObj:
                    def __init__(self, page_content, metadata):
                        self.page_content = page_content
                        self.metadata = metadata
                    def __repr__(self):
                        return f"<Doc source={self.metadata.get('source')}>"
                return [DocObj(d["page_content"], d["metadata"]) for d in docs]
        k = 4
        if search_kwargs and isinstance(search_kwargs, dict):
            k = search_kwargs.get("k", k)
        return Retriever(parent, k)

# Public loader function
def load_vector_store() -> SimpleFAISSVectorStore:
    """
    Build or load a simple FAISS vectorstore.
    - Looks for documents in DOCS_DIR (pdf, txt, md).
    - Builds an in-memory FAISS index and returns a store object.
    """
    items = _load_texts_from_docs(DOCS_DIR)
    if not items:
        raise RuntimeError(f"No documents found in {DOCS_DIR}. Put PDFs or .txt files there.")

    store = SimpleFAISSVectorStore()
    store.build(items)
    return store
