# --- add this adapter class somewhere near the top of llm_chain.py ---

class VectorstoreAdapter:
    """
    Adapter to normalise a vectorstore instance to a retriever exposing:
        get_relevant_documents(query) -> List[Document]
    The adapter tries several plausible call patterns (text query, with/without k,
    with search_type kwarg, with similarity_search_with_score, and, if necessary,
    treats the query as an embedding vector).
    """
    def __init__(self, vs, k: int = 4):
        self.vs = vs
        self.k = k

    def _try_call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError:
            # signature mismatch — caller will try alternatives
            raise
        except Exception:
            # other runtime error — bubble up so caller can try other fn variants
            raise

    def get_relevant_documents(self, query):
        vs = self.vs
        k = self.k

        # 1) Try common text query signatures
        candidates = []

        # direct similarity_search(query, k=...)
        if hasattr(vs, "similarity_search"):
            candidates.append(lambda: self._try_call(vs.similarity_search, query, k))
            # also try without k (some variants)
            candidates.append(lambda: self._try_call(vs.similarity_search, query))
            # try with search_type kwarg (some implementations validate this)
            candidates.append(lambda: self._try_call(vs.similarity_search, query, k=k, search_type="similarity"))

        # similarity_search_with_score(query, k=...)
        if hasattr(vs, "similarity_search_with_score"):
            candidates.append(lambda: self._try_call(vs.similarity_search_with_score, query, k))

        # fallback: retrieve(query) or get_relevant_documents if underlying store provides them
        if hasattr(vs, "retrieve"):
            candidates.append(lambda: self._try_call(vs.retrieve, query, k))
            candidates.append(lambda: self._try_call(vs.retrieve, query))

        if hasattr(vs, "get_relevant_documents"):
            candidates.append(lambda: self._try_call(vs.get_relevant_documents, query))

        if hasattr(vs, "get_relevant_entries"):
            candidates.append(lambda: self._try_call(vs.get_relevant_entries, query))

        # If nothing worked so far, maybe the vectorstore expects an embedding (vector)
        # Try to see if it has an embeddings model attached that can produce an embedding.
        # (This is best-effort; if you don't have embeddings available this will be skipped.)
        if hasattr(vs, "similarity_search_by_vector") and hasattr(vs, "embed_query"):
            # some wrappers provide embed_query or an embeddings property
            try:
                embed = None
                if hasattr(vs, "embed_query"):
                    embed = vs.embed_query(query)
                elif hasattr(vs, "embeddings") and hasattr(vs.embeddings, "embed_query"):
                    embed = vs.embeddings.embed_query(query)
                if embed is not None:
                    candidates.append(lambda: self._try_call(vs.similarity_search_by_vector, embed, k))
            except Exception:
                # ignore and continue
                pass

        # 2) Try all candidates in order, returning the first successful normalized result
        for cand in candidates:
            try:
                res = cand()
                if res is None:
                    continue
                # If similarity_search_with_score returned (doc, score) pairs, strip scores
                if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                    docs = [t[0] for t in res][:k]
                    return docs
                # if generator or list of docs
                if isinstance(res, list):
                    return res[:k]
                if hasattr(res, "__iter__"):
                    return list(res)[:k]
            except TypeError:
                # signature mismatch — try next candidate
                continue
            except Exception:
                # runtime error (e.g. validate_search_type raised); try next candidate
                continue

        # 3) last resort: attempt to call the vs object itself if it's callable
        if callable(vs):
            try:
                res = vs(query)
                if isinstance(res, list):
                    return res[:k]
                if hasattr(res, "__iter__"):
                    return list(res)[:k]
            except Exception:
                pass

        # If we reach here, nothing worked — raise a descriptive error
        raise RuntimeError(
            "VectorstoreAdapter could not call any known retrieval method on the vectorstore. "
            "If you paste the output of `dir(your_vectorstore)` I will add a direct mapping. "
            "Common working methods are: similarity_search(query, k), similarity_search_with_score(query,k), retrieve(query,k)."
        )

# --- now replace the retriever construction inside setup_qa_chain with the adapter ---

def setup_qa_chain(vectorstore, k: int = 4):
    """
    Returns a QA chain usable by the app. Wrap the vectorstore with VectorstoreAdapter
    so the rest of the code can call get_relevant_documents(query).
    """
    if vectorstore is None:
        raise ValueError("vectorstore must be provided to setup_qa_chain()")

    # Wrap the provided vectorstore in the adapter that guarantees get_relevant_documents(...)
    try:
        retriever = VectorstoreAdapter(vectorstore, k=k)
    except Exception:
        # fallback to the original attempt (if adapter construction fails for some reason)
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        except Exception:
            try:
                retriever = vectorstore.as_retriever()
            except Exception:
                retriever = vectorstore

    # If native RetrievalQA available and we have LLM factory, prefer it by handing our adapter as retriever
    if _RETRIEVAL_QA is not None and LLM_FACTORY is not None:
        try:
            llm = LLM_FACTORY()
            qa = _RETRIEVAL_QA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            return qa
        except Exception:
            # fall back to the SimpleRetrievalQA wrapper that expects retriever.get_relevant_documents
            pass

    # Fallback: our SimpleRetrievalQA expects retriever.get_relevant_documents(query)
    return SimpleRetrievalQA(retriever=retriever, llm_factory=LLM_FACTORY, k=k)
