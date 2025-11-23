# llm_chain.py
# Drop this file into /mount/src/librag/llm_chain.py (overwrite existing).

from typing import Any, Dict, List
import traceback

# PromptTemplate import with fallbacks
try:
    from langchain import PromptTemplate
except Exception:
    try:
        from langchain.prompts import PromptTemplate
    except Exception:
        try:
            from langchain_core.prompts.prompt import PromptTemplate  # type: ignore
        except Exception:
            PromptTemplate = None

# LLM selection with fallbacks
LLM_FACTORY = None
try:
    from langchain.chat_models import ChatOpenAI

    def _make_llm():
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    LLM_FACTORY = _make_llm
except Exception:
    try:
        from langchain.llms import OpenAI

        def _make_llm():
            return OpenAI(temperature=0)

        LLM_FACTORY = _make_llm
    except Exception:
        LLM_FACTORY = None

# Try to import RetrievalQA; set _RETRIEVAL_QA to RetrievalQA or None
_RETRIEVAL_QA = None
try:
    from langchain.chains import RetrievalQA

    _RETRIEVAL_QA = RetrievalQA
except Exception:
    try:
        from langchain.chains.question_answering import RetrievalQA  # type: ignore
        _RETRIEVAL_QA = RetrievalQA
    except Exception:
        _RETRIEVAL_QA = None


# ----------------------
# Helper to call LLM with common APIs
# ----------------------
def _call_llm(llm, prompt_text: str) -> str:
    if hasattr(llm, "predict"):
        try:
            return llm.predict(prompt_text)
        except Exception:
            pass
    try:
        return llm(prompt_text)
    except Exception:
        pass
    if hasattr(llm, "generate"):
        try:
            res = llm.generate([prompt_text])
            gens = getattr(res, "generations", None)
            if gens and isinstance(gens, list) and len(gens) > 0 and len(gens[0]) > 0:
                txt = getattr(gens[0][0], "text", None)
                if txt:
                    return txt
            return str(res)
        except Exception:
            pass
    raise RuntimeError(
        "Unable to call LLM: wrapper does not support predict/__call__/generate the way this code expects."
    )


# ----------------------
# VectorstoreAdapter: normalizes many vectorstore APIs to get_relevant_documents(query)
# ----------------------
class VectorstoreAdapter:
    """
    Adapter to normalize a vectorstore instance to a retriever exposing:
        get_relevant_documents(query) -> List[Document]
    The adapter tries several plausible call patterns (text query, with/without k,
    with search_type kwarg, similarity_search_with_score, and, if necessary,
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

        candidates = []

        # similarity_search(query, k=...)
        if hasattr(vs, "similarity_search"):
            candidates.append(lambda: self._try_call(vs.similarity_search, query, k))
            candidates.append(lambda: self._try_call(vs.similarity_search, query))
            candidates.append(lambda: self._try_call(vs.similarity_search, query, k=k, search_type="similarity"))

        # similarity_search_with_score(query, k=...)
        if hasattr(vs, "similarity_search_with_score"):
            candidates.append(lambda: self._try_call(vs.similarity_search_with_score, query, k))

        # retrieve / get_relevant_documents / get_relevant_entries
        if hasattr(vs, "retrieve"):
            candidates.append(lambda: self._try_call(vs.retrieve, query, k))
            candidates.append(lambda: self._try_call(vs.retrieve, query))
        if hasattr(vs, "get_relevant_documents"):
            candidates.append(lambda: self._try_call(vs.get_relevant_documents, query))
        if hasattr(vs, "get_relevant_entries"):
            candidates.append(lambda: self._try_call(vs.get_relevant_entries, query))

        # similarity_search_by_vector or embed-based call
        if hasattr(vs, "similarity_search_by_vector") and hasattr(vs, "embed_query"):
            try:
                embed = vs.embed_query(query)
                candidates.append(lambda: self._try_call(vs.similarity_search_by_vector, embed, k))
            except Exception:
                pass
        elif hasattr(vs, "similarity_search_by_vector"):
            # try to call by passing the query (some wrappers do text->vector internally)
            candidates.append(lambda: self._try_call(vs.similarity_search_by_vector, query, k))

        # Try candidates in order; return first normalized list of docs
        for cand in candidates:
            try:
                res = cand()
                if res is None:
                    continue
                # If similarity_search_with_score returned (doc, score) pairs, strip scores
                if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                    docs = [t[0] for t in res][:k]
                    return docs
                if isinstance(res, list):
                    return res[:k]
                if hasattr(res, "__iter__"):
                    return list(res)[:k]
            except TypeError:
                # signature mismatch — try next candidate
                continue
            except Exception:
                # runtime error (e.g. validate_search_type raised) — try next candidate
                continue

        # Last-resort: if vs is callable, try calling it
        if callable(vs):
            try:
                res = vs(query)
                if isinstance(res, list):
                    return res[:k]
                if hasattr(res, "__iter__"):
                    return list(res)[:k]
            except Exception:
                pass

        # Nothing worked
        raise RuntimeError(
            "VectorstoreAdapter could not call any known retrieval method on the vectorstore. "
            "If you paste the output of `dir(your_vectorstore)` I will add a direct mapping. "
            "Common working methods are: similarity_search(query, k), similarity_search_with_score(query,k), retrieve(query,k)."
        )


# ----------------------
# SimpleRetrievalQA: fallback QA wrapper that returns diagnostics instead of raising
# ----------------------
class SimpleRetrievalQA:
    """
    Minimal retrieval + LLM wrapper used when langchain's RetrievalQA isn't available.
    Returns diagnostic-friendly dicts instead of letting exceptions propagate to Streamlit.
    """

    def __init__(self, retriever, llm_factory, prompt_template: str = None, k: int = 4):
        self.retriever = retriever
        self.llm = llm_factory() if llm_factory is not None else None
        self.k = k
        self.prompt_template = (
            prompt_template
            or (
                "Use the following extracted context to answer the question as concisely as possible.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        )

    def _fetch_docs_with_diagnostics(self, query: str) -> Dict[str, Any]:
        tried = []
        # If retriever provides get_relevant_documents, call it
        if hasattr(self.retriever, "get_relevant_documents") and callable(self.retriever.get_relevant_documents):
            tried.append("get_relevant_documents")
            try:
                docs = self.retriever.get_relevant_documents(query)
                if docs:
                    return {"success": True, "docs": docs, "tried": tried}
            except Exception as e:
                tried.append(f"get_relevant_documents: raised {type(e).__name__}")

        # If retriever provides a direct similarity_search, call it
        if hasattr(self.retriever, "similarity_search") and callable(self.retriever.similarity_search):
            tried.append("similarity_search")
            try:
                docs = self.retriever.similarity_search(query, k=self.k)
                if docs:
                    return {"success": True, "docs": docs, "tried": tried}
            except Exception as e:
                tried.append(f"similarity_search: raised {type(e).__name__}")

        # If retriever is the VectorstoreAdapter, it will implement get_relevant_documents
        if isinstance(self.retriever, VectorstoreAdapter):
            tried.append("VectorstoreAdapter.get_relevant_documents")
            try:
                docs = self.retriever.get_relevant_documents(query)
                if docs:
                    return {"success": True, "docs": docs, "tried": tried}
            except Exception as e:
                tried.append(f"VectorstoreAdapter: raised {type(e).__name__}")

        # Last resort: try to call underlying retriever in a defensive way
        try:
            if callable(self.retriever):
                tried.append("callable_retriever")
                res = self.retriever(query)
                if isinstance(res, list):
                    return {"success": True, "docs": res[: self.k], "tried": tried}
                if hasattr(res, "__iter__"):
                    return {"success": True, "docs": list(res)[: self.k], "tried": tried}
        except Exception as e:
            tried.append(f"callable_retriever: raised {type(e).__name__}")

        return {"success": False, "docs": [], "tried": tried, "error": "No usable retrieval method found."}

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query") or inputs.get("question") or ""
        if not query:
            return {"result": "", "source_documents": [], "diagnostic": {"note": "empty query"}}

        diag = self._fetch_docs_with_diagnostics(query)

        if not diag.get("success", False):
            diagnostic_text = (
                "LLM chain could not find a retrieval method on the provided vectorstore/retriever.\n\n"
                f"Tried candidates (some recorded): {diag.get('tried')}\n\n"
                f"Error: {diag.get('error')}\n\n"
                "To fix: paste the output of `dir(vectorstore)` here, and I'll map its method to the adapter."
            )
            return {
                "result": diagnostic_text,
                "source_documents": [],
                "diagnostic": {"tried": diag.get("tried"), "error": diag.get("error")},
            }

        docs = diag["docs"]

        parts = []
        for d in docs:
            if isinstance(d, dict):
                content = d.get("page_content") or d.get("content") or str(d)
            else:
                content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            parts.append(content)
        context = "\n---\n".join([p for p in parts if p])

        if PromptTemplate and isinstance(PromptTemplate, type):
            try:
                prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
                prompt_text = prompt.format(context=context, question=query)
            except Exception:
                prompt_text = self.prompt_template.format(context=context, question=query)
        else:
            prompt_text = self.prompt_template.format(context=context, question=query)

        if self.llm is None:
            return {
                "result": "No LLM is available in this environment. Install an LLM or configure LLM_FACTORY in llm_chain.py.",
                "source_documents": docs,
                "diagnostic": {"note": "no_llm"},
            }

        try:
            answer = _call_llm(self.llm, prompt_text)
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "result": f"LLM invocation failed: {type(e).__name__}: {e}\n\nTraceback:\n{tb}",
                "source_documents": docs,
                "diagnostic": {"llm_error": str(e)},
            }

        return {"result": answer, "source_documents": docs, "diagnostic": {"tried": diag.get("tried")}}


# ----------------------
# setup_qa_chain: returns RetrievalQA if available otherwise fallback to SimpleRetrievalQA with an adapter
# ----------------------
def setup_qa_chain(vectorstore, k: int = 4):
    """
    Return a callable QA chain; prefer langchain's RetrievalQA if present, otherwise fallback.
    """
    if vectorstore is None:
        raise ValueError("vectorstore must be provided to setup_qa_chain()")

    # Wrap the vectorstore in our adapter so we get get_relevant_documents(query)
    try:
        retriever = VectorstoreAdapter(vectorstore, k=k)
    except Exception:
        # if adapter construction fails, fall back to asking the vectorstore for a retriever
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        except Exception:
            try:
                retriever = vectorstore.as_retriever()
            except Exception:
                retriever = vectorstore

    # If langchain's RetrievalQA is present, try to use it (pass our adapter as retriever)
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
            # fall through to fallback wrapper
            pass

    # Fallback: use SimpleRetrievalQA (it expects retriever.get_relevant_documents or VectorstoreAdapter)
    return SimpleRetrievalQA(retriever=retriever, llm_factory=LLM_FACTORY, k=k)
