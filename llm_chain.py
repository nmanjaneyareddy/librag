# llm_chain.py
# Defensive version: returns diagnostics instead of raising when retriever API is unknown.

from typing import Any, Dict, List
import inspect
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

# Try to import RetrievalQA; if not present we'll fallback
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

# Helper to call LLM with common APIs
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


class SimpleRetrievalQA:
    """
    Defensive wrapper. On failure it returns a result dict that contains a clear diagnostic
    instead of raising, so the Streamlit app can continue running and show diagnostics.
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

    def _try_call(self, fn, query: str):
        """Try common function signatures for a candidate method."""
        # try several call signatures; raise if the function raises (so we can try other candidates)
        try:
            try:
                return fn(query, k=self.k)
            except TypeError:
                pass
            try:
                return fn(query, self.k)
            except TypeError:
                pass
            try:
                return fn(query)
            except TypeError:
                pass
            try:
                return fn()
            except TypeError:
                pass
        except Exception:
            # bubble up the exception so callers can continue trying other candidates
            raise
        raise TypeError("Function did not accept common signatures")

    def _fetch_docs_with_diagnostics(self, query: str) -> Dict[str, Any]:
        """
        Attempts many call patterns. Returns dict:
         - success: bool
         - docs: list (if success)
         - tried: list of candidate attribute names considered
         - error: optional error string
        """
        r = self.retriever
        tried = []
        keywords = ("search", "similar", "retrieve", "query", "find", "get_relevant")

        # 1) Known names first
        names = [
            "get_relevant_documents",
            "get_relevant_entries",
            "retrieve",
            "similarity_search",
            "similarity_search_with_score",
            "similarity_search_by_vector",
            "search",
            "query",
            "similar_search",
        ]
        for name in names:
            fn = getattr(r, name, None)
            if callable(fn):
                tried.append(name)
                try:
                    res = self._try_call(fn, query)
                    # normalize similarity_search_with_score
                    if name == "similarity_search_with_score" and isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                        return {"success": True, "docs": [t[0] for t in res][: self.k], "tried": tried}
                    if isinstance(res, list):
                        return {"success": True, "docs": res[: self.k], "tried": tried}
                    if hasattr(res, "__iter__"):
                        return {"success": True, "docs": list(res)[: self.k], "tried": tried}
                except Exception as e:
                    tried.append(f"{name}: raised {type(e).__name__}")

        # 2) Scan attributes that contain keywords
        scanned = []
        for attr in dir(r):
            if any(kw in attr.lower() for kw in keywords):
                fn = getattr(r, attr)
                if callable(fn):
                    scanned.append(attr)
                    try:
                        res = self._try_call(fn, query)
                        if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                            return {"success": True, "docs": [t[0] for t in res][: self.k], "tried": names + scanned}
                        if isinstance(res, list):
                            return {"success": True, "docs": res[: self.k], "tried": names + scanned}
                        if hasattr(res, "__iter__"):
                            return {"success": True, "docs": list(res)[: self.k], "tried": names + scanned}
                    except Exception as e:
                        scanned.append(f"{attr}: raised {type(e).__name__}")

        # 3) If retriever itself is callable
        if callable(r):
            try:
                res = r(query)
                if isinstance(res, list):
                    return {"success": True, "docs": res[: self.k], "tried": names + scanned}
                if hasattr(res, "__iter__"):
                    return {"success": True, "docs": list(res)[: self.k], "tried": names + scanned}
            except Exception as e:
                tried.append(f"callable_retriever: raised {type(e).__name__}")

        # Nothing worked
        return {"success": False, "docs": [], "tried": names + scanned, "error": "No usable retrieval method found."}

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query") or inputs.get("question") or ""
        if not query:
            return {"result": "", "source_documents": [], "diagnostic": {"note": "empty query"}}

        diag = self._fetch_docs_with_diagnostics(query)

        if not diag.get("success", False):
            # Return a helpful error message instead of raising
            diagnostic_text = (
                "LLM chain could not find a retrieval method on the provided vectorstore/retriever.\n\n"
                f"Tried candidates and scanned attributes (some recorded): {diag.get('tried')}\n\n"
                f"Error: {diag.get('error')}\n\n"
                "To fix: tell me which attribute names exist on your vectorstore (paste the list shown below),\n"
                "or run `dir(vectorstore)` in your environment and paste the output here.\n"
            )
            # Also attach shorter diagnostic fields for programmatic inspection
            return {
                "result": diagnostic_text,
                "source_documents": [],
                "diagnostic": {"tried": diag.get("tried"), "error": diag.get("error")},
            }

        docs = diag["docs"]

        # build context text
        parts = []
        for d in docs:
            content = None
            if isinstance(d, dict):
                content = d.get("page_content") or d.get("content") or str(d)
            else:
                content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            parts.append(content)
        context = "\n---\n".join([p for p in parts if p])

        # prepare prompt text (use PromptTemplate if available)
        if PromptTemplate and isinstance(PromptTemplate, type):
            try:
                prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
                prompt_text = prompt.format(context=context, question=query)
            except Exception:
                prompt_text = self.prompt_template.format(context=context, question=query)
        else:
            prompt_text = self.prompt_template.format(context=context, question=query)

        if self.llm is None:
            # Return a clear diagnostic rather than raising so Streamlit UI stays alive
            return {
                "result": "No LLM is available in this environment. Install an LLM or configure LLM_FACTORY in llm_chain.py.",
                "source_documents": docs,
                "diagnostic": {"note": "no_llm"}
            }

        try:
            answer = _call_llm(self.llm, prompt_text)
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "result": f"LLM invocation failed: {type(e).__name__}: {e}\n\nTraceback:\n{tb}",
                "source_documents": docs,
                "diagnostic": {"llm_error": str(e)}
            }

        return {"result": answer, "source_documents": docs, "diagnostic": {"tried": diag.get("tried")}}

def setup_qa_chain(vectorstore, k: int = 4):
    """
    Return a callable QA chain; prefer langchain's RetrievalQA if present, otherwise fallback.
    """
    if vectorstore is None:
        raise ValueError("vectorstore must be provided to setup_qa_chain()")

    # Try to get a retriever wrapper first
    retriever = None
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception:
        try:
            if hasattr(vectorstore, "as_retriever"):
                retriever = vectorstore.as_retriever()
        except Exception:
            retriever = None

    # If we still don't have a retriever, use the vectorstore itself
    if retriever is None:
        retriever = vectorstore

    # If native RetrievalQA available and we have LLM factory, use it
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
            pass

    # Fallback to our defensive SimpleRetrievalQA
    return SimpleRetrievalQA(retriever=retriever, llm_factory=LLM_FACTORY, k=k)
