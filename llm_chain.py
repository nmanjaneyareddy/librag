# llm_chain.py
# Replace the whole file with this content (ensures robust retriever discovery).

from typing import Any, Dict, List
import inspect

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
    Minimal retrieval + LLM wrapper used when langchain's RetrievalQA isn't available.
    Uses a very defensive _fetch_docs that attempts many call patterns.
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
        """Try calling fn with several plausible signatures and return result or raise."""
        try:
            # try (query, k=...)
            try:
                return fn(query, k=self.k)
            except TypeError:
                pass
            # try (query, k)
            try:
                return fn(query, self.k)
            except TypeError:
                pass
            # try (query,)
            try:
                return fn(query)
            except TypeError:
                pass
            # try no args
            try:
                return fn()
            except TypeError:
                pass
        except Exception:
            # If the function raised for this query, propagate up to allow trying another function.
            raise
        raise TypeError("Function did not accept common signatures")

    def _fetch_docs(self, query: str) -> List[Any]:
        """
        Very defensive method detection:
         - tries known method names first
         - then scans all attributes for callable names containing keywords
         - tries calling with multiple signatures
        """
        r = self.retriever

        # 1) Known names first (fast)
        candidates = [
            ("get_relevant_documents", True),
            ("get_relevant_entries", True),
            ("retrieve", True),
            ("similarity_search", True),
            ("similarity_search_with_score", True),
            ("similarity_search_by_vector", True),
            ("search", True),
            ("query", True),
            ("similar_search", True),
        ]
        for name, _ in candidates:
            fn = getattr(r, name, None)
            if callable(fn):
                try:
                    res = self._try_call(fn, query)
                    # if similarity_search_with_score returns (doc, score) tuples, strip scores
                    if name == "similarity_search_with_score" and isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                        return [t[0] for t in res][: self.k]
                    # normalize lists
                    if isinstance(res, list):
                        return res[: self.k]
                    # if generator, convert
                    if hasattr(res, "__iter__"):
                        return list(res)[: self.k]
                except Exception:
                    # try next candidate
                    pass

        # 2) Scan attributes for anything that looks like a search/retrieve function
        keywords = ("search", "similar", "retrieve", "query", "find", "get_relevant")
        for attr in dir(r):
            lname = attr.lower()
            if any(kw in lname for kw in keywords):
                fn = getattr(r, attr)
                if callable(fn):
                    # de-prioritize private/protected attributes
                    if attr.startswith("_"):
                        continue
                    try:
                        res = self._try_call(fn, query)
                        # If returns tuples with score, strip
                        if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                            return [t[0] for t in res][: self.k]
                        if isinstance(res, list):
                            return res[: self.k]
                        if hasattr(res, "__iter__"):
                            return list(res)[: self.k]
                    except Exception:
                        # keep trying other attributes
                        pass

        # 3) If retriever itself is callable, try calling it
        if callable(r):
            try:
                res = r(query)
                if isinstance(res, list):
                    return res[: self.k]
                if hasattr(res, "__iter__"):
                    return list(res)[: self.k]
            except Exception:
                pass

        # Nothing worked — raise informative error with some diagnostics
        # Provide a short diagnostic listing candidate attribute names we tried
        tried_attrs = [a for a in dir(r) if any(kw in a.lower() for kw in keywords)]
        raise RuntimeError(
            "Retriever does not expose a known API for fetching documents. "
            "Tried common names and scanned attributes. Candidate callable attributes matching "
            f"{keywords} found: {tried_attrs}. If your vectorstore uses a custom API, update llm_chain.py "
            "to map its retrieval method to a list of documents."
        )

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query") or inputs.get("question") or ""
        if not query:
            return {"result": "", "source_documents": []}

        docs = self._fetch_docs(query)

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
            raise ImportError("No LLM available. Install an LLM or update llm_chain.py to use another backend.")

        answer = _call_llm(self.llm, prompt_text)
        return {"result": answer, "source_documents": docs}


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

    # Fallback to SimpleRetrievalQA
    return SimpleRetrievalQA(retriever=retriever, llm_factory=LLM_FACTORY, k=k)
