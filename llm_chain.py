# llm_chain.py
# Replace your current llm_chain.py with this file.

from typing import Any, Dict, List

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
    Works with retrievers exposing any of the common method names:
      - get_relevant_documents(query)
      - get_relevant_entries(query)
      - retrieve(query)
      - similarity_search(query, k=...)
      - similarity_search_with_score(query, k=...)
      - similarity_search_by_vector(...) (less common)
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

    def _fetch_docs(self, query: str) -> List[Any]:
        """
        Normalise many different retriever APIs to a list of document-like objects.
        """
        r = self.retriever

        # If object is a retriever wrapper (common), try known retriever method names first
        # 1. LangChain retriever standard
        if hasattr(r, "get_relevant_documents"):
            return r.get_relevant_documents(query)[: self.k]

        # 2. Some libraries use this
        if hasattr(r, "get_relevant_entries"):
            return r.get_relevant_entries(query)[: self.k]

        # 3. Common alternative name
        if hasattr(r, "retrieve"):
            try:
                return r.retrieve(query)[: self.k]
            except TypeError:
                # maybe retrieve(query, k=...) signature
                try:
                    return r.retrieve(query, k=self.k)[: self.k]
                except Exception:
                    pass

        # 4. Many vectorstores expose similarity_search
        if hasattr(r, "similarity_search"):
            try:
                return r.similarity_search(query, k=self.k)
            except TypeError:
                # some signatures differ
                try:
                    return r.similarity_search(query)[: self.k]
                except Exception:
                    pass

        # 5. similarity_search_with_score returns tuples (doc, score)
        if hasattr(r, "similarity_search_with_score"):
            try:
                res = r.similarity_search_with_score(query, k=self.k)
                # return only the docs
                return [t[0] for t in res][: self.k]
            except Exception:
                pass

        # 6. If it's the underlying vectorstore object (e.g., FAISS) with no retriever wrapper,
        # try calling similarity_search directly on it
        if hasattr(r, "similarity_search_by_vector"):
            try:
                return r.similarity_search_by_vector(query, k=self.k)
            except Exception:
                pass

        # 7. If none of the above worked, fall back to calling a generic method if present
        for candidate in ("search", "query", "similar_search"):
            if hasattr(r, candidate):
                fn = getattr(r, candidate)
                try:
                    return fn(query)[: self.k]
                except Exception:
                    pass

        # Last-ditch: if retriever is a callable (rare), call it
        if callable(r):
            try:
                res = r(query)
                if isinstance(res, list):
                    return res[: self.k]
            except Exception:
                pass

        # Nothing worked — raise informative error
        raise RuntimeError(
            "Retriever does not expose a known API for fetching documents. "
            "Expected one of: get_relevant_documents, get_relevant_entries, retrieve, "
            "similarity_search, similarity_search_with_score, similarity_search_by_vector."
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
        # if as_retriever exists but raised, try without kwargs
        try:
            if hasattr(vectorstore, "as_retriever"):
                retriever = vectorstore.as_retriever()
        except Exception:
            retriever = None

    # If we still don't have a retriever, try to use the vectorstore object directly
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
            # fall through to fallback
            pass

    # Fallback to SimpleRetrievalQA
    return SimpleRetrievalQA(retriever=retriever, llm_factory=LLM_FACTORY, k=k)
