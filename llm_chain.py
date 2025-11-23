# /mount/src/librag/llm_chain.py
import streamlit as st
from typing import Any, Dict

# Try ChatOpenAI from langchain_openai (modern) or fallback to langchain.chat_models
try:
    from langchain_openai import ChatOpenAI  # preferred for newer installs
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI  # fallback
    except Exception as e:
        raise ImportError("ChatOpenAI import failed; install langchain_openai or compatible langchain.") from e

# PromptTemplate fallback
PromptTemplate = None
for attempt in (
    "from langchain import PromptTemplate",
    "from langchain.prompts import PromptTemplate",
    "from langchain.prompts.prompt import PromptTemplate",
    "from langchain_core.prompts.prompt import PromptTemplate",
):
    try:
        exec(attempt, globals())
        if "PromptTemplate" in globals():
            break
    except Exception:
        pass

# Try RetrievalQA in common locations (0.1.x)
_RETRIEVAL_QA = None
try:
    from langchain.chains.retrieval_qa import RetrievalQA
    _RETRIEVAL_QA = RetrievalQA
except Exception:
    try:
        from langchain.chains import RetrievalQA
        _RETRIEVAL_QA = RetrievalQA
    except Exception:
        _RETRIEVAL_QA = None

# If RetrievalQA not available, try new factory (will be absent for 0.1.20 but we try defensively)
_CREATE_RETRIEVAL_CHAIN = None
_CREATE_STUFF_DOCS_CHAIN = None
if _RETRIEVAL_QA is None:
    try:
        from langchain.chains import create_retrieval_chain  # type: ignore
        from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
        _CREATE_RETRIEVAL_CHAIN = create_retrieval_chain
        _CREATE_STUFF_DOCS_CHAIN = create_stuff_documents_chain
    except Exception:
        _CREATE_RETRIEVAL_CHAIN = None
        _CREATE_STUFF_DOCS_CHAIN = None


def _make_deepseek_llm():
    """Initialize ChatOpenAI using DeepSeek-compatible OpenAI base URL and key from Streamlit secrets."""
    api_key = None
    # prefer DEEPSEEK_API_KEY then OPENAI_API_KEY
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in Streamlit secrets.")
    base_url = "https://api.deepseek.com/v1"

    # Try a couple common constructor signatures
    try:
        # langchain_openai expects model / api_key / base_url in some versions
        return ChatOpenAI(model="deepseek-chat", temperature=0.2, max_tokens=512, api_key=api_key, base_url=base_url)
    except TypeError:
        # older versions expect model_name / openai_api_key / openai_api_base
        return ChatOpenAI(model_name="deepseek-chat", temperature=0.2, max_tokens=512, openai_api_key=api_key, openai_api_base=base_url)


def setup_qa_chain(vectorstore, k: int = 4):
    """
    Build and return a QA chain compatible with LangChain 0.1.x and reasonably robust to nearby versions.
    vectorstore: object that supports .as_retriever() or similarity_search(...)
    """
    if vectorstore is None:
        raise ValueError("vectorstore must be provided to setup_qa_chain()")

    llm = _make_deepseek_llm()

    # Prompt template (use PromptTemplate if available)
    prompt_template = (
        "Use the following context to answer the user's question clearly and concisely.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
    )
    prompt = None
    if PromptTemplate:
        try:
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        except Exception:
            prompt = None

    # Build retriever
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception:
        try:
            retriever = vectorstore.as_retriever()
        except Exception:
            retriever = vectorstore  # fallback to raw vectorstore

    # 1) If RetrievalQA is available, use it (expected for langchain==0.1.20)
    if _RETRIEVAL_QA is not None:
        kwargs = {}
        if prompt is not None:
            kwargs["chain_type_kwargs"] = {"prompt": prompt}
        try:
            return _RETRIEVAL_QA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                **kwargs,
            )
        except Exception:
            # fall through to factory or fallback
            pass

    # 2) If new factory present, use it
    if _CREATE_RETRIEVAL_CHAIN is not None and _CREATE_STUFF_DOCS_CHAIN is not None:
        try:
            combine_chain = _CREATE_STUFF_DOCS_CHAIN(llm=llm) if _CREATE_STUFF_DOCS_CHAIN else None
            chain = _CREATE_RETRIEVAL_CHAIN(retriever=retriever, combine_documents_chain=combine_chain)
            return chain
        except Exception:
            pass

    # 3) Last-resort fallback simple callable chain
    class SimpleQA:
        def __init__(self, retriever, llm, k=4, prompt_template=prompt_template):
            self.retriever = retriever
            self.llm = llm
            self.k = k
            self.prompt_template = prompt_template

        def _get_docs(self, query):
            # try common retriever methods
            candidates = (
                "get_relevant_documents",
                "retrieve",
                "similarity_search",
                "similarity_search_with_score",
            )
            for name in candidates:
                fn = getattr(self.retriever, name, None)
                if callable(fn):
                    try:
                        res = fn(query, k=self.k)
                    except TypeError:
                        try:
                            res = fn(query)
                        except Exception:
                            continue
                    # strip scores if present
                    if isinstance(res, (list, tuple)) and len(res) and isinstance(res[0], (list, tuple)):
                        return [t[0] for t in res][: self.k]
                    if isinstance(res, list):
                        return res[: self.k]
            # last resort: if retriever is vectorstore exposing similarity_search
            fn = getattr(self.retriever, "similarity_search", None)
            if callable(fn):
                try:
                    return fn(query, k=self.k)
                except Exception:
                    try:
                        return fn(query)
                    except Exception:
                        pass
            raise RuntimeError("No retriever method found on vectorstore; ensure it supports as_retriever() or similarity_search().")

        def __call__(self, inputs: Dict[str, Any]):
            query = inputs.get("query") or inputs.get("input") or inputs.get("question") or ""
            if not query:
                return {"result": "", "source_documents": []}
            docs = self._get_docs(query)
            parts = []
            for d in docs:
                if isinstance(d, dict):
                    parts.append(d.get("page_content") or d.get("content") or str(d))
                else:
                    parts.append(getattr(d, "page_content", None) or getattr(d, "content", None) or str(d))
            context = "\n---\n".join([p for p in parts if p])
            prompt_text = self.prompt_template.format(context=context, question=query) if hasattr(self, "prompt_template") else self.prompt_template.format(context=context, question=query)
            if hasattr(self.llm, "predict"):
                answer = self.llm.predict(prompt_text)
            else:
                answer = self.llm(prompt_text)
            return {"result": answer, "source_documents": docs}

    return SimpleQA(retriever, llm, k=k, prompt_template=prompt_template)
