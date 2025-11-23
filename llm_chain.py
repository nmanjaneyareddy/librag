# llm_chain.py
# Place this next to appy.py

from typing import Any, Dict, List

# PromptTemplate import with fallbacks
try:
    from langchain import PromptTemplate
except Exception:
    try:
        from langchain.prompts import PromptTemplate
    except Exception:
        # last-ditch fallback (some distributions use langchain_core)
        try:
            from langchain_core.prompts.prompt import PromptTemplate  # type: ignore
        except Exception:
            PromptTemplate = None  # we'll build strings directly if missing

# LLM selection with fallbacks
LLM_FACTORY = None
try:
    from langchain.chat_models import ChatOpenAI

    def _make_llm():
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    LLM_FACTORY = _make_llm
except Exception:
    try:
        # older/alternate API
        from langchain.llms import OpenAI

        def _make_llm():
            return OpenAI(temperature=0)

        LLM_FACTORY = _make_llm
    except Exception:
        LLM_FACTORY = None

# Try to import RetrievalQA; if not available we'll provide a simple fallback implementation
_RETRIEVAL_QA = None
try:
    from langchain.chains import RetrievalQA

    _RETRIEVAL_QA = RetrievalQA
except Exception:
    # try alternative common locations (some versions differ)
    try:
        from langchain.chains.question_answering import RetrievalQA  # type: ignore
        _RETRIEVAL_QA = RetrievalQA
    except Exception:
        _RETRIEVAL_QA = None

# Small helper to call different LLM wrapper APIs safely
def _call_llm(llm, prompt_text: str) -> str:
    """
    Try several common ways to call a LangChain LLM wrapper and return text.
    """
    # 1) predict
    if hasattr(llm, "predict"):
        try:
            return llm.predict(prompt_text)
        except Exception:
            pass

    # 2) __call__
    try:
        return llm(prompt_text)  # many wrappers implement __call__
    except Exception:
        pass

    # 3) generate -> extract text
    if hasattr(llm, "generate"):
        try:
            res = llm.generate([prompt_text])
            # try to extract a sensible text representation
            # res.generations is usually a list of list of Generation objects
            gens = getattr(res, "generations", None)
            if gens and isinstance(gens, list) and len(gens) > 0 and len(gens[0]) > 0:
                txt = getattr(gens[0][0], "text", None)
                if txt:
                    return txt
            # some versions put text elsewhere:
            text = str(res)
            return text
        except Exception:
            pass

    # fallback: raise informative error
    raise RuntimeError(
        "Unable to call LLM: wrapper does not support predict/__call__/generate the way this code expects."
    )

# Fallback RetrievalQA-like wrapper
class SimpleRetrievalQA:
    """
    Minimal retrieval + LLM wrapper used when langchain's RetrievalQA isn't available.
    Expects a retriever (object with get_relevant_documents(query)->List[Document])
    and an llm factory (callable returning an llm wrapper).
    """

    def __init__(self, retriever, llm_factory, prompt_template: str = None, k: int = 4):
        self.retriever = retriever
        self.llm = llm_factory() if llm_factory is not None else None
        self.k = k
        # default prompt if PromptTemplate isn't available
        self.prompt_template = (
            prompt_template
            or (
                "Use the following extracted context to answer the question as concisely as possible.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        )

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query") or inputs.get("question") or ""
        if not query:
            return {"result": "", "source_documents": []}

        # fetch documents (try several retriever methods)
        docs = []
        if hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(query)[: self.k]
        elif hasattr(self.retriever, "get_relevant_entries"):
            docs = self.retriever.get_relevant_entries(query)[: self.k]
        elif hasattr(self.retriever, "retrieve"):
            docs = self.retriever.retrieve(query)[: self.k]
        else:
            raise RuntimeError("Retriever does not expose a known API for fetching documents.")

        # build context text
        parts = []
        for d in docs:
            # Documents may be dict-like or objects with .page_content / .content
            content = None
            if isinstance(d, dict):
                content = d.get("page_content") or d.get("content") or str(d)
            else:
                content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            parts.append(content)
        context = "\n---\n".join([p for p in parts if p])

        # apply prompt
        if PromptTemplate and isinstance(PromptTemplate, type):
            # If we have PromptTemplate class, we can use it to render
            try:
                prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
                prompt_text = prompt.format(context=context, question=query)
            except Exception:
                prompt_text = self.prompt_template.format(context=context, question=query)
        else:
            prompt_text = self.prompt_template.format(context=context, question=query)

        # call llm
        if self.llm is None:
            raise ImportError("No LLM available. Install OpenAI/ChatOpenAI or update llm_chain.py to use another LLM.")

        answer = _call_llm(self.llm, prompt_text)
        return {"result": answer, "source_documents": docs}


def setup_qa_chain(vectorstore, k: int = 4):
    """
    Returns a callable QA chain. Preferred: uses langchain.RetrievalQA if available.
    Fallback: uses SimpleRetrievalQA to avoid import-time failures.
    """
    if vectorstore is None:
        raise ValueError("vectorstore must be provided to setup_qa_chain()")

    # make retriever
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception:
        # some vectorstores expose a different method name
        if hasattr(vectorstore, "as_retriever"):
            retriever = vectorstore.as_retriever()
        else:
            # try building a simple retriever-like wrapper if vectorstore has similarity_search
            if hasattr(vectorstore, "similarity_search"):
                class _SimpleRetriever:
                    def __init__(self, vs, k):
                        self.vs = vs
                        self.k = k
                    def get_relevant_documents(self, q):
                        return self.vs.similarity_search(q, k=self.k)
                retriever = _SimpleRetriever(vectorstore, k)
            else:
                raise

    # If native RetrievalQA is available, use it (clean integration)
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
            # if something goes wrong, fall back to simple wrapper below
            pass

    # Fallback: use our SimpleRetrievalQA
    return SimpleRetrievalQA(retriever=retriever, llm_factory=LLM_FACTORY, k=k)
