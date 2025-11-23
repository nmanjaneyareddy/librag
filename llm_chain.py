# llm_chain.py
# Add this file next to appy.py

# PromptTemplate import with fallbacks for different langchain versions
try:
    # preferred modern, top-level import in many langchain versions
    from langchain import PromptTemplate
except Exception:
    try:
        from langchain.prompts import PromptTemplate
    except Exception:
        # langchain-core location fallback
        from langchain_core.prompts.prompt import PromptTemplate  # type: ignore

# LLM selection with fallbacks
LLM = None
try:
    # prefer chat models
    from langchain.chat_models import ChatOpenAI
    def _make_llm():
        # if the user has OPENAI_API_KEY set this will work; adjust model_name if needed
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    LLM = _make_llm
except Exception:
    try:
        from langchain.llms import OpenAI
        def _make_llm():
            return OpenAI(temperature=0)
        LLM = _make_llm
    except Exception:
        LLM = None

# RetrievalQA and chain utilities
try:
    # preferred modern import
    from langchain.chains import RetrievalQA
except Exception:
    # older variants may also expose RetrievalQA in the same place; if this fails,
    # the import error will surface and the user should install a compatible langchain
    from langchain.chains import RetrievalQA  # re-raise if not available

def setup_qa_chain(vectorstore, k: int = 4):
    """
    Create and return a RetrievalQA chain for the given vectorstore.
    The returned object supports being called like: qa_chain({"query": "..."})
    and returns a dict with keys like "result" and "source_documents".
    """
    if vectorstore is None:
        raise ValueError("vectorstore must be provided to setup_qa_chain()")

    # make retriever (adjust search params if you want different behaviour)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # instantiate LLM (if available)
    if LLM is None:
        raise ImportError(
            "No supported LLM found. Install langchain and an LLM (e.g. openai) "
            "or modify llm_chain.py to use a different model."
        )

    llm = LLM()

    # Basic prompt — you can customize this
    prompt = PromptTemplate(
        template=(
            "You are an assistant that answers questions from the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer concisely and cite sources when applicable."
        ),
        input_variables=["context", "question"],
    )

    # Build RetrievalQA chain using the LLM and retriever.
    # "from_chain_type" is convenient and commonly supported.
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",           # or "map_reduce" / "refine" depending on your needs
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    except TypeError:
        # fallback if chain_type_kwargs not accepted by older langchain versions
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

    return qa_chain
