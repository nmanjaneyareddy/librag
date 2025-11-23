# llm_chain.py — safe LLM factory + setup_qa_chain
import os
import streamlit as st
import traceback, sys

# Defensive imports for wrappers we expect
try:
    # preferred: langchain_openai wrapper
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# Optionally, you could add other wrappers here (HuggingFaceHub, etc.)

def _openai_factory():
    """
    Factory that constructs a ChatOpenAI-style LLM.
    Raises a clear RuntimeError if API key or package is missing.
    """
    # Prefer a provider-specific key (DeepSeek) then generic OpenAI key
    key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY in Streamlit secrets or as an environment variable."
        )

    # Use DeepSeek base if provided in secrets (and if DeepSeek implements OpenAI-compatible API)
    base = None
    if "DEEPSEEK_API_KEY" in st.secrets:
        base = st.secrets.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    else:
        base = os.environ.get("OPENAI_API_BASE")  # optional override

    if ChatOpenAI is None:
        raise RuntimeError(
            "Required package langchain_openai (and openai) are not installed. "
            "Add 'langchain-openai' and 'openai' to requirements.txt."
        )

    # Return a configured LLM instance when called
    return ChatOpenAI(
        model_name=st.secrets.get("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(st.secrets.get("LLM_TEMPERATURE", 0.2)),
        max_tokens=int(st.secrets.get("LLM_MAX_TOKENS", 512)),
        openai_api_key=key,
        openai_api_base=base
    )

# Export the factory (callable) — DO NOT call it at import time
LLM_FACTORY = _openai_factory

# Other imports for chains (do these after the factory is defined)
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain  # modern API

def setup_qa_chain(vectorstore):
    """
    Build and return a Retrieval QA chain.
    This will call LLM_FACTORY() at runtime (not at import time).
    """

    # Instantiate the LLM here, with clear error handling
    try:
        llm = LLM_FACTORY()
    except Exception as e:
        # Provide a clear error message that will appear in logs
        msg = (
            "Failed to create LLM via LLM_FACTORY().\n"
            "Reason: {}\n\n"
            "Possible fixes:\n"
            " - Ensure DEEPSEEK_API_KEY or OPENAI_API_KEY is set in Streamlit secrets (Manage app → Settings → Secrets).\n"
            " - Ensure 'langchain-openai' and 'openai' are in requirements.txt and were installed on deploy.\n"
            " - If using DeepSeek, ensure the API is OpenAI-compatible and the base URL is correct.\n"
        ).format(repr(e))
        print(msg, file=sys.stderr)
        traceback.print_exc()
        # Raise a runtime error with the helpful message (so logs show it)
        raise RuntimeError(msg)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the user's question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # Use modern LangChain API if available; falls back if not (we assume create_retrieval_chain exists)
    try:
        qa = create_retrieval_chain(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
    except Exception as e:
        # If create_retrieval_chain is unavailable in your LangChain, raise a helpful message
        msg = (
            "Failed to create retrieval chain using create_retrieval_chain(). "
            "If your LangChain version does not provide this, consider installing 'langchain-classic' "
            "and using RetrievalQA.from_chain_type, or upgrade LangChain.\n"
            "Original error: {}\n"
        ).format(repr(e))
        print(msg, file=sys.stderr)
        traceback.print_exc()
        raise RuntimeError(msg)

    return qa
