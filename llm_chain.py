# llm_chain.py
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def _make_llm():
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or DEEPSEEK_API_KEY in Streamlit secrets")

    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=512,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


def setup_qa_chain(vectorstore, k: int = 4):
    if vectorstore is None:
        raise ValueError("Vectorstore cannot be None")

    llm = _make_llm()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question clearly.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
