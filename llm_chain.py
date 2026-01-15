# llm_chain.py
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def _make_llm():
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key in Streamlit secrets")

    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=512,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


def setup_qa_chain(vectorstore, k: int = 4):
    """
    Modern LCEL-based retrieval QA chain.
    No deprecated classes used.
    """
    llm = _make_llm()

    prompt = PromptTemplate.from_template(
        """Use the following context to answer the question clearly.

Context:
{context}

Question:
{question}

Answer:"""
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
