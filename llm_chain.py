import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# NEW LangChain RAG imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def setup_qa_chain(vectorstore):

    # Load API key
    deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
    deepseek_base_url = "https://api.deepseek.com/v1"

    # Initialize DeepSeek LLM (via OpenAI-compatible API)
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=512,
        api_key=deepseek_api_key,
        base_url=deepseek_base_url
    )

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the user's question clearly:

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # Build "stuff" document QA chain
    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever = vectorstore.as_retriever()

    # Build final Retrieval-Augmented Generation (RAG) chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_documents_chain=doc_chain
    )

    return rag_chain
