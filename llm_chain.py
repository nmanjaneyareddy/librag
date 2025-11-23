import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(vectorstore):
    # ✅ DeepSeek API Key & Base URL
    deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
    deepseek_base_url = "https://api.deepseek.com/v1"  # Use the actual base if different

    # ✅ Initialize the LLM
    llm = ChatOpenAI(
        model_name="deepseek-chat",     # Or the actual model name like "deepseek-coder"
        temperature=0.2,
        max_tokens=512,
        openai_api_key=deepseek_api_key,
        openai_api_base=deepseek_base_url
    )

    # ✅ Prompt
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

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
