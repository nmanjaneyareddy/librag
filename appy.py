# /mount/src/librag/llm_chain.py
# DeepSeek-compatible LLM factory + RetrievalQA setup
# Expects Streamlit secrets to contain DEEPSEEK_API_KEY

import streamlit as st

# imports as in your snippet (will need langchain installed)
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def setup_qa_chain(vectorstore):
    """
    Create and return a RetrievalQA chain using a DeepSeek-compatible
    ChatOpenAI wrapper. Assumes Streamlit secrets contains DEEPSEEK_API_KEY.

    vectorstore: a LangChain-style vectorstore (must implement as_retriever()).
    """

    # ✅ DeepSeek API Key & Base URL — stored in Streamlit secrets
    deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
    deepseek_base_url = "https://api.deepseek.com/v1"  # adjust if DeepSeek base differs

    # ✅ Initialize the LLM
    # Note: ChatOpenAI wrapper accepts openai_api_key and openai_api_base kwargs in many setups.
    # If your langchain distribution uses a different constructor, adjust accordingly.
    llm = ChatOpenAI(
        model_name="deepseek-chat",     # change to actual model name if needed
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

    # Build and return RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
