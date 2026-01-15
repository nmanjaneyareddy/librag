# vectorstore.py
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

FAISS_DIR = "faiss_index"

def _embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner="🔄 Building vector store...")
def load_or_create_vector_store():
    """
    Streamlit Cloud–safe:
    - Builds FAISS index in memory
    - Caches it across reruns
    """

    # 👉 Load documents from repo
    loader = PyPDFLoader("data/sample.pdf")  # make sure this exists in GitHub
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = splitter.split_documents(docs)

    embeddings = _embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore
