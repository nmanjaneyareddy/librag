# vectorstore.py
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource(show_spinner="🔄 Loading documents and building vector index...")
def load_or_create_vector_store():
    """
    Builds FAISS index in memory (Streamlit Cloud safe).
    Cached across reruns.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ⚠️ Ensure this file exists in GitHub
    loader = PyPDFLoader("data/sample.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    documents = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
