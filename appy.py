# app.py
import streamlit as st
from vectorstore import load_or_create_vector_store
from llm_chain import setup_qa_chain

st.set_page_config(page_title="RAG QA App", layout="wide")
st.title("📚 RAG Question Answering App")

# Load vector store (cached, in-memory)
vectorstore = load_or_create_vector_store()

# Setup QA chain
qa_chain = setup_qa_chain(vectorstore)

question = st.text_input("Ask a question about your documents")

if question:
    with st.spinner("Thinking..."):
        try:
            if hasattr(qa_chain, "invoke"):
                result = qa_chain.invoke({"input": question})
                answer = (
                    result.get("answer")
                    or result.get("result")
                    or result.get("output_text")
                    or str(result)
                )
            else:
                result = qa_chain({"query": question})
                answer = result.get("result") if isinstance(result, dict) else str(result)

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Query failed: {type(e).__name__}: {e}")
