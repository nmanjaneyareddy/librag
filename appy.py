# /mount/src/librag/app.py
import streamlit as st
from llm_chain import setup_qa_chain
# use your uploaded vectorstore loader path:
# /mnt/data/vectorstore.py - adjust import if your loader function name differs
try:
    # if your vectorstore.py defines load_vector_store()
    from vectorstore import load_vector_store
    vectorstore = load_vector_store()
except Exception:
    try:
        from vectorstore import vectorstore as vectorstore  # maybe a variable was exported
    except Exception:
        st.error("Could not load vectorstore. Ensure /mnt/data/vectorstore.py has load_vector_store() or exports 'vectorstore'.")
        st.stop()

st.title("RAG QA App")

try:
    qa_chain = setup_qa_chain(vectorstore)
except Exception as e:
    st.error(f"setup_qa_chain failed: {type(e).__name__}: {e}")
    st.stop()

user_input = st.text_input("Ask a question about your documents")

if user_input:
    st.write("Querying...")
    try:
        if hasattr(qa_chain, "invoke"):
            out = qa_chain.invoke({"input": user_input})
            # new-style return keys
            answer = out.get("answer") or out.get("result") or out.get("output_text") or str(out)
        else:
            out = qa_chain({"query": user_input})
            if isinstance(out, dict):
                answer = out.get("result") or out.get("answer") or out.get("text") or str(out)
            else:
                answer = str(out)
        st.subheader("Answer:")
        st.write(answer)
        # show diagnostic if present
        if isinstance(out, dict) and out.get("diagnostic"):
            st.caption("Diagnostic:")
            st.json(out.get("diagnostic"))
    except Exception as e:
        st.error(f"Chain invocation failed: {type(e).__name__}: {e}")
        raise
