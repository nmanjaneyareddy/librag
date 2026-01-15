
# robust vectorstore loader for app.py
import importlib
import importlib.util
import os
import streamlit as st

# local path to the file you uploaded (we'll use this if normal import fails)
VECTORSTORE_LOCAL_PATH = "/mnt/data/vectorstore.py"

def load_vectorstore_module():
    """
    Try to import module named 'vectorstore' normally, else load by file path.
    Returns the module object or raises ImportError.
    """
    try:
        # prefer normal import if repo contains vectorstore.py
        return importlib.import_module("vectorstore")
    except Exception:
        # attempt to load from absolute file path
        if os.path.exists(VECTORSTORE_LOCAL_PATH):
            spec = importlib.util.spec_from_file_location("vectorstore", VECTORSTORE_LOCAL_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            return module
        else:
            raise ImportError(f"Vectorstore module not found at {VECTORSTORE_LOCAL_PATH} and not importable by name.")

# Load the module and obtain a vectorstore object
try:
    vs_module = load_vectorstore_module()
except Exception as e:
    st.error(f"Could not import vectorstore module: {type(e).__name__}: {e}")
    st.stop()

# Try to obtain vectorstore from module: first prefer a loader function, then a top-level variable
vectorstore = None
if hasattr(vs_module, "load_vector_store") and callable(getattr(vs_module, "load_vector_store")):
    try:
        vectorstore = vs_module.load_vector_store()
    except Exception as e:
        st.error(f"load_vector_store() raised: {type(e).__name__}: {e}")
        st.stop()
elif hasattr(vs_module, "vectorstore"):
    vectorstore = getattr(vs_module, "vectorstore")
else:
    # show helpful diagnostic: list attributes found so user can edit file accordingly
    attrs = [a for a in dir(vs_module) if not a.startswith("__")]
    st.error(
        "Vectorstore module found but it does not expose `load_vector_store()` or `vectorstore`.\n"
        "Please add one of them. Attributes present in the module: " + ", ".join(attrs)
    )
    st.stop()

# Final sanity checks
if vectorstore is None:
    st.error("Vectorstore is still None after attempting to load it.")
    st.stop()

# Optional quick diagnostic display
st.write("Vectorstore loaded:", type(vectorstore))
st.write("Has as_retriever?:", hasattr(vectorstore, "as_retriever"))
st.write("Has similarity_search?:", hasattr(vectorstore, "similarity_search"))



# /mount/src/librag/app.py
import streamlit as st
from llm_chain import setup_qa_chain
# use your uploaded vectorstore loader path:
# /mnt/data/vectorstore.py - adjust import if your loader function name differs
try:
    # if your vectorstore.py defines load_vector_store()
    from vectorstore import load_or_create_vector_store
    vectorstore = load_or_create_vector_store()

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
