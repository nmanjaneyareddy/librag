# vectorstore.py (top) — replace the HuggingFace import with this block
import traceback

# Try the common LangChain embeddings import locations for HuggingFaceEmbeddings
HuggingFaceEmbeddings = None
_import_errors = []

try:
    # common: direct import in newer langchain
    from langchain.embeddings import HuggingFaceEmbeddings
    HuggingFaceEmbeddings = HuggingFaceEmbeddings
    print("Imported HuggingFaceEmbeddings from langchain.embeddings")
except Exception as e:
    _import_errors.append(("langchain.embeddings", type(e).__name__, str(e)))
    try:
        # some installs expose submodule
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
        HuggingFaceEmbeddings = HuggingFaceEmbeddings
        print("Imported HuggingFaceEmbeddings from langchain.embeddings.huggingface")
    except Exception as e2:
        _import_errors.append(("langchain.embeddings.huggingface", type(e2).__name__, str(e2)))
        try:
            # fallback: older examples
            from langchain.embeddings import HuggingFaceHubEmbeddings as HuggingFaceEmbeddings
            print("Imported HuggingFaceHubEmbeddings as HuggingFaceEmbeddings from langchain.embeddings")
        except Exception as e3:
            _import_errors.append(("langchain.embeddings.HF fallback", type(e3).__name__, str(e3)))

if HuggingFaceEmbeddings is None:
    print("ERROR: Could not import HuggingFaceEmbeddings. Import attempts and errors:")
    for mod, errtype, err in _import_errors:
        print(f" - {mod} -> {errtype}: {err}")
    print("Common fixes: add 'transformers', 'sentence-transformers', 'huggingface-hub', and 'langchain-text-splitters' to requirements, or install a LangChain version that exposes the class.")
    traceback.print_stack()  # prints stack to logs for debugging
    raise ImportError("HuggingFaceEmbeddings import failed; see logs for details")

# rest of vectorstore.py continues...
