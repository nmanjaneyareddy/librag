# vectorstore.py (top) - robust imports for embeddings
import traceback
import sys

HuggingFaceEmbeddings = None
_import_errors = []

def _note(msg):
    print("VECTORSTORE:", msg, file=sys.stderr)

# Try common LangChain / HF embedding import locations
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    _note("Imported HuggingFaceEmbeddings from langchain.embeddings")
    HuggingFaceEmbeddings = HuggingFaceEmbeddings
except Exception as e:
    _import_errors.append(("langchain.embeddings", type(e).__name__, str(e)))

try:
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings as HF2
    _note("Imported HuggingFaceEmbeddings from langchain.embeddings.huggingface")
    HuggingFaceEmbeddings = HF2
except Exception as e:
    _import_errors.append(("langchain.embeddings.huggingface", type(e).__name__, str(e)))

try:
    # Some installs expose a HuggingFaceHub embedding wrapper
    from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings as HFHub
    _note("Imported HuggingFaceHubEmbeddings as fallback from langchain.embeddings.huggingface_hub")
    HuggingFaceEmbeddings = HFHub
except Exception as e:
    _import_errors.append(("langchain.embeddings.huggingface_hub", type(e).__name__, str(e)))

# If still None, print helpful instructions and full errors, then raise
if HuggingFaceEmbeddings is None:
    _note("ERROR: Could not import any HuggingFace embeddings class. Import attempts and errors:")
    for mod, errtype, err in _import_errors:
        _note(f" - {mod} -> {errtype}: {err}")
    _note("Common fixes (pick one):")
    _note("  1) Install HF deps: pip install transformers sentence-transformers huggingface-hub torch accelerate")
    _note("  2) Or use a lighter alternative like OpenAI embeddings (requires OPENAI_API_KEY and openai package).")
    _note("  3) If you use HuggingFaceHub embeddings, ensure HUGGINGFACEHUB_API_TOKEN is set.")
    traceback.print_stack()
    raise ImportError("HuggingFaceEmbeddings import failed; see logs for details")
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
