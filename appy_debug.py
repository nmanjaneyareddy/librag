# DEBUG: temporary import-trace printing — remove after debugging
import traceback
try:
    from loaders import load_documents, split_documents
except Exception:
    traceback.print_exc()
    raise

# If you placed this in appy.py, continue with the rest of your app below...
