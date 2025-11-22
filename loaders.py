# loaders.py
# try community loaders import (package name on PyPI: langchain-community)
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader

# robust import for the splitter (works across different langchain versions)
try:
    # preferred (separate package)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        # newer langchain layout
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        # older / fallback layout (some older examples)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
