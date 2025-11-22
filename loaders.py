# loaders.py (top)
import os
import traceback

# community loaders (PyPI package: langchain-community)
try:
    from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
except Exception as e:
    print("ERROR importing langchain_community.document_loaders:", type(e).__name__, e)
    traceback.print_exc()
    raise

# robust import for RecursiveCharacterTextSplitter (covers common layouts)
try:
    # separate package (preferred in many installs)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("Imported RecursiveCharacterTextSplitter from langchain_text_splitters")
except Exception:
    try:
        # newer langchain layout
        from langchain.text_splitters import RecursiveCharacterTextSplitter
        print("Imported RecursiveCharacterTextSplitter from langchain.text_splitters")
    except Exception:
        try:
            # older/fallback layout
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            print("Imported RecursiveCharacterTextSplitter from langchain.text_splitter")
        except Exception as e:
            print("ERROR importing any RecursiveCharacterTextSplitter variant:", type(e).__name__, e)
            traceback.print_exc()
            raise
