# loaders.py
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

def load_documents():
    docs = []

    # Load PDF
    pdf_path = "data/igidr_library_details.pdf"
    if os.path.exists(pdf_path):
        pdf_loader = PyPDFLoader(pdf_path)
        docs += pdf_loader.load()
    else:
        print(f"⚠️ PDF file not found at {pdf_path}")

    # Load HTML using built-in parser
    html_path = "data/li.html"
    if os.path.exists(html_path):
        html_loader = BSHTMLLoader(html_path, bs_kwargs={"features": "html.parser"})
        docs += html_loader.load()
    else:
        print(f"⚠️ HTML file not found at {html_path}")

    if not docs:
        raise ValueError("No valid documents found in the data/ directory.")
    
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)
