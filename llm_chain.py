# old
# llm = ChatOpenAI(...)

# new
llm = LLM_FACTORY()

# llm_chain.py (top) — auto-detection factory
import os, streamlit as st, traceback

# Try OpenAI wrapper
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# Try HuggingFace Hub wrapper (hosted)
try:
    from langchain_hub import HuggingFaceHub  # example; adjust to actual wrapper you use
except Exception:
    HuggingFaceHub = None

def _detect_factory():
    # 1) DeepSeek/OpenAI-compatible
    key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key and ChatOpenAI is not None:
        base = "https://api.deepseek.com/v1" if "DEEPSEEK_API_KEY" in st.secrets else os.environ.get("OPENAI_API_BASE")
        return lambda: ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=key, openai_api_base=base)

    # 2) HuggingFaceHub (hosted) — requires HUGGINGFACEHUB_API_TOKEN and an HF wrapper available
    hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if hf_token and HuggingFaceHub is not None:
        return lambda: HuggingFaceHub(repo_id="google/bison", huggingfacehub_api_token=hf_token)  # example

    # 3) Nothing available -> helpful error
    raise RuntimeError(
        "No LLM available: set DEEPSEEK_API_KEY or OPENAI_API_KEY in st.secrets (or install and configure a HuggingFace provider)."
    )

LLM_FACTORY = _detect_factory()
