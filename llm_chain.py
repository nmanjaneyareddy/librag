# llm_chain.py
import streamlit as st
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


def _make_llm():
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or DEEPSEEK_API_KEY in Streamlit secrets")

    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=512,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


class SimpleRetrievalQA:
    """
    Minimal, stable Retrieval-QA implementation.
    No langchain.chains usage.
    """

    def __init__(self, retriever, llm, k: int = 4):
        self.retriever = retriever
        self.llm = llm
        self.k = k

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question clearly and concisely.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs.get("input") or inputs.get("question")
        if not question:
            return {"answer": ""}

        docs = self.retriever.get_relevant_documents(question)[: self.k]

        context_parts: List[str] = []
        for d in docs:
            text = getattr(d, "page_content", None)
            if text:
                context_parts.append(text)

        context = "\n\n".join(context_parts)
        prompt_text = self.prompt.format(context=context, question=question)

        if hasattr(self.llm, "invoke"):
            answer = self.llm.invoke(prompt_text).content
        else:
            answer = self.llm.predict(prompt_text)

        return {"answer": answer}


def setup_qa_chain(vectorstore, k: int = 4):
    if vectorstore is None:
        raise ValueError("vectorstore cannot be None")

    llm = _make_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return SimpleRetrievalQA(retriever, llm, k=k)
