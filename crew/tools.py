# crew/tools.py
from __future__ import annotations
from typing import List

from crewai.tools import tool
from langchain_core.documents import Document
from rag.retriever import get_retriever

# Global retriever hooked to the Canada AI vector store
_retriever = get_retriever(k=5)


def _retrieve_docs(query: str) -> List[Document]:
    """Return top-k retrieved documents from the Canada AI vector store."""
    return _retriever.invoke(query)


@tool("retrieve_context")
def retrieve_context(description: str) -> str:
    """
    Given the user's question, return one concatenated string of the top-k
    retrieved chunks from the Canada AI background corpus (PDF methodology + CSV scores).
    """
    query = description
    docs = _retrieve_docs(query)
    return "\n\n".join(
        (d.page_content or "").strip()
        for d in docs
        if d.page_content
    )
