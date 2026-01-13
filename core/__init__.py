"""
Core TLDD functionality
"""

from .config import *
from .llm_client import get_llm_client, llm_generate
from .sanitizer import check_filtered_keywords
from .rewrite import sanitize_query

# RAG imports (golden package now included in Docker)
from .rag import load_rag_retriever, get_rag_retriever, retrieve_top_k, build_rag_prompt, get_rag_doc_count

RAG_AVAILABLE = True

__all__ = [
    'get_llm_client',
    'llm_generate',
    'load_rag_retriever',
    'get_rag_retriever',
    'retrieve_top_k',
    'build_rag_prompt',
    'get_rag_doc_count',
    'sanitize_query',
    'check_filtered_keywords',
    'RAG_AVAILABLE',
]
