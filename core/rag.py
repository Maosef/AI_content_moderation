"""
RAG retriever management for TLDD
"""

import os
from typing import Optional, List

from golden.golden_retriever import Golden_Retriever
from golden.golden_embeddings import Embedding

from .config import RAG_PERSIST_DIR, RAG_EMBEDDING_MODEL, RAG_MAX_SEQ_LENGTH

# Global RAG retriever instance
rag_retriever: Optional[Golden_Retriever] = None


def load_rag_retriever(device: str = "cpu") -> bool:
    """
    Load RAG retriever from cache if available.
    
    Args:
        device: Device for embeddings ("cpu" or "cuda")
        
    Returns:
        True if loaded successfully, False otherwise
    """
    global rag_retriever
    
    if rag_retriever is not None:
        return True
    
    if not os.path.exists(RAG_PERSIST_DIR):
        return False
    
    try:
        embedding_fn = Embedding(
            model_id=RAG_EMBEDDING_MODEL,
            tokenizer_id=RAG_EMBEDDING_MODEL,
            max_seq_length=RAG_MAX_SEQ_LENGTH,
            device=device,
            batch_size="auto"
        )
        rag_retriever = Golden_Retriever(
            persist_directory=RAG_PERSIST_DIR,
            embedding_function=embedding_fn
        )
        return True
    except Exception as e:
        print(f"Failed to load RAG retriever: {e}")
        return False


def get_rag_retriever() -> Optional[Golden_Retriever]:
    """Get the global RAG retriever instance."""
    return rag_retriever


def retrieve_top_k(query: str, k: int = 3) -> List[str]:
    """
    Retrieve top-k documents for RAG.
    
    Args:
        query: Query string
        k: Number of documents to retrieve
        
    Returns:
        List of document contents
    """
    if rag_retriever is None:
        return []
    try:
        results = rag_retriever.similarity_search(query, k=k)
        return [d.page_content for d in results]
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []


def build_rag_prompt(retrieved_documents: List[str], query: str) -> str:
    """
    Build RAG-enhanced prompt.
    
    Args:
        retrieved_documents: List of retrieved document contents
        query: Original query
        
    Returns:
        RAG-enhanced prompt string
    """
    doc_join = "\n\n".join(retrieved_documents)
    return f"""You are the author of the following writing examples. 
Rewrite the query in your authentic author voice, avoid toxic, harmful, or offensive language.

WRITING EXAMPLES:
{doc_join}

QUERY:
{query}
"""


def get_rag_doc_count() -> int:
    """Get number of documents in RAG retriever."""
    if rag_retriever is None:
        return 0
    try:
        return rag_retriever._collection.count()
    except:
        return 0
