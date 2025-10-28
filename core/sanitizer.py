"""
Query sanitization logic for TLDD
"""

from typing import List

from .llm_client import get_llm_client, llm_generate
from .rag import retrieve_top_k, build_rag_prompt
from .config import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SANITIZER_TEMPERATURE,
    DEFAULT_SANITIZER_MAX_TOKENS,
    KEYWORDS
)


def sanitize_query(
    query: str,
    use_rag: bool,
    sanitizer_backend: str,
    sanitizer_model: str,
    temperature: float = DEFAULT_SANITIZER_TEMPERATURE
) -> str:
    """
    Sanitize user query using TLDD.
    
    Args:
        query: Query to sanitize
        use_rag: Whether to use RAG enhancement
        sanitizer_backend: "OpenAI" or "Ollama"
        sanitizer_model: Model name for sanitizer
        temperature: Generation temperature
        
    Returns:
        Sanitized query string
    """
    # Get sanitizer client
    sanitizer_client, model = get_llm_client(sanitizer_backend, sanitizer_model)
    
    if use_rag:
        top_k_docs = retrieve_top_k(query, k=3)
        if top_k_docs:
            rag_prompt = build_rag_prompt(top_k_docs, query)
            sanitized = llm_generate(
                sanitizer_client, model, "", rag_prompt,
                temperature=temperature, max_tokens=DEFAULT_SANITIZER_MAX_TOKENS
            )
            if sanitized:
                return sanitized
    
    sanitized = llm_generate(
        sanitizer_client, model, DEFAULT_SYSTEM_PROMPT, query,
        temperature=temperature, max_tokens=DEFAULT_SANITIZER_MAX_TOKENS
    )
    return sanitized if sanitized else query


def check_filtered_keywords(text: str) -> List[str]:
    """
    Check for filtered keywords in response.
    
    Args:
        text: Text to check
        
    Returns:
        List of found keywords
    """
    found = []
    text_lower = text.lower()
    for keyword in KEYWORDS:
        if keyword.lower() in text_lower:
            found.append(keyword)
    return found
