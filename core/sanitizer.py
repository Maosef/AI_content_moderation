"""
Query sanitization logic for TLDD
Now with prompt injection detection support
"""

from typing import List, Optional, Tuple

from .llm_client import get_llm_client, llm_generate
from .rag import retrieve_top_k, build_rag_prompt
from .prompt_injection import detect_prompt_injection
from .config import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SANITIZER_TEMPERATURE,
    DEFAULT_SANITIZER_MAX_TOKENS,
    DEFAULT_PROMPT_INJECTION_MODEL,
    DEFAULT_PROMPT_INJECTION_THRESHOLD,
    PROMPT_INJECTION_BLOCK_MODE,
    KEYWORDS
)


def sanitize_query(
    query: str,
    use_rag: bool,
    sanitizer_backend: str,
    sanitizer_model: str,
    temperature: float = DEFAULT_SANITIZER_TEMPERATURE,
    system_prompt: str = "",
    use_prompt_injection_detection: bool = True,
    prompt_injection_model: str = DEFAULT_PROMPT_INJECTION_MODEL,
    prompt_injection_threshold: float = DEFAULT_PROMPT_INJECTION_THRESHOLD,
    block_mode: str = PROMPT_INJECTION_BLOCK_MODE,
    verbose: bool = False
) -> str:
    """
    Sanitize user query using TLDD with prompt injection detection.

    Args:
        query: Query to sanitize
        use_rag: Whether to use vanilla RAG enhancement
        sanitizer_backend: "OpenAI" or "Ollama"
        sanitizer_model: Model name for sanitizer
        temperature: Generation temperature
        system_prompt: Custom system prompt (optional, defaults to DEFAULT_SYSTEM_PROMPT)
        use_prompt_injection_detection: Whether to detect prompt injections (recommended)
        prompt_injection_model: Model for detection ("llama-guard-2-86m", "llama-guard-2-22m", "deberta-v3-v2")
        prompt_injection_threshold: Detection threshold (0-1, higher=stricter)
        block_mode: How to handle malicious queries ("block" or "sanitize")
        verbose: Print detection details

    Returns:
        Sanitized query string (or blocked message if malicious and block_mode="block")
    """
    # Step 1: Prompt Injection Detection (if enabled)
    if use_prompt_injection_detection:
        is_safe, detection_details = detect_prompt_injection(
            query,
            model_name=prompt_injection_model,
            threshold=prompt_injection_threshold,
            verbose=verbose
        )

        if not is_safe:
            # Malicious query detected
            label = detection_details.get("label", "MALICIOUS")
            confidence = detection_details.get("confidence", 0.0)

            if verbose or True:  # Always log malicious detections
                print(f"⚠️  Prompt injection detected: {label} (confidence: {confidence:.4f})")

            if block_mode == "block":
                # Block the query entirely
                return f"[BLOCKED: {label} detected]"
            # else: continue to sanitization (block_mode == "sanitize")

    # Step 2: Get sanitizer client
    sanitizer_client, model = get_llm_client(sanitizer_backend, sanitizer_model)

    # Step 3: RAG enhancement (if enabled)
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

    # Step 4: Default sanitization (with or without RAG)
    prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    sanitized = llm_generate(
        sanitizer_client, model, prompt, query,
        temperature=temperature, max_tokens=DEFAULT_SANITIZER_MAX_TOKENS
    )
    return sanitized if sanitized else query


def check_prompt_injection(
    query: str,
    model: str = DEFAULT_PROMPT_INJECTION_MODEL,
    threshold: float = DEFAULT_PROMPT_INJECTION_THRESHOLD,
    verbose: bool = False
) -> Tuple[bool, dict]:
    """
    Check if a query contains prompt injection without sanitizing it.

    Args:
        query: Query to check
        model: Detection model to use
        threshold: Detection threshold (0-1)
        verbose: Print detection details

    Returns:
        Tuple of (is_safe, details_dict)
    """
    return detect_prompt_injection(query, model_name=model, threshold=threshold, verbose=verbose)


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
