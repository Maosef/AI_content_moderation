"""
Query sanitization logic for TLDD
Now with prompt injection detection support and LLM Guard integration
"""

from typing import List, Optional, Tuple, Dict, Any

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

# LLM Guard imports
try:
    from llm_guard import scan_prompt
    from llm_guard.input_scanners import (
        PromptInjection as LLMGuardPromptInjection,
        Toxicity,
        BanTopics,
    )
    LLM_GUARD_AVAILABLE = True
except ImportError:
    LLM_GUARD_AVAILABLE = False


def scan_with_llm_guard(
    query: str,
    enable_prompt_injection: bool = True,
    enable_toxicity: bool = True,
    enable_banned_topics: bool = False,
    toxicity_threshold: float = 0.5,
    banned_topics: Optional[List[str]] = None,
    banned_topics_threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Scan query using LLM Guard input scanners.

    Args:
        query: Query to scan
        enable_prompt_injection: Enable prompt injection detection
        enable_toxicity: Enable toxicity detection
        enable_banned_topics: Enable banned topics detection
        toxicity_threshold: Threshold for toxicity (0-1, higher=stricter)
        banned_topics: List of topics to ban (e.g., ["violence", "illegal activities"])
        banned_topics_threshold: Threshold for banned topics detection
        verbose: Print scan details

    Returns:
        Tuple of (sanitized_query, is_valid, scan_results)
    """
    if not LLM_GUARD_AVAILABLE:
        if verbose:
            print("⚠️  LLM Guard not available, skipping scan")
        return query, True, {"error": "LLM Guard not installed"}

    # Build scanner list
    input_scanners = []

    if enable_prompt_injection:
        input_scanners.append(LLMGuardPromptInjection())

    if enable_toxicity:
        input_scanners.append(Toxicity(threshold=toxicity_threshold))

    if enable_banned_topics and banned_topics:
        input_scanners.append(BanTopics(topics=banned_topics, threshold=banned_topics_threshold))

    if not input_scanners:
        # No scanners enabled
        return query, True, {}

    try:
        # Scan the prompt
        sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, query)

        if verbose:
            print(f"\n{'='*60}")
            print("LLM Guard Scan Results")
            print(f"{'='*60}")
            print(f"Original query: {query}")
            print(f"Sanitized query: {sanitized_prompt}")
            print(f"Valid: {results_valid}")
            print(f"Scores: {results_score}")
            print(f"{'='*60}\n")

        return sanitized_prompt, results_valid, results_score

    except Exception as e:
        if verbose:
            print(f"⚠️  LLM Guard scan error: {e}")
        return query, True, {"error": str(e)}


def check_prompt_with_llm_guard(
    query: str,
    enable_prompt_injection: bool = True,
    enable_toxicity: bool = True,
    enable_banned_topics: bool = False,
    toxicity_threshold: float = 0.5,
    banned_topics: Optional[List[str]] = None,
    banned_topics_threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check query using LLM Guard without sanitizing.

    Args:
        query: Query to check
        enable_prompt_injection: Enable prompt injection detection
        enable_toxicity: Enable toxicity detection
        enable_banned_topics: Enable banned topics detection
        toxicity_threshold: Threshold for toxicity (0-1)
        banned_topics: List of topics to ban
        banned_topics_threshold: Threshold for banned topics
        verbose: Print details

    Returns:
        Tuple of (is_safe, scan_details)
    """
    _, is_valid, scan_results = scan_with_llm_guard(
        query=query,
        enable_prompt_injection=enable_prompt_injection,
        enable_toxicity=enable_toxicity,
        enable_banned_topics=enable_banned_topics,
        toxicity_threshold=toxicity_threshold,
        banned_topics=banned_topics,
        banned_topics_threshold=banned_topics_threshold,
        verbose=verbose
    )

    return is_valid, scan_results


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
