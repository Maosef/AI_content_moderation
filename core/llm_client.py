"""
LLM client management for TLDD
"""

import os
import time
from typing import Optional, Tuple
from openai import OpenAI

from .config import DEFAULT_OPENAI_MODEL, NUM_RETRIES


def get_llm_client(backend: str, ollama_model: str) -> Tuple[OpenAI, str]:
    """
    Get LLM client based on backend selection.
    
    Args:
        backend: "OpenAI" or "Ollama"
        ollama_model: Model name for Ollama backend
        
    Returns:
        Tuple of (client, model_name)
        
    Raises:
        ValueError: If OpenAI API key not set when using OpenAI backend
    """
    if backend == "OpenAI":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key), os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    else:
        return OpenAI(base_url='http://localhost:11434/v1', api_key='ollama'), ollama_model


def llm_generate(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    retries: int = NUM_RETRIES
) -> Optional[str]:
    """
    Generate LLM response with retry logic.
    
    Args:
        client: OpenAI client instance
        model: Model name
        system_prompt: System prompt (can be empty string)
        user_prompt: User prompt
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        retries: Number of retry attempts
        
    Returns:
        Generated text or None if all retries failed
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            if getattr(response.choices[0], "finish_reason", None) == "content_filter":
                time.sleep(0.25)
                continue
                
            content = response.choices[0].message.content
            if content:
                return content
                
        except Exception as e:
            print(f"Generation error: {e}")
            
        time.sleep(0.25)
    
    return None
