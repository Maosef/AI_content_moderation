"""
LLM client management for TLDD
"""

import os
import time
from typing import Optional, Tuple, Union
from openai import OpenAI

from .config import DEFAULT_OPENAI_MODEL, DEFAULT_HUGGINGFACE_MODEL, NUM_RETRIES

# HuggingFace imports (optional, only needed for local models)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


class HuggingFaceModel:
    """Wrapper for HuggingFace local models"""

    def __init__(self, model_name: str, use_4bit: bool = True):
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers not available. "
                "Install with: pip install transformers torch accelerate bitsandbytes"
            )

        self.model_name = model_name
        self.use_4bit = use_4bit

        # Configure 4-bit quantization if requested
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None

        print(f"Loading HuggingFace model: {model_name} (4-bit: {use_4bit})...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Model loaded successfully!")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Optional[str]:
        """Generate response using HuggingFace model"""

        # Format prompt for Llama-3 chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract only the generated text
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (remove the prompt)
        response = full_response[len(prompt):].strip()

        return response if response else None


# Global cache for HuggingFace model (avoid reloading)
_hf_model_cache = {}


def get_llm_client(backend: str, model_name: str) -> Tuple[Union[OpenAI, HuggingFaceModel], str]:
    """
    Get LLM client based on backend selection.

    Args:
        backend: "OpenAI", "Ollama", or "HuggingFace"
        model_name: Model name for the backend

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
    elif backend == "HuggingFace":
        # Use cached model if available
        if model_name not in _hf_model_cache:
            use_4bit = os.environ.get("HF_USE_4BIT", "true").lower() == "true"
            _hf_model_cache[model_name] = HuggingFaceModel(model_name, use_4bit=use_4bit)
        return _hf_model_cache[model_name], model_name
    else:  # Ollama
        return OpenAI(base_url='http://localhost:11434/v1', api_key='ollama'), model_name


def llm_generate(
    client: Union[OpenAI, HuggingFaceModel],
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
        client: OpenAI or HuggingFaceModel instance
        model: Model name (used for OpenAI/Ollama, ignored for HuggingFace)
        system_prompt: System prompt (can be empty string)
        user_prompt: User prompt
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        retries: Number of retry attempts

    Returns:
        Generated text or None if all retries failed
    """
    # Handle HuggingFace models differently
    if isinstance(client, HuggingFaceModel):
        try:
            return client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"HuggingFace generation error: {e}")
            return None

    # OpenAI/Ollama clients
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
