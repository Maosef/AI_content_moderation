"""
Prompt injection detection for query sanitization
Supports multiple detection models: Llama Prompt Guard 2, DeBERTa-v3
"""

from typing import Tuple, Optional
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers not available. Prompt injection detection disabled.")


class PromptInjectionDetector:
    """Detects prompt injection and jailbreak attempts in queries"""

    MODELS = {
        "llama-guard-2-86m": "meta-llama/Llama-Prompt-Guard-2-86M",
        "llama-guard-2-22m": "meta-llama/Llama-Prompt-Guard-2-22M",
        "deberta-v3-v2": "protectai/deberta-v3-base-prompt-injection-v2",
    }

    def __init__(
        self,
        model_name: str = "llama-guard-2-86m",
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize the prompt injection detector

        Args:
            model_name: Model to use. Options:
                - "llama-guard-2-86m" (default, best overall, multilingual)
                - "llama-guard-2-22m" (faster, English only)
                - "deberta-v3-v2" (ProtectAI's model, injections only)
            threshold: Detection threshold (0-1). Higher = more strict
            device: Device to use ("cpu", "cuda", or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not available. Install with: pip install transformers torch")

        self.model_name = model_name
        self.threshold = threshold

        # Get full model path
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.MODELS.keys())}")

        model_path = self.MODELS[model_name]

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading prompt injection detector: {model_name} on {self.device}...")

        # Get Hugging Face token from environment (for gated models)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        # Load model and tokenizer (with token if available)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, token=hf_token)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Prompt injection detector loaded: {model_name}")
        except Exception as e:
            if "gated" in str(e).lower() or "restricted" in str(e).lower():
                print(f"⚠️  Model {model_name} requires Hugging Face authentication.")
                print(f"    Set HF_TOKEN environment variable with your HF token.")
                print(f"    Or use an open model like 'deberta-v3-v2'")
            raise

    def detect(self, query: str) -> Tuple[bool, float, str]:
        """
        Detect if a query contains prompt injection or jailbreak attempts

        Args:
            query: The input query to check

        Returns:
            Tuple of (is_malicious, confidence_score, label)
            - is_malicious: True if injection/jailbreak detected
            - confidence_score: Confidence score (0-1)
            - label: Classification label ("INJECTION", "JAILBREAK", or "BENIGN")
        """
        if not TRANSFORMERS_AVAILABLE:
            return False, 0.0, "BENIGN"

        # Tokenize (max 512 tokens for all models)
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Handle different model output formats
        if self.model_name.startswith("llama-guard"):
            # Llama Prompt Guard 2: 3 classes (BENIGN, INJECTION, JAILBREAK)
            probabilities = torch.softmax(logits, dim=1)[0]

            benign_score = probabilities[0].item()
            injection_score = probabilities[1].item() if len(probabilities) > 1 else 0.0
            jailbreak_score = probabilities[2].item() if len(probabilities) > 2 else 0.0

            # Get the max malicious score
            max_malicious_score = max(injection_score, jailbreak_score)
            is_malicious = max_malicious_score > self.threshold

            if jailbreak_score > injection_score and jailbreak_score > self.threshold:
                label = "JAILBREAK"
                confidence = jailbreak_score
            elif injection_score > self.threshold:
                label = "INJECTION"
                confidence = injection_score
            else:
                label = "BENIGN"
                confidence = benign_score

        else:
            # DeBERTa: 2 classes (BENIGN, INJECTION)
            probabilities = torch.softmax(logits, dim=1)[0]
            benign_score = probabilities[0].item()
            injection_score = probabilities[1].item()

            is_malicious = injection_score > self.threshold
            label = "INJECTION" if is_malicious else "BENIGN"
            confidence = injection_score if is_malicious else benign_score

        return is_malicious, confidence, label

    def scan_query(self, query: str, verbose: bool = False) -> Tuple[bool, dict]:
        """
        Scan a query and return detailed results

        Args:
            query: Query to scan
            verbose: Print detection details

        Returns:
            Tuple of (is_safe, details_dict)
            - is_safe: True if query is safe, False if malicious
            - details_dict: Dictionary with detection details
        """
        is_malicious, confidence, label = self.detect(query)

        details = {
            "is_safe": not is_malicious,
            "is_malicious": is_malicious,
            "confidence": confidence,
            "label": label,
            "threshold": self.threshold,
            "model": self.model_name
        }

        if verbose:
            status = "❌ BLOCKED" if is_malicious else "✅ SAFE"
            print(f"{status} - {label} (confidence: {confidence:.4f}, threshold: {self.threshold})")

        return not is_malicious, details


# Global detector instance (lazy loading)
_detector_instance: Optional[PromptInjectionDetector] = None


def get_prompt_injection_detector(
    model_name: str = "llama-guard-2-86m",
    threshold: float = 0.5
) -> Optional[PromptInjectionDetector]:
    """
    Get or create the global prompt injection detector instance

    Args:
        model_name: Model to use
        threshold: Detection threshold

    Returns:
        PromptInjectionDetector instance or None if not available
    """
    global _detector_instance

    if not TRANSFORMERS_AVAILABLE:
        return None

    # Create new instance if needed or if settings changed
    if _detector_instance is None or \
       _detector_instance.model_name != model_name or \
       _detector_instance.threshold != threshold:
        try:
            _detector_instance = PromptInjectionDetector(model_name, threshold)
        except Exception as e:
            print(f"⚠️  Failed to load prompt injection detector: {e}")
            return None

    return _detector_instance


def detect_prompt_injection(
    query: str,
    model_name: str = "llama-guard-2-86m",
    threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[bool, dict]:
    """
    Convenience function to detect prompt injection in a query

    Args:
        query: Query to check
        model_name: Model to use
        threshold: Detection threshold
        verbose: Print detection details

    Returns:
        Tuple of (is_safe, details_dict)
    """
    detector = get_prompt_injection_detector(model_name, threshold)

    if detector is None:
        # If detector not available, assume safe (fail open)
        return True, {
            "is_safe": True,
            "is_malicious": False,
            "confidence": 0.0,
            "label": "BENIGN",
            "threshold": threshold,
            "model": "none",
            "error": "Detector not available"
        }

    return detector.scan_query(query, verbose=verbose)
