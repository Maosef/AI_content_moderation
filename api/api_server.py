#!/usr/bin/env python3

"""
FastAPI server for LLM Guard Prompt Scanning
Provides REST API endpoint for scanning prompts with LLM Guard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# LLM Guard imports
try:
    from llm_guard import scan_prompt
    from llm_guard.input_scanners import (
        PromptInjection,
        Toxicity,
        BanTopics,
    )
    LLM_GUARD_AVAILABLE = True
except ImportError:
    LLM_GUARD_AVAILABLE = False
    print("⚠️  LLM Guard not installed. Install with: pip install llm-guard")

app = FastAPI(
    title="LLM Guard Prompt Scanner API",
    description="API for scanning prompts using LLM Guard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to scan")
    enable_prompt_injection: bool = Field(default=True, description="Enable prompt injection detection")
    enable_toxicity: bool = Field(default=False, description="Enable toxicity detection")
    enable_banned_topics: bool = Field(default=False, description="Enable banned topics detection")
    toxicity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Toxicity threshold")
    banned_topics_list: Optional[List[str]] = Field(default=None, description="List of banned topics")
    banned_topics_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Banned topics threshold")

class ScanResponse(BaseModel):
    sanitized_prompt: str = Field(..., description="Sanitized version of the prompt")
    is_valid: bool = Field(..., description="Whether the prompt passed all scanners")
    scan_scores: Dict[str, Any] = Field(..., description="Detailed scanner scores")
    llm_guard_available: bool = Field(..., description="Whether LLM Guard is available")

def scan_prompt_with_llm_guard(
    prompt: str,
    enable_prompt_injection: bool = True,
    enable_toxicity: bool = False,
    enable_banned_topics: bool = False,
    toxicity_threshold: float = 0.5,
    banned_topics: Optional[List[str]] = None,
    banned_topics_threshold: float = 0.5,
) -> tuple[str, bool, Dict[str, Any]]:
    """
    Scan prompt using LLM Guard with focus on prompt injection detection.

    Args:
        prompt: The prompt to scan
        enable_prompt_injection: Enable prompt injection detection
        enable_toxicity: Enable toxicity detection
        enable_banned_topics: Enable banned topics detection
        toxicity_threshold: Toxicity threshold (0-1)
        banned_topics: List of banned topics
        banned_topics_threshold: Banned topics threshold (0-1)

    Returns:
        Tuple of (sanitized_prompt, is_valid, scan_scores)
    """
    if not LLM_GUARD_AVAILABLE:
        return prompt, True, {"error": "LLM Guard not available"}

    # Build scanner list
    scanners = []

    if enable_prompt_injection:
        scanners.append(PromptInjection())

    if enable_toxicity:
        scanners.append(Toxicity(threshold=toxicity_threshold))

    if enable_banned_topics and banned_topics:
        scanners.append(BanTopics(topics=banned_topics, threshold=banned_topics_threshold))

    if not scanners:
        return prompt, True, {}

    try:
        # scan_prompt returns (sanitized_prompt, is_valid, scan_scores)
        result = scan_prompt(scanners, prompt)

        sanitized_prompt = result[0]  # Always the sanitized prompt string

        # Determine which is is_valid (bool) and which is scan_scores (dict)
        if isinstance(result[1], bool):
            is_valid = result[1]
            scan_scores = result[2] if len(result) > 2 else {}
        elif isinstance(result[1], dict):
            scan_scores = result[1]
            is_valid = result[2] if len(result) > 2 and isinstance(result[2], bool) else True
            # If is_valid wasn't a bool, compute it from scores
            if not isinstance(is_valid, bool):
                is_valid = all(
                    score == 0 or score is False
                    for score in scan_scores.values()
                    if isinstance(score, (int, float, bool))
                )
        else:
            # Fallback
            is_valid = True
            scan_scores = {}

        return sanitized_prompt, is_valid, scan_scores
    except Exception as e:
        print(f"⚠️  LLM Guard scan error: {e}")
        import traceback
        traceback.print_exc()
        return prompt, True, {"error": str(e)}

@app.get("/")
async def root():
    return {
        "service": "LLM Guard Prompt Scanner API",
        "version": "1.0.0",
        "status": "running",
        "llm_guard_available": LLM_GUARD_AVAILABLE
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llm_guard_available": LLM_GUARD_AVAILABLE
    }

@app.post("/api/v1/scan", response_model=ScanResponse)
async def scan(request: ScanRequest):
    """
    Scan a prompt using LLM Guard scanners.

    Returns the sanitized prompt, validation status, and detailed scan scores.
    """
    if not LLM_GUARD_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LLM Guard is not available. Please install it with: pip install llm-guard"
        )

    sanitized_prompt, is_valid, scan_scores = scan_prompt_with_llm_guard(
        prompt=request.prompt,
        enable_prompt_injection=request.enable_prompt_injection,
        enable_toxicity=request.enable_toxicity,
        enable_banned_topics=request.enable_banned_topics,
        toxicity_threshold=request.toxicity_threshold,
        banned_topics=request.banned_topics_list,
        banned_topics_threshold=request.banned_topics_threshold,
    )

    return ScanResponse(
        sanitized_prompt=sanitized_prompt,
        is_valid=is_valid,
        scan_scores=scan_scores,
        llm_guard_available=LLM_GUARD_AVAILABLE
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
