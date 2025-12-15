"""
Confluence content rewriter functions
Analyzes harmful content and rewrites it to be safe
"""

import sys
import os
import json
from typing import Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import sanitize_query, check_filtered_keywords
from .moderator import moderate_content
from .client import fetch_test_comments


def rewrite_harmful_content(
    content: str,
    backend: str,
    ollama_model: str,
    temperature: float,
    max_tokens: int,
    use_rag: bool = False,
    system_prompt: str = ""
) -> Tuple[str, Dict]:
    """
    Rewrite harmful content to be safe.
    
    Uses the sanitize_query logic from core.sanitizer
    
    Args:
        content: Original content to rewrite
        backend: LLM backend ("OpenAI" or "Ollama")
        ollama_model: Model name for Ollama
        temperature: Temperature for generation
        max_tokens: Max tokens for generation
        use_rag: Use RAG retriever
        system_prompt: Custom system prompt for rewriting
        
    Returns:
        Tuple of (rewritten_content, metadata_dict)
        metadata_dict contains:
            - original_length
            - rewritten_length
            - changes_made
            - filtered_keywords
            - reduction_percentage
    """
    try:
        # Use existing sanitization logic
        rewritten = sanitize_query(
            query=content,
            use_rag=use_rag,
            sanitizer_backend=backend,
            sanitizer_model=ollama_model,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        # Check what was filtered
        filtered_keywords = check_filtered_keywords(content)
        
        metadata = {
            'original_length': len(content),
            'rewritten_length': len(rewritten),
            'changes_made': content != rewritten,
            'filtered_keywords': filtered_keywords,
            'reduction_percentage': 
                round(100 * (1 - len(rewritten) / len(content)), 2)
                if len(content) > 0 else 0
        }
        
        return rewritten, metadata
        
    except Exception as e:
        return content, {
            'error': str(e),
            'original_length': len(content),
            'rewritten_length': len(content),
            'changes_made': False,
            'filtered_keywords': [],
            'reduction_percentage': 0
        }


def process_and_rewrite_comments(
    backend: str,
    ollama_model: str,
    mod_temperature: float,
    rewrite_temperature: float,
    max_tokens: int,
    use_rag: bool = False,
    progress_callback: Optional[object] = None
) -> str:
    """
    Analyze test content and rewrite harmful ones.
    
    Workflow:
    1. Load test content
    2. For each content:
       a. Analyze with moderate_content()
       b. If harmful, rewrite with rewrite_harmful_content()
       c. Show before/after comparison
    
    Args:
        backend: LLM backend
        ollama_model: Ollama model name
        mod_temperature: Temperature for moderation
        rewrite_temperature: Temperature for rewriting
        max_tokens: Max tokens
        use_rag: Use RAG for rewriting
        progress_callback: Optional callback for progress updates (progress, desc)
        
    Returns:
        Formatted markdown string with analysis and rewrites
    """
    def progress(value, desc=""):
        if progress_callback:
            progress_callback(value, desc=desc)
    
    progress(0, desc="Loading test content...")
    comments = fetch_test_comments()
    
    if not comments:
        return "❌ **Error:** Failed to load test content\n\nCheck that ui/test_comments.json exists."
    
    # First pass: analyze all content to count harmful ones
    progress(0.1, desc="Analyzing content...")
    harmful_count = 0
    analyses = []
    
    for idx, comment in enumerate(comments, 1):
        progress(0.1 + (0.3 * idx / len(comments)), 
                desc=f"Analyzing content {idx}/{len(comments)}...")
        
        comment_id = comment.get('id', 'Unknown')
        body = comment.get('body', '')
        
        raw_response, parsed = moderate_content(
            body,
            backend,
            ollama_model,
            mod_temperature,
            max_tokens
        )
        
        is_harmful = parsed.get('is_harmful', False) if parsed else False
        if is_harmful:
            harmful_count += 1
        
        analyses.append({
            'comment': comment,
            'raw_response': raw_response,
            'parsed': parsed,
            'is_harmful': is_harmful
        })
    
    # Build result header
    result = f"# 🔄 Test Content Rewriting Results\n\n"
    result += f"**Total Content:** {len(comments)}\n\n"
    result += f"**Harmful Content Found:** {harmful_count}\n\n"
    result += f"**Content Rewritten:** {harmful_count}\n\n"
    result += "---\n\n"
    
    # Second pass: rewrite harmful content and format output
    rewritten_count = 0
    for idx, analysis in enumerate(analyses, 1):
        comment = analysis['comment']
        parsed = analysis['parsed']
        is_harmful = analysis['is_harmful']
        raw_response = analysis['raw_response']
        
        comment_id = comment.get('id', 'Unknown')
        body = comment.get('body', '')
        
        result += f"## Content {idx} (ID: {comment_id})\n\n"
        
        # Original content
        result += f"### 📝 Original Content\n```\n{body}\n```\n\n"
        
        # Moderation analysis
        result += f"### 🛡️ Moderation Analysis\n"
        
        if parsed:
            severity = parsed.get('severity', 'unknown')
            issues = parsed.get('issues', [])
            recommendations = parsed.get('recommendations', [])
            
            if is_harmful:
                result += f"⚠️ **Status:** HARMFUL (`{severity.upper()}`)\n\n"
                if issues:
                    result += "**Issues:**\n"
                    for issue in issues:
                        result += f"- {issue}\n"
                    result += "\n"
            else:
                result += f"✅ **Status:** SAFE\n\n"
        else:
            result += f"⚠️ **Status:** Unable to parse analysis\n\n"
        
        # Rewrite if harmful
        if is_harmful:
            rewritten_count += 1
            progress(0.4 + (0.5 * rewritten_count / harmful_count),
                    desc=f"Rewriting harmful content {rewritten_count}/{harmful_count}...")
            
            rewritten, metadata = rewrite_harmful_content(
                body,
                backend,
                ollama_model,
                rewrite_temperature,
                max_tokens,
                use_rag
            )
            
            result += f"### ✏️ Rewritten Content\n```\n{rewritten}\n```\n\n"
        else:
            result += f"✅ **No rewriting needed** (content is safe)\n\n"
        
        result += "---\n\n"
    
    progress(1.0, desc="Complete!")
    return result


def rewrite_all_comments(
    backend: str,
    ollama_model: str,
    rewrite_temperature: float,
    max_tokens: int,
    use_rag: bool = False,
    system_prompt: str = "",
    progress_callback: Optional[object] = None
) -> str:
    """
    Rewrite ALL test content without moderation analysis.
    
    Workflow:
    1. Load test content
    2. For each content:
       a. Rewrite with rewrite_harmful_content()
       b. Show before/after comparison with system prompt
    
    Args:
        backend: LLM backend
        ollama_model: Ollama model name
        rewrite_temperature: Temperature for rewriting
        max_tokens: Max tokens
        use_rag: Use RAG for rewriting
        system_prompt: Custom system prompt for rewriting
        progress_callback: Optional callback for progress updates (progress, desc)
        
    Returns:
        Formatted markdown string with rewrites
    """
    def progress(value, desc=""):
        if progress_callback:
            progress_callback(value, desc=desc)
    
    progress(0, desc="Loading test content...")
    comments = fetch_test_comments()
    
    if not comments:
        return "❌ **Error:** Failed to load test content\n\nCheck that ui/test_comments.json exists."
    
    # Build result header with system prompt
    result = f"# ✏️ All Content Rewriting Results\n\n"
    result += f"**Total Content:** {len(comments)}\n\n"
    result += f"**Content Rewritten:** {len(comments)}\n\n"
    result += "---\n\n"
    
    # Rewrite all content
    for idx, comment in enumerate(comments, 1):
        progress(idx / len(comments), 
                desc=f"Rewriting content {idx}/{len(comments)}...")
        
        comment_id = comment.get('id', 'Unknown')
        body = comment.get('body', '')
        
        result += f"## Content {idx} (ID: {comment_id})\n\n"
        
        # Original content
        result += f"### 📝 Original Content\n```\n{body}\n```\n\n"
        
        # Rewrite
        rewritten, metadata = rewrite_harmful_content(
            body,
            backend,
            ollama_model,
            rewrite_temperature,
            max_tokens,
            use_rag,
            system_prompt
        )
        
        result += f"### ✏️ Rewritten Content\n```\n{rewritten}\n```\n\n"
        
        if metadata.get('error'):
            result += f"⚠️ **Error:** {metadata['error']}\n\n"
        
        result += "---\n\n"
    
    progress(1.0, desc="Complete!")
    return result
