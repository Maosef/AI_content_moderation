"""
Confluence integration module
"""

from .client import parse_confluence_url, fetch_confluence_page, fetch_test_comments, strip_html_tags
from .config import get_confluence_credentials
from .moderator import moderate_content, process_confluence_page, process_test_comments, MODERATION_SYSTEM_PROMPT
from .rewriter import rewrite_harmful_content, process_and_rewrite_comments, rewrite_all_comments

__all__ = [
    'parse_confluence_url',
    'fetch_confluence_page',
    'fetch_test_comments',
    'strip_html_tags',
    'get_confluence_credentials',
    'moderate_content',
    'process_confluence_page',
    'process_test_comments',
    'MODERATION_SYSTEM_PROMPT',
    'rewrite_harmful_content',
    'process_and_rewrite_comments',
    'rewrite_all_comments',
]
