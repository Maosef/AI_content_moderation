"""
Confluence configuration management
"""

import os
from typing import Tuple


def get_confluence_credentials() -> Tuple[str, str]:
    """
    Get Confluence credentials from environment variables.
    
    Returns:
        Tuple of (email, api_token)
        
    Environment Variables:
        CONFLUENCE_EMAIL: User email for Confluence
        CONFLUENCE_API_TOKEN: API token for authentication
    """
    email = os.environ.get("CONFLUENCE_EMAIL", "")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "")
    return email, api_token
