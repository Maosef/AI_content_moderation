"""
Confluence API client
"""

import re
import json
from typing import Optional, Dict, Tuple, List
import requests
from requests.auth import HTTPBasicAuth


def parse_confluence_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse Confluence URL to extract domain, space key, and page ID.
    
    Args:
        url: Confluence page URL
        
    Returns:
        Tuple of (base_url, space_key, page_id) or (None, None, None) if invalid
        
    Example:
        >>> parse_confluence_url("https://example.atlassian.net/wiki/spaces/DEMO/pages/12345")
        ("https://example.atlassian.net/wiki", "DEMO", "12345")
    """
    pattern = r'https://([^/]+)/wiki/spaces/([^/]+)/pages/(\d+)'
    match = re.match(pattern, url)
    
    if match:
        domain = match.group(1)
        space_key = match.group(2)
        page_id = match.group(3)
        base_url = f"https://{domain}/wiki"
        return base_url, space_key, page_id
    
    return None, None, None


def fetch_confluence_page(
    base_url: str,
    page_id: str,
    email: str,
    api_token: str
) -> Optional[Dict]:
    """
    Fetch Confluence page content via REST API.
    
    Args:
        base_url: Confluence base URL (e.g., "https://example.atlassian.net/wiki")
        page_id: Page ID
        email: Confluence user email
        api_token: Confluence API token
        
    Returns:
        Dictionary with page data including title, body, version, space info
        Returns None if fetch fails
        
    Example:
        >>> page_data = fetch_confluence_page(
        ...     "https://example.atlassian.net/wiki",
        ...     "12345",
        ...     "user@example.com",
        ...     "api_token_here"
        ... )
        >>> print(page_data['title'])
    """
    api_url = f"{base_url}/rest/api/content/{page_id}?expand=body.storage,version,space"
    
    try:
        response = requests.get(
            api_url,
            auth=HTTPBasicAuth(email, api_token),
            headers={'Accept': 'application/json'}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Confluence page: {e}")
        return None


def fetch_confluence_comments(
    base_url: str,
    page_id: str,
    email: str,
    api_token: str
) -> Optional[List[Dict]]:
    """
    Fetch comments for a Confluence page via REST API.
    
    Args:
        base_url: Confluence base URL
        page_id: Page ID
        email: Confluence user email
        api_token: Confluence API token
        
    Returns:
        List of comment dictionaries with id, status, and body fields
        Returns None if fetch fails
    """
    api_url = f"{base_url}/rest/api/content/{page_id}/child/comment?expand=body.storage"
    
    try:
        response = requests.get(
            api_url,
            auth=HTTPBasicAuth(email, api_token),
            headers={'Accept': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant fields
        comments = []
        for comment in data.get('results', []):
            comments.append({
                'id': comment.get('id'),
                'status': comment.get('status'),
                'body': strip_html_tags(comment.get('body', {}).get('storage', {}).get('value', ''))
            })
        
        return comments
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Confluence comments: {e}")
        return None


def fetch_test_comments(test_file_path: Optional[str] = None) -> Optional[List[Dict]]:
    """
    Load test comments from JSON file for testing/demo purposes.
    
    Args:
        test_file_path: Path to test comments JSON file
                       Defaults to ui/test_comments.json
        
    Returns:
        List of comment dictionaries with id, status, and body fields
        Returns None if file not found or invalid
        
    Example:
        >>> comments = fetch_test_comments()
        >>> for comment in comments:
        ...     print(f"Comment {comment['id']}: {comment['body'][:50]}...")
    """
    import os
    
    if test_file_path is None:
        # Default to ui/test_comments.json relative to project root
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file_path = os.path.join(current_dir, 'ui', 'test_comments.json')
    
    try:
        with open(test_file_path, 'r') as f:
            data = json.load(f)
            return data.get('results', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading test comments: {e}")
        return None


def strip_html_tags(html: str) -> str:
    """
    Remove HTML tags from content.
    
    Args:
        html: HTML content
        
    Returns:
        Plain text content with HTML tags removed
        
    Example:
        >>> strip_html_tags("<p>Hello <strong>World</strong>!</p>")
        "Hello World!"
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html)
