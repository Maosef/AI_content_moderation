#!/usr/bin/env python3

"""
Example: Using the Confluence module
Demonstrates fetching pages, comments, and test data
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confluence import (
    parse_confluence_url,
    fetch_confluence_page,
    fetch_test_comments,
    strip_html_tags,
    get_confluence_credentials
)


def example_parse_url():
    """Example: Parse Confluence URL"""
    print("=" * 60)
    print("Example 1: Parse Confluence URL")
    print("=" * 60)
    
    url = "https://example.atlassian.net/wiki/spaces/DEMO/pages/12345"
    base_url, space_key, page_id = parse_confluence_url(url)
    
    if base_url:
        print(f"URL: {url}")
        print(f"Base URL: {base_url}")
        print(f"Space Key: {space_key}")
        print(f"Page ID: {page_id}")
    else:
        print("Invalid URL format")
    print()


def example_fetch_page():
    """Example: Fetch Confluence page"""
    print("=" * 60)
    print("Example 2: Fetch Confluence Page")
    print("=" * 60)
    
    email, api_token = get_confluence_credentials()
    
    if not email or not api_token:
        print("⚠️  Credentials not set in environment variables")
        print("Set CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN to run this example")
        print()
        return
    
    # Example URL - replace with your actual page
    url = "https://yoursite.atlassian.net/wiki/spaces/SPACE/pages/12345"
    base_url, space_key, page_id = parse_confluence_url(url)
    
    if not base_url:
        print("Invalid URL")
        return
    
    print(f"Fetching page {page_id}...")
    page_data = fetch_confluence_page(base_url, page_id, email, api_token)
    
    if page_data:
        title = page_data.get('title', 'Unknown')
        html_content = page_data.get('body', {}).get('storage', {}).get('value', '')
        plain_text = strip_html_tags(html_content)
        
        print(f"✅ Successfully fetched page")
        print(f"Title: {title}")
        print(f"Content length: {len(plain_text)} characters")
        print(f"Preview: {plain_text[:200]}...")
    else:
        print("❌ Failed to fetch page")
    print()


def example_test_comments():
    """Example: Load test comments"""
    print("=" * 60)
    print("Example 3: Load Test Comments")
    print("=" * 60)
    
    comments = fetch_test_comments()
    
    if comments:
        print(f"✅ Loaded {len(comments)} test comments")
        print()
        
        for idx, comment in enumerate(comments[:3], 1):  # Show first 3
            comment_id = comment.get('id', 'Unknown')
            body = comment.get('body', '')
            
            print(f"Comment {idx} (ID: {comment_id})")
            print(f"Body: {body[:100]}...")
            print()
    else:
        print("❌ Failed to load test comments")
        print("Ensure ui/test_comments.json exists")
    print()


def example_strip_html():
    """Example: Strip HTML tags"""
    print("=" * 60)
    print("Example 4: Strip HTML Tags")
    print("=" * 60)
    
    html = """
    <p>This is a <strong>test</strong> page with <a href="#">links</a>.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    """
    
    plain_text = strip_html_tags(html)
    
    print("Original HTML:")
    print(html)
    print("\nPlain text:")
    print(plain_text.strip())
    print()


if __name__ == "__main__":
    print("\n🚀 Confluence Module Examples\n")
    
    example_parse_url()
    example_strip_html()
    example_test_comments()
    example_fetch_page()
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)
