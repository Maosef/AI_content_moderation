# Confluence Module

Python module for interacting with Atlassian Confluence via REST API.

## Files

- `__init__.py` - Module exports
- `client.py` - API client functions (fetch pages, comments, parse URLs)
- `config.py` - Configuration management (credentials from env vars)
- `moderator.py` - Content moderation functions (analyze for harmful content)
- `rewriter.py` - Content rewriting functions (sanitize harmful content)

## Usage

### Import

```python
from confluence import (
    parse_confluence_url,
    fetch_confluence_page,
    fetch_test_comments,
    strip_html_tags,
    get_confluence_credentials
)
```

### Parse URL

```python
url = "https://example.atlassian.net/wiki/spaces/DEMO/pages/12345"
base_url, space_key, page_id = parse_confluence_url(url)
# Returns: ("https://example.atlassian.net/wiki", "DEMO", "12345")
```

### Fetch Page

```python
from confluence import fetch_confluence_page, get_confluence_credentials

email, api_token = get_confluence_credentials()
page_data = fetch_confluence_page(
    base_url="https://example.atlassian.net/wiki",
    page_id="12345",
    email=email,
    api_token=api_token
)

if page_data:
    title = page_data['title']
    html_content = page_data['body']['storage']['value']
    plain_text = strip_html_tags(html_content)
```

### Fetch Comments

```python
from confluence.client import fetch_confluence_comments

comments = fetch_confluence_comments(
    base_url="https://example.atlassian.net/wiki",
    page_id="12345",
    email=email,
    api_token=api_token
)

for comment in comments:
    print(f"Comment {comment['id']}: {comment['body']}")
```

### Load Test Data

```python
from confluence import fetch_test_comments

# Loads from ui/test_comments.json
comments = fetch_test_comments()

for comment in comments:
    print(f"ID: {comment['id']}, Body: {comment['body']}")
```

## Configuration

Set environment variables for automatic credential loading:

```bash
export CONFLUENCE_EMAIL="your-email@example.com"
export CONFLUENCE_API_TOKEN="your-api-token-here"
```

Get API token at: https://id.atlassian.com/manage-profile/security/api-tokens

## Functions

### `parse_confluence_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]`

Parse Confluence URL to extract base URL, space key, and page ID.

**Returns:** `(base_url, space_key, page_id)` or `(None, None, None)` if invalid

### `fetch_confluence_page(base_url: str, page_id: str, email: str, api_token: str) -> Optional[Dict]`

Fetch page content via REST API with authentication.

**Returns:** Dictionary with page data (title, body, version, space) or None

### `fetch_confluence_comments(base_url: str, page_id: str, email: str, api_token: str) -> Optional[List[Dict]]`

Fetch comments for a page.

**Returns:** List of comments with id, status, and body fields

### `fetch_test_comments(test_file_path: Optional[str] = None) -> Optional[List[Dict]]`

Load test comments from JSON file (defaults to `ui/test_comments.json`).

**Returns:** List of comment dictionaries

### `strip_html_tags(html: str) -> str`

Remove HTML tags from content.

**Returns:** Plain text

### `get_confluence_credentials() -> Tuple[str, str]`

Get credentials from environment variables.

**Returns:** `(email, api_token)` tuple

## Test Data Format

The `test_comments.json` file should follow this structure:

```json
{
  "results": [
    {
      "id": "123456",
      "status": "current",
      "body": "Comment text here..."
    }
  ]
}
```

## Content Moderation & Rewriting

### Moderate Content

```python
from confluence import moderate_content

raw_response, parsed = moderate_content(
    content="Comment text to analyze",
    backend="OpenAI",  # or "Ollama"
    ollama_model="llama2",
    temperature=0.7,
    max_tokens=1024
)

if parsed:
    is_harmful = parsed['is_harmful']
    severity = parsed['severity']  # low/medium/high/critical
    issues = parsed['issues']
    recommendations = parsed['recommendations']
```

### Rewrite Harmful Content

```python
from confluence import rewrite_harmful_content

rewritten, metadata = rewrite_harmful_content(
    content="Harmful comment text",
    backend="OpenAI",
    ollama_model="llama2",
    temperature=0.3,  # Lower for consistent rewrites
    max_tokens=1024,
    use_rag=False  # Optional: Use RAG corpus for safer rewrites
)

print(f"Original: {metadata['original_length']} chars")
print(f"Rewritten: {metadata['rewritten_length']} chars")
print(f"Filtered keywords: {metadata['filtered_keywords']}")
```

### Process & Rewrite Comments

```python
from confluence import process_and_rewrite_comments

result = process_and_rewrite_comments(
    backend="OpenAI",
    ollama_model="llama2",
    mod_temperature=0.7,      # For analysis
    rewrite_temperature=0.3,  # For rewriting
    max_tokens=1024,
    use_rag=False
)

print(result)  # Markdown-formatted analysis and rewrites
```

## Integration

Used by:
- `ui/confluence_moderator_app.py` - Gradio content moderation UI
- `ui/confluence_rewriter_app.py` - Gradio content rewriting UI (NEW!)
- Future moderation scripts and batch processors

## API Reference

Confluence REST API: https://developer.atlassian.com/cloud/confluence/rest/v1/intro/
