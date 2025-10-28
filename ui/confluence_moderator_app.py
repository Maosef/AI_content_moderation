#!/usr/bin/env python3

"""
Gradio Confluence Content Moderator
Fetches Confluence pages and checks for prompt injections or harmful content
"""

import gradio as gr
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DEFAULT_LLM_TEMPERATURE, DEFAULT_MAX_TOKENS
from confluence import process_confluence_page as _process_page, process_test_comments as _process_comments


def process_confluence_page_ui(
    url: str,
    backend: str,
    ollama_model: str,
    temperature: float,
    max_tokens: int,
    progress=gr.Progress()
) -> str:
    """Wrapper for Gradio UI with progress tracking"""
    return _process_page(url, backend, ollama_model, temperature, max_tokens, progress)


def process_test_comments_ui(
    backend: str,
    ollama_model: str,
    temperature: float,
    max_tokens: int,
    progress=gr.Progress()
) -> str:
    """Wrapper for Gradio UI with progress tracking"""
    return _process_comments(backend, ollama_model, temperature, max_tokens, progress)


def create_ui():
    """Create Gradio UI for Confluence moderation"""
    with gr.Blocks(title="Confluence Content Moderator") as demo:
        gr.Markdown("# 🛡️ Confluence Content Moderator")
        gr.Markdown("Analyze Confluence pages for prompt injections and harmful content")
        
        with gr.Row():
            with gr.Column(scale=2):
                confluence_url = gr.Textbox(
                    label="Confluence Page URL",
                    placeholder="https://yoursite.atlassian.net/wiki/spaces/SPACE/pages/12345",
                    lines=1
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Page", variant="primary")
                    test_btn = gr.Button("Analyze Test Comments", variant="secondary")
                
                output = gr.Markdown(label="Analysis Results")
            
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Configuration")
                
                backend = gr.Dropdown(
                    choices=["OpenAI", "Ollama"],
                    value="OpenAI",
                    label="LLM Backend"
                )
                
                ollama_model = gr.Textbox(
                    label="Ollama Model",
                    value="llama2",
                    visible=False
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_LLM_TEMPERATURE,
                    step=0.1,
                    label="Temperature"
                )
                
                max_tokens = gr.Slider(
                    minimum=256,
                    maximum=4096,
                    value=1024,
                    step=256,
                    label="Max Tokens"
                )
        
        def update_backend_visibility(backend_choice):
            return gr.update(visible=(backend_choice == "Ollama"))
        
        backend.change(
            update_backend_visibility,
            inputs=[backend],
            outputs=[ollama_model]
        )
        
        analyze_btn.click(
            process_confluence_page_ui,
            inputs=[confluence_url, backend, ollama_model, temperature, max_tokens],
            outputs=[output]
        )
        
        test_btn.click(
            process_test_comments_ui,
            inputs=[backend, ollama_model, temperature, max_tokens],
            outputs=[output]
        )
        
        gr.Markdown("""
        ### 📝 Instructions
        
        **Option 1: Analyze Confluence Page**
        1. **Set Environment Variables:**
           ```bash
           export CONFLUENCE_EMAIL="your-email@example.com"
           export CONFLUENCE_API_TOKEN="your-api-token"
           export OPENAI_API_KEY="sk-..."  # or configure Ollama
           ```
           Get API token at: [Atlassian Account Settings](https://id.atlassian.com/manage-profile/security/api-tokens)
        
        2. **Enter Confluence URL:** Full URL to the page you want to analyze
        3. **Click "Analyze Page"**
        
        **Option 2: Analyze Test Comments**
        1. Click "Analyze Test Comments" to analyze sample data from `ui/test_comments.json`
        2. Great for testing without Confluence credentials!
        3. Only requires LLM backend configuration (OPENAI_API_KEY or Ollama)
        
        **Note:** Credentials are loaded from environment variables only.
        """)
        
        gr.Markdown("""
        ### 🔒 Future Features
        
        - [ ] Flag and highlight harmful content sections
        - [ ] Automatic content rewriting suggestions
        - [ ] Bulk page analysis
        - [ ] Export moderation reports
        - [ ] Integration with MCP (Model Context Protocol)
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )
