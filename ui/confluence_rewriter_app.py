#!/usr/bin/env python3

"""
Gradio SAFEGUARD Content Rewriter
Analyzes test content, detects harmful content, and rewrites them using AI sanitization
"""

import gradio as gr
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DEFAULT_LLM_TEMPERATURE, DEFAULT_SANITIZER_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_SYSTEM_PROMPT
from confluence import process_and_rewrite_comments as _process_and_rewrite, rewrite_all_comments as _rewrite_all


def load_test_content():
    """Load and format test_comments.json for display"""
    try:
        test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_comments.json')
        with open(test_file, 'r') as f:
            data = json.load(f)
            comments = data.get('results', [])
            
            # Format as readable rows
            formatted = ""
            for idx, comment in enumerate(comments, 1):
                comment_id = comment.get('id', 'Unknown')
                body = comment.get('body', '')
                # Truncate long content
                preview = body[:100] + "..." if len(body) > 100 else body
                formatted += f"{idx}. ID: {comment_id}\n   {preview}\n\n"
            
            return formatted if formatted else "No content found"
    except Exception as e:
        return f"Error loading content: {str(e)}"


def process_with_moderation_ui(
    backend: str,
    ollama_model: str,
    rewrite_temperature: float,
    max_tokens: int,
    use_rag: bool,
    progress=gr.Progress()
):
    """Wrapper for moderation + rewriting with progress tracking"""
    status_msg = "⏳ **Processing started...** Analyzing and rewriting content."
    yield status_msg, ""
    
    # Use default moderation temperature
    result = _process_and_rewrite(
        backend, 
        ollama_model, 
        DEFAULT_LLM_TEMPERATURE,  # Fixed moderation temperature
        rewrite_temperature, 
        max_tokens, 
        use_rag, 
        progress
    )
    
    final_status = "✅ **Processing complete!**"
    yield final_status, result


def rewrite_all_ui(
    backend: str,
    ollama_model: str,
    rewrite_temperature: float,
    max_tokens: int,
    use_rag: bool,
    system_prompt: str,
    progress=gr.Progress()
):
    """Wrapper for rewriting all content without moderation"""
    status_msg = "⏳ **Processing started...** Rewriting all content."
    yield status_msg, ""
    
    result = _rewrite_all(
        backend, 
        ollama_model, 
        rewrite_temperature, 
        max_tokens, 
        use_rag,
        system_prompt,
        progress
    )
    
    final_status = "✅ **Processing complete!**"
    yield final_status, result


def save_system_prompt(prompt_text):
    """Save the system prompt to a file"""
    try:
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'custom_system_prompt.txt')
        with open(prompt_file, 'w') as f:
            f.write(prompt_text)
        return "✅ System prompt saved successfully!"
    except Exception as e:
        return f"❌ Error saving prompt: {str(e)}"


def load_system_prompt():
    """Load saved system prompt or return default"""
    try:
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'custom_system_prompt.txt')
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                return f.read()
    except:
        pass
    return DEFAULT_SYSTEM_PROMPT


def create_ui():
    """Create Gradio UI for SAFEGUARD content rewriting"""
    with gr.Blocks(title="SAFEGUARD Content Rewriter") as demo:
        gr.Markdown("# 🔄 SAFEGUARD Content Rewriter")
        gr.Markdown("Analyze and rewrite test content using AI sanitization")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Rewrite All"):                    
                    # Show test content preview
                    gr.Markdown("### 📄 Content (from test_comments.json)")
                    test_content_display = gr.Textbox(
                        value=load_test_content(),
                        lines=8,
                        max_lines=15,
                        label="Content Preview",
                        interactive=False
                    )
                    
                    rewrite_all_btn = gr.Button(
                        "✏️ Rewrite All Content",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Tab("Analyze & Rewrite"):
                    gr.Markdown("**Analyze content first, then rewrite only harmful ones**")
                    rewrite_harmful_btn = gr.Button(
                        "🔄 Analyze & Rewrite Harmful Content",
                        variant="secondary",
                        size="lg"
                    )
                
                status_indicator = gr.Markdown("")
                output = gr.Markdown(label="Results: Before & After")
            
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
                
                gr.Markdown("### 🎨 System Prompt")
                system_prompt = gr.Textbox(
                    label="System Prompt for Rewriting",
                    value=load_system_prompt(),
                    lines=10,
                    max_lines=20,
                    placeholder="Enter custom system prompt for the rewriter...",
                    info="This prompt guides how content is rewritten"
                )
                
                with gr.Row():
                    save_prompt_btn = gr.Button("💾 Save Prompt", size="sm")
                    reset_prompt_btn = gr.Button("🔄 Reset to Default", size="sm")
                
                save_status = gr.Markdown("")
                
                use_rag = gr.Checkbox(
                    label="Use Reference Corpus (UltraFeedback, RAG)",
                    value=False,
                    info="Optional: Use safe examples to guide rewriting"
                )

                rewrite_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_SANITIZER_TEMPERATURE,
                    step=0.1,
                    label="Rewrite Temperature",
                    info="Lower values (0.1-0.3) for consistent rewrites"
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
        
        def reset_prompt():
            return DEFAULT_SYSTEM_PROMPT, "🔄 Reset to default prompt"
        
        backend.change(
            update_backend_visibility,
            inputs=[backend],
            outputs=[ollama_model]
        )
        
        save_prompt_btn.click(
            save_system_prompt,
            inputs=[system_prompt],
            outputs=[save_status]
        )
        
        reset_prompt_btn.click(
            reset_prompt,
            outputs=[system_prompt, save_status]
        )
        
        rewrite_all_btn.click(
            rewrite_all_ui,
            inputs=[
                backend, 
                ollama_model,
                rewrite_temperature,
                max_tokens, 
                use_rag,
                system_prompt
            ],
            outputs=[status_indicator, output],
            show_progress="full"
        )
        
        rewrite_harmful_btn.click(
            process_with_moderation_ui,
            inputs=[
                backend, 
                ollama_model,
                rewrite_temperature,
                max_tokens, 
                use_rag
            ],
            outputs=[status_indicator, output],
            show_progress="full"
        )
        
        gr.Markdown("""
        ### 📝 Two Modes Available
        
        **1. Rewrite All (Recommended)**
        - Rewrites ALL content without analyzing them first
        - Faster processing
        - Uses the custom system prompt
        - Good for blanket sanitization
        - Shows system prompt in output
        
        **2. Analyze & Rewrite**
        - First analyzes each content for harmful material
        - Only rewrites content detected as harmful
        - Slower but more selective
        - Shows moderation analysis results
        
        ### 🎨 System Prompt
        
        The system prompt is now always visible and guides how content is rewritten:
        - Edit the prompt text directly
        - Click "Save Prompt" to persist changes
        - Click "Reset to Default" to restore original
        - The prompt will be shown in the output when using "Rewrite All" mode
        
        ### 🔧 Setup
        
        **Required Environment Variables:**
        ```bash
        export OPENAI_API_KEY="sk-..."  # For OpenAI backend
        # OR configure Ollama locally for Ollama backend
        ```
        
        ### 📊 Output
        
        For each content item, you'll see:
        - **Original Content:** The raw text
        - **Rewritten Content:** Sanitized version
        - *(Rewrite All)* System prompt used
        - *(Analyze & Rewrite only)* Moderation analysis with severity level
        
        ### 🧪 Test Data
        
        The app processes 4 sample content items from `ui/test_comments.json`
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    )
