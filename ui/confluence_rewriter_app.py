#!/usr/bin/env python3

"""
Gradio Confluence Content Rewriter
Analyzes test comments, detects harmful content, and rewrites them using AI sanitization
"""

import gradio as gr
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DEFAULT_LLM_TEMPERATURE, DEFAULT_SANITIZER_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_SYSTEM_PROMPT
from confluence import process_and_rewrite_comments as _process_and_rewrite, rewrite_all_comments as _rewrite_all


def process_with_moderation_ui(
    backend: str,
    ollama_model: str,
    rewrite_temperature: float,
    max_tokens: int,
    use_rag: bool,
    progress=gr.Progress()
):
    """Wrapper for moderation + rewriting with progress tracking"""
    status_msg = "⏳ **Processing started...** Analyzing and rewriting comments."
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
    status_msg = "⏳ **Processing started...** Rewriting all comments."
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
    """Create Gradio UI for Confluence content rewriting"""
    with gr.Blocks(title="Confluence Content Rewriter") as demo:
        gr.Markdown("# 🔄 Confluence Content Rewriter")
        gr.Markdown("Analyze and rewrite test comments using AI sanitization")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Rewrite All"):
                    gr.Markdown("**Rewrite all comments without moderation analysis**")
                    rewrite_all_btn = gr.Button(
                        "✏️ Rewrite All Comments",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Tab("Analyze & Rewrite"):
                    gr.Markdown("**Analyze comments first, then rewrite only harmful ones**")
                    rewrite_harmful_btn = gr.Button(
                        "🔄 Analyze & Rewrite Harmful Comments",
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
                
                with gr.Accordion("🎨 System Prompt", open=False):
                    system_prompt = gr.Textbox(
                        label="System Prompt for Rewriting",
                        value=load_system_prompt(),
                        lines=10,
                        max_lines=20,
                        placeholder="Enter custom system prompt for the rewriter...",
                        info="This prompt guides how content is rewritten",
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        save_prompt_btn = gr.Button("💾 Save Prompt", size="sm")
                        reset_prompt_btn = gr.Button("🔄 Reset to Default", size="sm")
                    
                    save_status = gr.Markdown("")
                
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
                
                use_rag = gr.Checkbox(
                    label="Use Reference Corpus (UltraFeedback, RAG)",
                    value=False,
                    info="Optional: Use safe examples to guide rewriting"
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
        - Rewrites ALL comments without analyzing them first
        - Faster processing
        - Uses the custom system prompt
        - Good for blanket sanitization
        
        **2. Analyze & Rewrite**
        - First analyzes each comment for harmful content
        - Only rewrites comments detected as harmful
        - Slower but more selective
        - Shows moderation analysis results
        
        ### 🎨 Custom System Prompt
        
        Edit the system prompt to control how content is rewritten:
        - Click "System Prompt" accordion to expand
        - Modify the prompt text
        - Click "Save Prompt" to persist changes
        - Click "Reset to Default" to restore original
        
        The prompt guides the LLM on how to sanitize content while preserving meaning.
        
        ### 🔧 Setup
        
        **Required Environment Variables:**
        ```bash
        export OPENAI_API_KEY="sk-..."  # For OpenAI backend
        # OR configure Ollama locally for Ollama backend
        ```
        
        ### 📊 Example Output
        
        For each comment, you'll see:
        - **Original Content:** The raw comment text
        - **Rewritten Content:** Sanitized version
        - **Rewrite Metadata:** Length changes, filtered keywords, etc.
        - *(Analyze & Rewrite only)* Moderation analysis with severity level
        
        ### 🧪 Test Data
        
        The app processes 4 sample comments from `ui/test_comments.json`
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
