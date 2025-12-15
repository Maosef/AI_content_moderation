#!/usr/bin/env python3

"""
Gradio SAFEGUARD Passage Rewriter
Analyzes test passages, detects harmful content, and rewrites them using AI sanitization
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

            # Format as readable passages with full content
            formatted = ""
            for comment in comments:
                comment_id = comment.get('id', 'Unknown')
                body = comment.get('body', '')
                formatted += f"Passage {comment_id}:\n{body}\n\n"

            return formatted if formatted else "No passages found"
    except Exception as e:
        return f"Error loading passages: {str(e)}"


def process_with_moderation_ui(
    backend: str,
    ollama_model: str,
    rewrite_temperature: float,
    max_tokens: int,
    use_rag: bool,
    progress=gr.Progress()
):
    """Wrapper for moderation + rewriting with progress tracking"""
    status_msg = "⏳ **Processing started...** Analyzing and rewriting passages."
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
    """Wrapper for rewriting all passages without moderation"""
    status_msg = "⏳ **Processing started...** Rewriting all passages."
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

    # Custom CSS to increase UI size and optimize 3-column layout
    custom_css = """
    .gradio-container {
        font-size: 18px !important;
        max-width: 98% !important;
    }
    button {
        font-size: 18px !important;
        padding: 12px 24px !important;
        min-height: 50px !important;
    }
    .gr-button {
        font-size: 18px !important;
    }
    label {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    .gr-text-input, .gr-textbox, textarea {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    /* Make content/passage text even larger */
    textarea {
        font-size: 20px !important;
    }
    h1 {
        font-size: 2.5em !important;
        margin: 10px 0 !important;
    }
    h2 {
        font-size: 1.8em !important;
        margin: 15px 0 10px 0 !important;
    }
    h3 {
        font-size: 1.4em !important;
        margin: 10px 0 8px 0 !important;
    }
    .gr-markdown {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    .gr-markdown p {
        margin: 8px 0 !important;
    }
    .gr-markdown ul, .gr-markdown ol {
        margin: 5px 0 !important;
    }
    /* Make accordion labels more compact */
    .gr-accordion {
        font-size: 16px !important;
    }
    /* Reduce spacing in results */
    .gr-markdown hr {
        margin: 15px 0 !important;
    }
    """

    with gr.Blocks(title="IRONCLAD Passage Rewriter", css=custom_css) as demo:
        gr.Markdown("# 🔄 IRONCLAD Passage Rewriter")
        gr.Markdown("Analyze and rewrite test passages using AI sanitization")
        
        with gr.Row():
            # Left column: Passage Preview
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Passages (from test_comments.json)")
                test_content_display = gr.Textbox(
                    value=load_test_content(),
                    lines=20,
                    max_lines=40,
                    label="Passage Preview",
                    interactive=False
                )

                with gr.Row():
                    rewrite_all_btn = gr.Button(
                        "✏️ Rewrite All Passages",
                        variant="primary",
                        size="lg"
                    )
                    rewrite_harmful_btn = gr.Button(
                        "🔄 Analyze & Rewrite Harmful",
                        variant="secondary",
                        size="lg"
                    )

            # Middle column: Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Passage Rewriting Results")
                status_indicator = gr.Markdown("")
                output = gr.Markdown(label="Results")

            # Right sidebar: Configuration (collapsible)
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion("⚙️ Configuration", open=True):
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

                    gr.Markdown("**System Prompt**")
                    system_prompt = gr.Textbox(
                        label="System Prompt for Rewriting",
                        value=load_system_prompt(),
                        lines=8,
                        max_lines=15,
                        placeholder="Enter custom system prompt for the rewriter...",
                        info="This prompt guides how passages are rewritten"
                    )

                    with gr.Row():
                        save_prompt_btn = gr.Button("💾 Save", size="sm")
                        reset_prompt_btn = gr.Button("🔄 Reset", size="sm")

                    save_status = gr.Markdown("")

                    use_rag = gr.Checkbox(
                        label="Use Reference Corpus (RAG)",
                        value=False,
                        info="Use safe examples to guide rewriting"
                    )

                    rewrite_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=DEFAULT_SANITIZER_TEMPERATURE,
                        step=0.1,
                        label="Rewrite Temperature",
                        info="Lower = more consistent"
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
        ---
        ### 📝 Two Modes

        **Rewrite All Passages** - Rewrites ALL passages using the system prompt (faster)

        **Analyze & Rewrite Harmful** - First analyzes, then only rewrites harmful passages (selective)

        ### 🔧 Setup

        Set environment variable: `export OPENAI_API_KEY="sk-..."`

        ### 📊 Layout

        **Left:** Passage Preview | **Middle:** Results | **Right:** Configuration (collapsible)
        """)
    
    return demo


# Create demo at module level for Gradio auto-reload
demo = create_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True,
        show_error=True  # Show detailed error messages
    )
