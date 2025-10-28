#!/usr/bin/env python3

"""
Gradio Chat Application with SafePrompt Sanitization
Interactive chat interface with optional query rewriting/sanitization
Refactored to use modular architecture
"""

import gradio as gr
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    get_llm_client, llm_generate, load_rag_retriever, get_rag_doc_count,
    sanitize_query, check_filtered_keywords,
    DEFAULT_LLM_SYSTEM_PROMPT, DEFAULT_OLLAMA_MODEL, DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_TOKENS, DEFAULT_SANITIZER_TEMPERATURE, DEFAULT_SANITIZER_MODEL
)

from typing import List

def chat_fn(message: str, history: List[List[str]], 
            backend: str, ollama_model: str, enable_sanitization: bool,
            sanitizer_backend: str, sanitizer_model: str,
            use_rag: bool, sanitizer_temp: float, llm_temp: float,
            max_tokens: int, system_prompt: str):
    """Main chat function"""
    try:
        # Get LLM client
        llm_client, llm_model = get_llm_client(backend, ollama_model)
        
        # Store original message
        original_message = message
        sanitization_info = ""
        
        # Sanitize if enabled
        if enable_sanitization:
            sanitized_message = sanitize_query(message, use_rag, sanitizer_backend, 
                                              sanitizer_model, sanitizer_temp)
            
            if sanitized_message != message:
                filtered = check_filtered_keywords(message)
                sanitization_info = f"🛡️ **Sanitized Query:**\n\n"
                sanitization_info += f"**Original:** {message}\n\n"
                sanitization_info += f"**Sanitized:** {sanitized_message}\n\n"
                if filtered:
                    sanitization_info += f"⚠️ **Filtered keywords:** {', '.join(filtered)}\n\n"
                sanitization_info += "---\n\n"
                message = sanitized_message
        
        # Generate LLM response
        response = llm_generate(llm_client, llm_model, system_prompt, 
                               message, llm_temp, max_tokens)
        
        if not response:
            response = "Failed to generate response. Please try again."
        
        # Prepend sanitization info if available
        if sanitization_info:
            response = sanitization_info + response
        
        # Return updated history
        history.append([original_message, response])
        return history
    
    except Exception as e:
        history.append([message, f"Error: {str(e)}"])
        return history


def create_ui():
    """Create Gradio UI"""
    with gr.Blocks(title="SafePrompt Chat Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 SafePrompt Chat Interface")
        gr.Markdown("Chat with LLM with optional query sanitization")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    render_markdown=True
                )
                
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Enter your message here...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
            
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Configuration")
                
                # LLM Backend
                backend = gr.Dropdown(
                    choices=["OpenAI", "Ollama"],
                    value="OpenAI",
                    label="LLM Backend"
                )
                
                ollama_model = gr.Textbox(
                    label="Ollama Model",
                    value=DEFAULT_OLLAMA_MODEL,
                    visible=False
                )
                
                # Sanitization settings
                gr.Markdown("### 🛡️ Sanitization")
                enable_sanitization = gr.Checkbox(
                    label="Enable Query Sanitization",
                    value=False
                )
                
                sanitizer_backend = gr.Dropdown(
                    choices=["Ollama", "OpenAI"],
                    value="Ollama",
                    label="Sanitizer Backend",
                    visible=False
                )
                
                sanitizer_model = gr.Textbox(
                    label="Sanitizer Model",
                    value=DEFAULT_SANITIZER_MODEL,
                    visible=False
                )
                
                use_rag = gr.Checkbox(
                    label="Use reference corpus (UltraFeedback, RAG)",
                    value=False,
                    visible=False
                )
                
                sanitizer_temp = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_SANITIZER_TEMPERATURE,
                    step=0.05,
                    label="Sanitizer Temperature",
                    visible=False
                )
                
                # LLM settings
                gr.Markdown("### 🤖 Generation")
                llm_temp = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=DEFAULT_LLM_TEMPERATURE,
                    step=0.1,
                    label="LLM Temperature"
                )
                
                max_tokens = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=DEFAULT_MAX_TOKENS,
                    step=128,
                    label="Max Tokens"
                )
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=DEFAULT_LLM_SYSTEM_PROMPT,
                    lines=3
                )
                
                # RAG status
                rag_status = gr.Markdown("", visible=False)
        
        # Event handlers
        def update_backend_visibility(backend_choice):
            return gr.update(visible=(backend_choice == "Ollama"))
        
        def update_sanitization_visibility(enabled):
            return [
                gr.update(visible=enabled),  # sanitizer_backend
                gr.update(visible=enabled),  # sanitizer_model
                gr.update(visible=enabled),  # use_rag
                gr.update(visible=enabled),  # sanitizer_temp
                gr.update(visible=False)     # rag_status (hidden until RAG enabled)
            ]
        
        def update_sanitizer_model_visibility(backend_choice):
            if backend_choice == "Ollama":
                return gr.update(visible=True, value=DEFAULT_SANITIZER_MODEL)
            else:
                return gr.update(visible=False, value="")
        
        def try_load_rag(use_rag_enabled):
            if use_rag_enabled:
                success = load_rag_retriever()
                if success:
                    count = get_rag_doc_count()
                    return gr.update(value=f"✅ RAG loaded ({count} docs)", visible=True)
                else:
                    return gr.update(value="⚠️ RAG not available (cache not found)", visible=True)
            return gr.update(visible=False)
        
        backend.change(
            update_backend_visibility,
            inputs=[backend],
            outputs=[ollama_model]
        )
        
        enable_sanitization.change(
            update_sanitization_visibility,
            inputs=[enable_sanitization],
            outputs=[sanitizer_backend, sanitizer_model, use_rag, sanitizer_temp, rag_status]
        )
        
        sanitizer_backend.change(
            update_sanitizer_model_visibility,
            inputs=[sanitizer_backend],
            outputs=[sanitizer_model]
        )
        
        use_rag.change(
            try_load_rag,
            inputs=[use_rag],
            outputs=[rag_status]
        )
        
        # Chat functionality
        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, backend, ollama_model, enable_sanitization,
                   sanitizer_backend, sanitizer_model, use_rag, sanitizer_temp, 
                   llm_temp, max_tokens, system_prompt],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        submit_btn.click(
            chat_fn,
            inputs=[msg, chatbot, backend, ollama_model, enable_sanitization,
                   sanitizer_backend, sanitizer_model, use_rag, sanitizer_temp, 
                   llm_temp, max_tokens, system_prompt],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        gr.Markdown("""
        ### 📝 Instructions
        1. Select your LLM backend (OpenAI or Ollama)
        2. Toggle sanitization to enable query rewriting
        3. Enable RAG for context-aware sanitization
        4. Adjust temperature and max tokens as needed
        5. Start chatting!
        
        **Note:** Set `OPENAI_API_KEY` environment variable for OpenAI backend.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
