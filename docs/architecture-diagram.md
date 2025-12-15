# TLDD Sanitize Web Application - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER / CLIENT                              │
│                         (Browser - Port 7862)                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DOCKER CONTAINER                                 │
│                     safeguard-rewriter:7862                             │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      GRADIO WEB UI                                │ │
│  │              (ui/confluence_rewriter_app.py)                      │ │
│  │                                                                   │ │
│  │  • Test Content Viewer                                            │ │
│  │  • Moderation + Rewriting Interface                               │ │
│  │  • Bulk Rewriting Interface                                       │ │
│  │  • System Prompt Editor                                           │ │
│  │  • Progress Tracking                                              │ │
│  └──────────────────┬────────────────────────────────────────────────┘ │
│                     │                                                   │
│                     ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      CORE LOGIC LAYER                             │ │
│  │                                                                   │ │
│  │  ┌─────────────────────┐      ┌──────────────────────┐          │ │
│  │  │   Content Moderator │      │   Content Sanitizer  │          │ │
│  │  │  (confluence/       │◄────►│   (core/sanitizer.py)│          │ │
│  │  │   moderator.py)     │      │                      │          │ │
│  │  │                     │      │  • Query Rewriting   │          │ │
│  │  │ • Harm Detection    │      │  • TLDD Algorithm    │          │ │
│  │  │ • Content Analysis  │      │  • Keyword Filtering │          │ │
│  │  └─────────┬───────────┘      └──────────┬───────────┘          │ │
│  │            │                               │                      │ │
│  │            └───────────────┬───────────────┘                      │ │
│  │                            ▼                                      │ │
│  │                  ┌──────────────────────┐                        │ │
│  │                  │    LLM Client        │                        │ │
│  │                  │ (core/llm_client.py) │                        │ │
│  │                  │                      │                        │ │
│  │                  │ • Unified Interface  │                        │ │
│  │                  │ • Backend Selection  │                        │ │
│  │                  └──────────┬───────────┘                        │ │
│  │                             │                                     │ │
│  └─────────────────────────────┼─────────────────────────────────────┘ │
│                                │                                       │
│                    ┌───────────┴──────────┐                           │
│                    ▼                      ▼                           │
│         ┌──────────────────┐   ┌──────────────────┐                  │
│         │   RAG System     │   │  Configuration   │                  │
│         │  (core/rag.py)   │   │ (core/config.py) │                  │
│         │                  │   │                  │                  │
│         │ • Vector Search  │   │ • Defaults       │                  │
│         │ • Context Build  │   │ • Prompts        │                  │
│         └────────┬─────────┘   │ • Keywords       │                  │
│                  │              └──────────────────┘                  │
│                  ▼                                                    │
│         ┌──────────────────┐                                          │
│         │   ChromaDB       │                                          │
│         │  Vector Store    │                                          │
│         │ (Persistent Vol) │                                          │
│         └──────────────────┘                                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼
┌──────────────────────────────┐   ┌─────────────────────────────────┐
│   EXTERNAL LLM SERVICES      │   │    LOCAL LLM OPTION             │
│                              │   │                                 │
│   OpenAI API                 │   │   Ollama (Optional)             │
│   • GPT-4o / GPT-3.5         │   │   • http://host.docker.internal │
│   • Chat Completions         │   │     :11434                      │
│   • Analysis & Rewriting     │   │   • Local Models                │
│   • Env: OPENAI_API_KEY      │   │   • Env: OLLAMA_HOST            │
└──────────────────────────────┘   └─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         PERSISTENT STORAGE                              │
│                                                                         │
│  Volume: chroma_data                                                    │
│  ├─ /app/chroma_ultrafeedback_cache/  (ChromaDB vector database)       │
│                                                                         │
│  Volume Mounts:                                                         │
│  ├─ ./custom_system_prompt.txt  (Editable system prompt)               │
│  └─ ./ui/test_comments.json     (Test data for rewriting)              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### **Web UI Layer (Gradio)**
- **Framework**: Gradio 5.49.1
- **Port**: 7862 (exposed to host)
- **Features**:
  - Interactive content viewer for test data
  - Dual-mode operation: moderation+rewrite or bulk rewrite
  - Real-time progress tracking
  - System prompt customization
  - Temperature controls for analysis vs. rewriting

### **Core Processing Layer**

#### **Content Moderator** (`confluence/moderator.py`)
- Analyzes content for harmful patterns
- Detects prompt injections, budget manipulation, etc.
- Returns analysis with harm classification

#### **Content Sanitizer** (`core/sanitizer.py`)
- Implements TLDD (Training-time Language-based Data Decontamination)
- Rewrites harmful content into safe alternatives
- Keyword filtering system
- Temperature-controlled generation

#### **LLM Client** (`core/llm_client.py`)
- Unified interface for multiple LLM backends
- Supports OpenAI API and Ollama
- Backend selection via configuration
- Handles API authentication and requests

#### **RAG System** (`core/rag.py`)
- Retrieval-Augmented Generation
- Uses ChromaDB for vector storage
- Sentence-transformers for embeddings
- Top-K document retrieval
- Context building for enhanced rewrites

### **Data Storage**

#### **ChromaDB**
- Vector database for RAG functionality
- Persistent volume: `chroma_data`
- Stores ultra-feedback cache
- Enables semantic search for similar content

#### **Configuration Files**
- `custom_system_prompt.txt`: Editable prompt template
- `ui/test_comments.json`: Test data for rewriting
- `.env`: Environment variables (API keys, etc.)

### **External Services**

#### **OpenAI API** (Primary)
- Models: GPT-4o, GPT-3.5, etc.
- Used for content analysis and rewriting
- Requires `OPENAI_API_KEY`

#### **Ollama** (Optional)
- Local LLM alternative
- Accessible via `host.docker.internal:11434`
- Configurable via `OLLAMA_HOST`
- No API key required

## Data Flow

### **Moderation + Rewrite Flow**
```
1. User Input (Test Comment)
   ↓
2. Content Moderator Analysis (OpenAI/Ollama)
   ↓
3. Harm Detection & Classification
   ↓
4. If Harmful → Sanitizer (TLDD Rewrite)
   ↓
5. Optional: RAG Enhancement (ChromaDB lookup)
   ↓
6. LLM Generation (Rewritten Content)
   ↓
7. Keyword Filtering Check
   ↓
8. Return: Before/After + Metadata
```

### **Bulk Rewrite Flow**
```
1. Load All Test Comments
   ↓
2. For Each Comment:
   a. Sanitizer Direct Rewrite
   b. Optional: RAG Enhancement
   c. LLM Generation
   ↓
3. Progress Tracking (Gradio UI)
   ↓
4. Return: Bulk Results with Metadata
```

## Key Technologies

- **Python**: 3.11-slim
- **Web Framework**: Gradio 5.49.1
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers
- **LLM APIs**: OpenAI, Ollama
- **Container**: Docker with multi-stage build
- **Storage**: Docker volumes + file mounts

## Resource Limits

- **CPU**: 1-2 cores (reserved-limit)
- **Memory**: 2-4 GB (reserved-limit)
- **Health Check**: Every 30s on port 7862
- **Restart Policy**: unless-stopped

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API authentication | Required |
| `OLLAMA_HOST` | Local Ollama endpoint | `http://host.docker.internal:11434` |
| `GRADIO_SERVER_NAME` | Gradio bind address | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | Gradio port | `7862` |
| `PYTHONUNBUFFERED` | Python output buffering | `1` |
