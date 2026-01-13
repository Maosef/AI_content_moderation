# RAG Pipeline for XML Documents

This directory contains scripts to build and query a Retrieval-Augmented Generation (RAG) pipeline for XML documents.

## Overview

The RAG pipeline processes XML documents from the `data/` directory, chunks them into manageable pieces, creates embeddings using transformer models, and stores them in a ChromaDB vector database for efficient semantic search.

## Architecture

```
┌─────────────────┐
│  XML Documents  │
│   (data/)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract & Chunk │ (RecursiveCharacterTextSplitter)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create          │ (Sentence Transformers)
│ Embeddings      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Store in        │ (ChromaDB)
│ Vector DB       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query & Retrieve│ (Similarity Search)
└─────────────────┘
```

## Components

### 1. `build_rag_pipeline.py`
Main script to build the RAG pipeline from XML documents.

**Features:**
- Loads all XML files from the data directory recursively
- Extracts text content from XML
- Chunks documents using LangChain's RecursiveCharacterTextSplitter
- Creates embeddings using the Golden Embedding system
- Stores in ChromaDB vector database
- Supports both CPU and GPU processing

### 2. `test_rag_pipeline.py`
Script to test and query the built RAG pipeline.

**Features:**
- Loads existing vector database
- Supports single query mode
- Interactive query mode for exploration
- Returns top-k most relevant chunks

### 3. Golden Retriever Components
Located in `golden/` directory:
- `golden_retriever.py`: Custom ChromaDB wrapper with advanced features
- `golden_embeddings.py`: Embedding generation using transformer models
- Supports multiple embedding models

## Quick Start

### Step 1: Build the RAG Pipeline

```bash
# Using default settings (CPU, sentence-transformers/all-MiniLM-L6-v2)
python build_rag_pipeline.py

# With custom settings
python build_rag_pipeline.py \
    --data-dir /path/to/xml/files \
    --persist-dir ./my_vector_db \
    --device cuda \
    --chunk-size 512 \
    --chunk-overlap 20
```

**Options:**
- `--data-dir`: Directory containing XML files (default: `/home/ec2-user/git/data_sanitizer/data`)
- `--persist-dir`: Where to save vector database (default: `./chroma_ultrafeedback_cache`)
- `--model-id`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--chunk-size`: Size of text chunks (default: `512`)
- `--chunk-overlap`: Overlap between chunks (default: `20`)
- `--device`: `cpu` or `cuda` (default: `cpu`)
- `--batch-size`: Batch size for embeddings (default: `auto`)
- `--max-batch-size`: Maximum batch size (default: `64`)

### Step 2: Test/Query the Pipeline

#### Single Query Mode
```bash
python test_rag_pipeline.py --query "procurement aircraft" --top-k 5
```

#### Interactive Mode
```bash
python test_rag_pipeline.py
```

In interactive mode, you can enter multiple queries and see results immediately:
```
Query: missile procurement
Query: budget activity
Query: quit
```

**Options:**
- `--persist-dir`: Vector database location (default: `./chroma_ultrafeedback_cache`)
- `--query`: Single query string (optional, triggers interactive mode if not provided)
- `--top-k`: Number of results to retrieve (default: `5`)
- `--device`: `cpu` or `cuda` (default: `cpu`)

## Integration with Existing Code

The RAG pipeline is already integrated with the core system in [`core/rag.py`](core/rag.py):

```python
from core.rag import load_rag_retriever, retrieve_top_k, build_rag_prompt

# Load the retriever
load_rag_retriever(device="cpu")

# Retrieve relevant documents
docs = retrieve_top_k("your query", k=3)

# Build RAG-enhanced prompt
prompt = build_rag_prompt(docs, "your query")
```

## Embedding Models

The pipeline supports multiple embedding models. Default is `sentence-transformers/all-MiniLM-L6-v2`.

Other supported models (see `golden/golden_embeddings.py`):
- `togethercomputer/m2-bert-80M-8k-retrieval` (8k context)
- `thenlper/gte-large`
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct`

To use a different model:
```bash
python build_rag_pipeline.py \
    --model-id "thenlper/gte-large" \
    --chunk-size 512
```

## Performance Tips

### GPU Acceleration
If you have a CUDA-compatible GPU:
```bash
python build_rag_pipeline.py --device cuda --batch-size auto
```

### Batch Size
- Use `--batch-size auto` to automatically detect optimal batch size
- Or set manually: `--batch-size 32`
- Larger batches = faster processing (but more memory)

### Chunk Size
- Smaller chunks (256-512): Better for precise matching
- Larger chunks (1024-2048): Better for context
- Default 512 is a good balance

## Data Structure

### Input Data
XML files are located in:
- `data/Procurement XML/` - Procurement documents
- `data/RDTE XML/` - RDTE (Research, Development, Test & Evaluation) documents

### Output
Vector database saved to: `./chroma_ultrafeedback_cache/` (configurable)
- `chroma.sqlite3` - SQLite database
- `embedding_settings.json` - Model configuration
- Collection data

## Troubleshooting

### Out of Memory
If you get OOM errors:
1. Reduce batch size: `--batch-size 16`
2. Reduce chunk size: `--chunk-size 256`
3. Use CPU instead of GPU: `--device cpu`

### No Results Found
Check that:
1. Vector database was built successfully
2. `--persist-dir` matches between build and test scripts
3. Database contains documents: Check log for "Total chunks stored"

### Slow Performance
1. Use GPU if available: `--device cuda`
2. Increase batch size: `--batch-size 64`
3. Use a smaller/faster model

## Example Workflow

```bash
# 1. Build the RAG pipeline with GPU acceleration
python build_rag_pipeline.py --device cuda --batch-size auto

# 2. Test with a single query
python test_rag_pipeline.py --query "aircraft procurement budget" --top-k 3

# 3. Explore in interactive mode
python test_rag_pipeline.py

# 4. Use in your application
python -c "
from core.rag import load_rag_retriever, retrieve_top_k
load_rag_retriever()
docs = retrieve_top_k('missile systems', k=5)
for doc in docs:
    print(doc)
"
```

## Customization

### Custom XML Parsing
Edit `build_rag_pipeline.py` function `extract_text_from_xml()` to customize how text is extracted from XML files.

### Custom Chunking
Modify chunk parameters in `build_rag_pipeline.py`:
- `chunk_size`: Maximum chunk size in characters
- `chunk_overlap`: Overlap between consecutive chunks
- `language`: Set to specific language for language-aware chunking (e.g., "python")

### Custom Similarity Function
In `build_rag_pipeline.py`, you can specify similarity function when creating the retriever:
- `l2`: Euclidean distance
- `cosine`: Cosine similarity (default)
- `ip`: Inner product

## Files Created

After running the pipeline:
- `chroma_ultrafeedback_cache/` - Vector database directory
- `golden_retriever.log` - Detailed build logs
- `embedding_settings.json` - Model configuration (inside persist dir)

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- Golden Retriever: Custom retrieval library in `golden/` directory
