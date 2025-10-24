# **Data Sanitization Service - Technical Architecture Specification**

## **1. Overview**

AI-powered service that sanitizes sensitive, toxic, or malicious content from various data sources (Jira tickets, websites, Confluence pages) using LLM-based rewriting with optional RAG enhancement.

## **2. Core API Interface**

### **2.1 Primary Functions**

```python
def sanitize(input: str, type: InputType, config: SanitizerConfig) -> str:
    """
    Sanitizes input data based on type.
    
    Args:
        input: Raw content string
        type: InputType enum (JIRA_TICKET, WEBSITE, CONFLUENCE_PAGE, GENERIC)
        config: Configuration object with model settings
        
    Returns:
        Sanitized content string
    """

def configure(
    samples: List[str],
    role_prompt: str,
    input_type: InputType,
    use_rag: bool = False,
    rag_top_k: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 256
) -> SanitizerConfig:
    """
    Initialize sanitizer configuration.
    
    Args:
        samples: Example documents for RAG retrieval (if use_rag=True)
        role_prompt: System prompt defining sanitization behavior
        input_type: Type of input data to process
        use_rag: Enable Golden Retriever RAG
        rag_top_k: Number of documents to retrieve
        temperature: LLM generation temperature
        max_tokens: Maximum output tokens
        
    Returns:
        SanitizerConfig object
    """
```

### **2.2 Data Models**

```python
class InputType(Enum):
    JIRA_TICKET = "jira_ticket"
    WEBSITE = "website"
    CONFLUENCE_PAGE = "confluence_page"
    GENERIC = "generic"

class SanitizerConfig:
    role_prompt: str
    input_type: InputType
    use_rag: bool
    rag_retriever: Optional[Golden_Retriever]
    rag_top_k: int
    llm_backend: LLMBackend  # OpenAI or Ollama
    temperature: float
    max_tokens: int
    num_retries: int
```

## **3. Architecture Components**

### **3.1 LLM Backend Layer**
- **OpenAI Client**: GPT-4o or configurable models via OPENAI_API_KEY
- **Ollama Client**: Local model support (default: neuraldaredevil-8b-abliterated)
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Content Filter Handling**: Automatic retry on content_filter rejection

### **3.2 RAG Enhancement (Optional)**
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (configurable)
- **Vector Store**: ChromaDB with persistent caching
- **Document Source**: UltraFeedback dataset or custom samples
- **Retrieval**: Dense similarity search, top-k document selection

### **3.3 Sanitization Pipeline**

```
Input → Type-Specific Parser → Context Builder → LLM Rewriter → Validator → Output
                                      ↓
                               [Optional RAG]
```

1. **Parser**: Extract structured data from input type (JSON for Jira, HTML for web, etc.)
2. **Context Builder**: Construct prompt with retrieved examples (if RAG enabled)
3. **LLM Rewriter**: Generate sanitized version using configured prompt
4. **Validator**: Check for refusal patterns, keyword filtering
5. **Output**: Return sanitized string

## **4. Configuration Modes**

### **4.1 Default System Prompt** (run_tldd.py:115-122)
```
You are a safety auditor, able to efficiently filter malicious, 
toxic, or harmful content and turn them into safe phrases.
Rephrase the query, removing any malicious intent 
(spreading false information, manipulation, propaganda).
Do not over explain, keep responses terse.
Focus on safe, responsible, and ethical phrasing.
```

### **4.2 RAG-Enhanced Mode**
Uses author voice matching from sample documents to maintain authenticity while removing harmful content.

## **5. API Endpoints (Proposed REST API)**

```
POST /api/v1/sanitize
Body: {
  "input": "string",
  "type": "jira_ticket|website|confluence_page|generic",
  "config_id": "string"  // Pre-configured sanitizer
}
Response: {
  "sanitized": "string",
  "confidence": float,
  "filtered_keywords": ["string"]
}

POST /api/v1/config
Body: {
  "role_prompt": "string",
  "samples": ["string"],
  "input_type": "string",
  "use_rag": bool,
  "rag_top_k": int
}
Response: {
  "config_id": "string"
}

POST /api/v1/batch
Body: {
  "inputs": [{"input": "string", "type": "string"}],
  "config_id": "string"
}
Response: {
  "results": [{"sanitized": "string"}]
}
```

## **6. Implementation Details**

### **6.1 Keyword Filtering** (run_tldd.py:108-113)
Refusal detection keywords:
- "I'm sorry", "As an AI", "not ethical", "illegal", "I cannot create", etc.

### **6.2 Performance Optimization**
- **Parallel Processing**: Dataset sharding for batch operations (--num-shards, --shard-id)
- **Map Parallelism**: Multi-process dataset mapping (--map-num-proc)
- **RAG Caching**: Persistent ChromaDB storage (./chroma_ultrafeedback_cache)
- **Device Selection**: CUDA/CPU support for embeddings (--rag-device)

### **6.3 Error Handling**
- Retry on content filter (NUM_RETRIES = 3)
- Fallback to original input if all retries fail
- Graceful degradation if RAG unavailable

## **7. Deployment Considerations**

- **Environment Variables**: OPENAI_API_KEY, OPENAI_MODEL
- **Dependencies**: openai, transformers, golden-retriever, chromadb, torch
- **Storage**: Persistent vector store for RAG (~500MB-2GB)
- **GPU Requirements**: Optional (CPU fallback available)
- **Scalability**: Stateless design, horizontal scaling via sharding

## **8. Security & Privacy**

- No logging of sensitive content (configurable debug mode only)
- Local model option (Ollama) for air-gapped environments
- RAG document isolation per tenant/config
- Rate limiting on API endpoints (recommended)

---

**Based on**: `tldd_sanitize/run_tldd.py`
