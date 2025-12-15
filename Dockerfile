# Multi-stage build for optimal image size
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir \
    transformers \
    datasets \
    trl \
    peft \
    accelerate \
    sentence-transformers \
    openai \
    tqdm \
    numpy \
    lm-eval \
    langchain \
    langchain_community \
    chromadb \
    fastapi \
    uvicorn[standard] \
    pydantic \
    streamlit \
    gradio \
    sqlalchemy

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY core/ ./core/
COPY confluence/ ./confluence/
COPY golden/ ./golden/
COPY ui/ ./ui/
COPY custom_system_prompt.txt ./custom_system_prompt.txt

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_ultrafeedback_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7862

# Expose Gradio port
EXPOSE 7862

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7862/ || exit 1

# Set the entrypoint
ENTRYPOINT ["python3", "/app/ui/confluence_rewriter_app.py"]
