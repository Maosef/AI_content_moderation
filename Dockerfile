FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with uv (much faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy core module (needed for sanitize_query import)
COPY core/ ./core/

# Copy API application code
COPY api/api_server.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
