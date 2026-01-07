# IRONCLAD Content Moderator

LLM tool that analyzes text for prompt injections and harmful content, moderates/sanitizes.

<img width="2448" height="1233" alt="image" src="https://github.com/user-attachments/assets/f70b91a0-cd9a-4ce5-a64e-041c83429b23" />

## Features

- **LLM-Powered Analysis**: Uses [Llama Prompt Guard 2](https://www.llama.com/docs/model-cards-and-prompt-formats/prompt-guard/) to detect harmful content
- **Security Checks**: Identifies prompt injections, jailbreaks or other harmful content
- **Web application demo**: Runs out of the box with Docker Compose and Gradio

## Architecture

```
tldd_sanitize/
├── confluence/              # Confluence integration module
│   ├── __init__.py         # Module exports
│   ├── client.py           # API client (fetch pages, comments, parse URLs)
│   ├── config.py           # Credential management from env vars
│   └── README.md           # Module documentation
├── ui/
│   ├── confluence_moderator_app.py  # Gradio UI
│   └── test_comments.json           # Sample data for testing
└── core/                   # Shared LLM client
```

# Docker Deployment Guide

This guide explains how to containerize and deploy the Sanitizer demo application.

## Quick Start

### 1. Prerequisites

- Docker installed (20.10+)
- Docker Compose installed (v2.0+)
- OpenAI API key (or local Ollama installation)

### 2. Setup Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Build and Run with Docker Compose

```bash
# Build and start the container
docker compose up -d

# View logs
docker compose logs -f

# Stop the container
docker compose down
```

The application will be available at: **http://localhost:7862**

## Manual Docker Build

If you prefer to use Docker directly without docker-compose:

### Build the Image

```bash
docker build -t ironclad-rewriter:latest .
```

### Run the Container

```bash
docker run -d \
  --name ironclad-rewriter \
  -p 7862:7862 \
  -e OPENAI_API_KEY=sk-your-api-key \
  -v $(pwd)/chroma_ultrafeedback_cache:/app/chroma_ultrafeedback_cache \
  -v $(pwd)/custom_system_prompt.txt:/app/custom_system_prompt.txt \
  -v $(pwd)/ui/test_comments.json:/app/ui/test_comments.json \
  ironclad-rewriter:latest
```

### View Logs

```bash
docker logs -f ironclad-rewriter
```

### Stop and Remove

```bash
docker stop ironclad-rewriter
docker rm ironclad-rewriter
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | Your OpenAI API key |
| `OLLAMA_HOST` | No | `http://host.docker.internal:11434` | Ollama API endpoint |
| `GRADIO_SERVER_NAME` | No | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | No | `7862` | Server port |

*Required if using OpenAI backend. Optional if using Ollama only.

### Using with Ollama

If you have Ollama running locally on the host machine:

1. Make sure Ollama is accessible from Docker
2. Update `.env` with:
   ```bash
   OLLAMA_HOST=http://host.docker.internal:11434
   ```

If Ollama is running in another Docker container:

```yaml
# In docker-compose.yml, add:
services:
  ironclad-rewriter:
    # ... existing config
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
```

## Volume Mounts

The docker-compose configuration mounts several directories/files:

1. **ChromaDB Cache** (`chroma_data` volume)
   - Persists the RAG embeddings database
   - Avoids rebuilding the index on each restart

2. **Custom System Prompt** (`./custom_system_prompt.txt`)
   - Allows editing the prompt without rebuilding
   - Changes persist between container restarts

3. **Test Comments** (`./ui/test_comments.json`)
   - Easily update test data without rebuilding
   - Hot-reload new test cases

## Troubleshooting

### Container won't start

```bash
# Check logs for errors
docker-compose logs

# Verify environment variables
docker-compose config
```

### Can't connect to OpenAI

- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check network connectivity: `docker exec ironclad-rewriter curl -I https://api.openai.com`

### Can't connect to Ollama

```bash
# Test Ollama connectivity from container
docker exec ironclad-rewriter curl http://host.docker.internal:11434/api/tags

# On Linux, you may need to use host network mode
docker run --network host ...
```

### Permission issues with volumes

```bash
# Ensure files have correct permissions
chmod 644 custom_system_prompt.txt
chmod 644 ui/test_comments.json
```

### Application not accessible

```bash
# Check if port is already in use
lsof -i :7862

# Check container health
docker inspect ironclad-rewriter | grep -A 10 Health
```

## Production Deployment

### Security Recommendations

1. **Don't expose port publicly without authentication**
   - Gradio has built-in sharing, but consider adding a reverse proxy
   - Use HTTPS with proper SSL certificates

2. **Environment variables**
   - Never commit `.env` to version control
   - Use secrets management (Docker secrets, Kubernetes secrets, etc.)

## Development Workflow

### Live Code Editing

Mount your source code for development:

```yaml
services:
  ironclad-rewriter:
    volumes:
      - ./core:/app/core
      - ./confluence:/app/confluence
      - ./ui:/app/ui
```

Then restart to apply changes:

```bash
docker-compose restart
```

### Debugging

Run container with interactive shell:

```bash
docker run -it --rm \
  --entrypoint /bin/bash \
  -e OPENAI_API_KEY=sk-your-key \
  ironclad-rewriter:latest
```
