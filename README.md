# IRONCLAD Docker Deployment Guide

This guide explains how to containerize and deploy the IRONCLAD Sanitizer Gradio application.

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
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The application will be available at: **http://localhost:7862**

## Manual Docker Build

If you prefer to use Docker directly without docker-compose:

### Build the Image

```bash
docker build -t safeguard-rewriter:latest .
```

### Run the Container

```bash
docker run -d \
  --name safeguard-rewriter \
  -p 7862:7862 \
  -e OPENAI_API_KEY=sk-your-api-key \
  -v $(pwd)/chroma_ultrafeedback_cache:/app/chroma_ultrafeedback_cache \
  -v $(pwd)/custom_system_prompt.txt:/app/custom_system_prompt.txt \
  -v $(pwd)/ui/test_comments.json:/app/ui/test_comments.json \
  safeguard-rewriter:latest
```

### View Logs

```bash
docker logs -f safeguard-rewriter
```

### Stop and Remove

```bash
docker stop safeguard-rewriter
docker rm safeguard-rewriter
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
  safeguard-rewriter:
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

## Resource Limits

Default resource limits in `docker-compose.yml`:

- **CPU**: 1-2 cores
- **Memory**: 2-4 GB

Adjust based on your workload:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

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
- Check network connectivity: `docker exec safeguard-rewriter curl -I https://api.openai.com`

### Can't connect to Ollama

```bash
# Test Ollama connectivity from container
docker exec safeguard-rewriter curl http://host.docker.internal:11434/api/tags

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
docker inspect safeguard-rewriter | grep -A 10 Health
```

## Production Deployment

### Security Recommendations

1. **Don't expose port publicly without authentication**
   - Gradio has built-in sharing, but consider adding a reverse proxy
   - Use HTTPS with proper SSL certificates

2. **Environment variables**
   - Never commit `.env` to version control
   - Use secrets management (Docker secrets, Kubernetes secrets, etc.)

3. **Resource limits**
   - Set appropriate CPU/memory limits
   - Monitor resource usage

### Reverse Proxy Example (nginx)

```nginx
server {
    listen 80;
    server_name safeguard.example.com;

    location / {
        proxy_pass http://localhost:7862;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Health Check Endpoint

The container includes a health check on port 7862:

```bash
curl http://localhost:7862/
```

## Development Workflow

### Live Code Editing

Mount your source code for development:

```yaml
services:
  safeguard-rewriter:
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
  safeguard-rewriter:latest
```

## Cleaning Up

### Remove containers and volumes

```bash
# Stop and remove containers
docker-compose down

# Remove volumes too (WARNING: deletes ChromaDB cache)
docker-compose down -v
```

### Remove images

```bash
docker rmi safeguard-rewriter:latest
```

### Clean build cache

```bash
docker builder prune
```

## Image Size Optimization

The Dockerfile uses multi-stage builds to minimize image size:

- **Builder stage**: Compiles dependencies
- **Production stage**: Only runtime files
- **Excluded**: Virtual env, docs, test files, model artifacts

Current image size: ~2-3 GB (mostly PyTorch/transformers dependencies)

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t safeguard-rewriter:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push safeguard-rewriter:${{ github.sha }}
```

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- Review the main README.md
- Check Gradio documentation: https://gradio.app/docs/
