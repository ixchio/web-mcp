# Web MCP Server - Agent Guide

> This file helps AI agents understand and work with this repository effectively.

## Project Overview

**Web MCP Server** is a Model Context Protocol server providing:
- Web search via Serper API
- URL content extraction via Trafilatura
- RAG pipeline with streaming generation

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
export SERPER_API_KEY="your-key"
python app.py

# Run tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=term-missing

# Lint code
ruff check .

# Format code
ruff format .

# Build Docker image
docker build -t web-mcp .
```

## Architecture

```
app.py          → Main Gradio app, MCP tools (search, fetch), health check
rag.py          → RAG pipeline: chunking, embedding, FAISS, generation
reranker.py     → Cross-encoder reranking
analytics.py    → Request counting and dashboard data
config.py       → Environment-based configuration
benchmark.py    → Performance evaluation script
```

## Key Patterns

### Configuration
All config via environment variables with defaults in `config.py`.

### Async/Await
All I/O operations are async. Use `asyncio.to_thread()` for CPU-bound work.

### Error Handling
Return `{"error": "message"}` dicts from tool functions, don't raise.

### Logging
Use `logging` module, not `print()`. Logger per module.

### Testing
Tests in `tests/` directory. Use pytest fixtures from `conftest.py`.

## Important Files

| File | Purpose |
|------|---------|
| `app.py` | Entry point, Gradio UI, MCP tools |
| `rag.py` | RAG pipeline class |
| `config.py` | All configuration |
| `requirements.txt` | Dependencies (version pinned) |
| `Dockerfile` | Production container |
| `.github/workflows/ci.yml` | CI pipeline |

## Environment Variables

Required:
- `SERPER_API_KEY` - Serper.dev API key

Optional:
- `API_AUTH_TOKEN` - Enable bearer auth
- `EMBED_MODEL` - Embedding model
- `GEN_MODEL` - Generator model
- `LOG_LEVEL` - Logging level (DEBUG, INFO, etc.)

## Testing Guidelines

1. Mock external services (Serper API, ML models)
2. Use fixtures from `conftest.py`
3. Test both success and error paths
4. Use `@pytest.mark.asyncio` for async tests

## Common Tasks

### Add a new MCP tool
1. Create async function in `app.py`
2. Return dict with results or `{"error": "..."}`
3. Register with `gr.api(func, api_name="name")`

### Modify RAG pipeline
1. Edit `rag.py`
2. Update tests in `tests/test_rag.py`
3. Run benchmarks: `python benchmark.py`

### Update dependencies
1. Edit `requirements.txt`
2. Update CI matrix if Python version changes
3. Test Docker build

## Code Style

- Ruff for linting and formatting
- Google-style docstrings
- Type hints on public functions
- Max line length: 88 characters
