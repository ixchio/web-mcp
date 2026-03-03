# Examples

This directory contains example code for interacting with the Web MCP Server.

## Python Client

```bash
# Install httpx if needed
pip install httpx

# Run the example
python python_client.py
```

See [`python_client.py`](python_client.py) for a complete example.

## cURL Examples

```bash
# Make executable
chmod +x curl_examples.sh

# Run examples
./curl_examples.sh

# With custom URL
WEB_MCP_URL=https://your-server.com ./curl_examples.sh

# With authentication
WEB_MCP_TOKEN=your-token ./curl_examples.sh
```

See [`curl_examples.sh`](curl_examples.sh) for all API endpoints.

## MCP Client Integration

### Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "web-mcp": {
      "url": "http://localhost:7860/gradio_api/mcp/sse"
    }
  }
}
```

### Cursor

Add to Cursor settings:

```json
{
  "mcp.servers": {
    "web-mcp": {
      "url": "http://localhost:7860/gradio_api/mcp/sse"
    }
  }
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/search` | POST | Web/news search |
| `/api/fetch` | POST | URL content extraction |
| `/api/ask_rag` | POST | RAG pipeline (streaming) |
| `/gradio_api/mcp/sse` | SSE | MCP protocol endpoint |
