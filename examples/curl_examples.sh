#!/bin/bash
# =============================================================================
# Web MCP Server - cURL Examples
# =============================================================================
# Usage: ./curl_examples.sh
# =============================================================================

BASE_URL="${WEB_MCP_URL:-http://localhost:7860}"
AUTH_TOKEN="${WEB_MCP_TOKEN:-}"

# Helper function for authenticated requests
curl_auth() {
    if [ -n "$AUTH_TOKEN" ]; then
        curl -H "Authorization: Bearer $AUTH_TOKEN" "$@"
    else
        curl "$@"
    fi
}

echo "=============================================="
echo "Web MCP Server - cURL Examples"
echo "Base URL: $BASE_URL"
echo "=============================================="

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
echo -e "\n📋 Health Check:"
curl_auth -s "$BASE_URL/api/health" | python3 -m json.tool

# -----------------------------------------------------------------------------
# Search - Web
# -----------------------------------------------------------------------------
echo -e "\n\n🔍 Search (Web):"
curl_auth -s -X POST "$BASE_URL/api/search" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "Model Context Protocol MCP",
        "search_type": "search",
        "num_results": 3
    }' | python3 -m json.tool

# -----------------------------------------------------------------------------
# Search - News
# -----------------------------------------------------------------------------
echo -e "\n\n📰 Search (News):"
curl_auth -s -X POST "$BASE_URL/api/search" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "AI developments 2024",
        "search_type": "news",
        "num_results": 3
    }' | python3 -m json.tool

# -----------------------------------------------------------------------------
# Fetch URL
# -----------------------------------------------------------------------------
echo -e "\n\n📥 Fetch URL:"
curl_auth -s -X POST "$BASE_URL/api/fetch" \
    -H "Content-Type: application/json" \
    -d '{
        "url": "https://httpbin.org/html",
        "timeout": 15
    }' | python3 -m json.tool

# -----------------------------------------------------------------------------
# MCP SSE Endpoint Info
# -----------------------------------------------------------------------------
echo -e "\n\n🔌 MCP SSE Endpoint:"
echo "The MCP Server-Sent Events endpoint is available at:"
echo "$BASE_URL/gradio_api/mcp/sse"
echo ""
echo "Use this URL in MCP-compatible clients like Claude Desktop or Cursor."

echo -e "\n\n✅ All examples completed!"
