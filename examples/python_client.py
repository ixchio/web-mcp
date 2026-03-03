#!/usr/bin/env python3
"""
Example Python client for Web MCP Server.

Demonstrates how to use the API programmatically.
"""

import json

import httpx

# Configuration
BASE_URL = "http://localhost:7860"
AUTH_TOKEN = None  # Set if API_AUTH_TOKEN is configured on server


def get_headers():
    """Get request headers with optional auth."""
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    return headers


def search(query: str, search_type: str = "search", num_results: int = 5):
    """
    Search the web using Serper API.

    Args:
        query: Search query string
        search_type: "search" for web, "news" for news
        num_results: Number of results (1-20)

    Returns:
        Search results with metadata
    """
    response = httpx.post(
        f"{BASE_URL}/api/search",
        json={
            "query": query,
            "search_type": search_type,
            "num_results": num_results,
        },
        headers=get_headers(),
        timeout=30,
    )
    return response.json()


def fetch(url: str, timeout: int = 20):
    """
    Fetch and extract content from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Extracted content with metadata
    """
    response = httpx.post(
        f"{BASE_URL}/api/fetch",
        json={
            "url": url,
            "timeout": timeout,
        },
        headers=get_headers(),
        timeout=timeout + 10,
    )
    return response.json()


def health_check():
    """Check server health status."""
    response = httpx.get(
        f"{BASE_URL}/api/health",
        headers=get_headers(),
        timeout=10,
    )
    return response.json()


def main():
    """Run example queries."""
    print("=" * 60)
    print("Web MCP Server - Python Client Example")
    print("=" * 60)

    # Health check
    print("\n📋 Health Check:")
    try:
        health = health_check()
        print(json.dumps(health, indent=2))
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Make sure the server is running at", BASE_URL)
        return

    # Search example
    print("\n🔍 Search Example:")
    results = search("Python async programming", num_results=3)
    print(json.dumps(results, indent=2))

    # Fetch example
    print("\n📥 Fetch Example:")
    content = fetch("https://httpbin.org/html")
    print(f"Title: {content.get('title', 'N/A')}")
    print(f"Word count: {content.get('word_count', 0)}")
    print(f"Content preview: {content.get('content', '')[:200]}...")

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
