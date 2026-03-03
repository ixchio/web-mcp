"""
Centralized configuration for Web MCP Server.

All ML model names, tuning knobs, and auth settings are driven by
environment variables with sensible defaults.

Example usage:
    export GEN_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    export API_AUTH_TOKEN="my-secret"
    python app.py

Environment Variables:
    EMBED_MODEL      - Sentence-transformer model for embeddings
    RERANKER_MODEL   - Cross-encoder model for reranking
    GEN_MODEL        - Causal LM for answer generation
    CHUNK_SIZE       - Max words per chunk (default: 200)
    CHUNK_OVERLAP    - Overlap words between chunks (default: 40)
    RAG_TOP_K        - Chunks retrieved from FAISS (default: 6)
    RERANK_TOP_K     - Chunks kept after reranking (default: 3)
    FAISS_CACHE_TTL  - Seconds to cache FAISS index (default: 300)
    API_AUTH_TOKEN   - Bearer token for API auth (disabled if not set)
"""

import os
from typing import Optional

__all__ = [
    "EMBED_MODEL",
    "RERANKER_MODEL",
    "GEN_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RAG_TOP_K",
    "RERANK_TOP_K",
    "FAISS_CACHE_TTL",
    "API_AUTH_TOKEN",
]


def _env(name: str, default: str) -> str:
    """Get string environment variable with default."""
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    """Get integer environment variable with default."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ── ML Models ────────────────────────────────────────────────────────────────
EMBED_MODEL: str = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL: str = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEN_MODEL: str = _env("GEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = _int("CHUNK_SIZE", 200)  # max words per chunk
CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 40)  # overlap words between chunks

# ── Retrieval ────────────────────────────────────────────────────────────────
RAG_TOP_K: int = _int("RAG_TOP_K", 6)  # chunks retrieved from FAISS
RERANK_TOP_K: int = _int("RERANK_TOP_K", 3)  # chunks kept after reranking

# ── FAISS Cache ──────────────────────────────────────────────────────────────
FAISS_CACHE_TTL: int = _int("FAISS_CACHE_TTL", 300)  # seconds

# ── Auth ─────────────────────────────────────────────────────────────────────
API_AUTH_TOKEN: Optional[str] = os.getenv("API_AUTH_TOKEN")  # None = auth disabled
