"""
Centralized configuration — all ML model names, tuning knobs, and auth
are driven by environment variables with sensible defaults.

Change any setting without touching code:
    export GEN_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    export API_AUTH_TOKEN="my-secret"
    python app.py
"""
import os

def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()

def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

# ── ML Models ────────────────────────────────────────────────────────────────
EMBED_MODEL    = _env("EMBED_MODEL",    "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEN_MODEL      = _env("GEN_MODEL",      "Qwen/Qwen2.5-0.5B-Instruct")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = _int("CHUNK_SIZE", 200)       # max words per chunk
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 40)     # overlap words between chunks

# ── Retrieval ────────────────────────────────────────────────────────────────
RAG_TOP_K     = _int("RAG_TOP_K", 6)         # chunks retrieved from FAISS
RERANK_TOP_K  = _int("RERANK_TOP_K", 3)      # chunks kept after reranking

# ── FAISS Cache ──────────────────────────────────────────────────────────────
FAISS_CACHE_TTL = _int("FAISS_CACHE_TTL", 300)  # seconds

# ── Auth ─────────────────────────────────────────────────────────────────────
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")  # None = auth disabled
