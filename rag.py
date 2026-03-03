"""
RAG (Retrieval-Augmented Generation) pipeline module.

Provides the RAGPipeline class for document embedding, FAISS indexing,
retrieval, and streaming LLM generation.
"""

import hashlib
import logging
import threading
import time
from collections.abc import Generator
from typing import Any, Optional

import faiss
import nltk
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    FAISS_CACHE_TTL,
    GEN_MODEL,
    RAG_TOP_K,
)

__all__ = ["RAGPipeline"]

logger = logging.getLogger(__name__)

# Download sentence tokenizer data (runs once)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Index Cache (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
class _IndexCacheEntry:
    """Cache entry holding FAISS index and associated metadata."""

    __slots__ = ("index", "chunks", "chunk_sources", "expires_at")

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[str],
        chunk_sources: list[str],
        ttl: int,
    ):
        self.index = index
        self.chunks = chunks
        self.chunk_sources = chunk_sources
        self.expires_at = time.time() + ttl


_index_cache: dict[str, _IndexCacheEntry] = {}
_index_cache_lock = threading.Lock()


def _docs_cache_key(documents: list[dict[str, str]]) -> str:
    """Deterministic hash of document URLs to identify identical index sets."""
    urls = sorted(d.get("url", "") for d in documents)
    return hashlib.sha256("|".join(urls).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class RAGPipeline:
    """RAG pipeline for document retrieval and answer generation."""

    def __init__(
        self,
        embed_model_name: Optional[str] = None,
        gen_model_name: Optional[str] = None,
    ):
        """Initialize models for retrieval and generation.

        Args:
            embed_model_name: Sentence-transformer model for embeddings.
                              Defaults to config.EMBED_MODEL.
            gen_model_name: Causal LM model for answer generation.
                            Defaults to config.GEN_MODEL.

        Raises:
            RuntimeError: If model loading fails.
        """
        embed_model_name = embed_model_name or EMBED_MODEL
        gen_model_name = gen_model_name or GEN_MODEL

        try:
            # 1. Embedding model
            self.embed_model = SentenceTransformer(embed_model_name)
            self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
            logger.info(
                "Loaded embedding model: %s (dim=%d)",
                embed_model_name,
                self.embedding_dim,
            )
        except Exception as e:
            logger.error("Failed to load embedding model %s: %s", embed_model_name, e)
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

        # State (set by build_index)
        self.index: Optional[faiss.Index] = None
        self.chunks: list[str] = []
        self.chunk_sources: list[str] = []

        try:
            # 2. Generator LM
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
            self.generator = AutoModelForCausalLM.from_pretrained(
                gen_model_name,
                torch_dtype=torch.float32,
                device_map=self.device,
            )
            logger.info(
                "Loaded generator model: %s (device=%s)", gen_model_name, self.device
            )
        except Exception as e:
            logger.error("Failed to load generator model %s: %s", gen_model_name, e)
            raise RuntimeError(f"Failed to load generator model: {e}") from e

    # ── Sentence-aware chunking ──────────────────────────────────────────────
    def chunk_text(
        self,
        text: str,
        source_url: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> tuple[list[str], list[str]]:
        """Split text into chunks at sentence boundaries.

        Groups sentences until ``chunk_size`` words are reached, then starts
        a new chunk with the last ``overlap`` words of the previous chunk
        prepended for continuity.

        Args:
            text: The text to chunk.
            source_url: URL to associate with each chunk.
            chunk_size: Max words per chunk. Defaults to config.CHUNK_SIZE.
            overlap: Overlap words between chunks. Defaults to config.CHUNK_OVERLAP.

        Returns:
            Tuple of (chunks, sources) lists.
        """
        chunk_size = chunk_size or CHUNK_SIZE
        overlap = overlap or CHUNK_OVERLAP

        sentences = nltk.sent_tokenize(text)
        chunks: list[str] = []
        sources: list[str] = []

        current_words: list[str] = []
        for sent in sentences:
            words = sent.split()
            # If adding this sentence overflows, flush the current chunk
            if current_words and len(current_words) + len(words) > chunk_size:
                chunks.append(" ".join(current_words))
                sources.append(source_url)
                # Keep last `overlap` words for context continuity
                current_words = current_words[-overlap:] if overlap else []
            current_words.extend(words)

        # Final leftover chunk
        if current_words:
            chunks.append(" ".join(current_words))
            sources.append(source_url)

        return chunks, sources

    # ── Index building (with TTL cache) ──────────────────────────────────────
    def build_index(self, documents: list[dict[str, str]]) -> bool:
        """Chunk documents and create a FAISS index.

        Uses a thread-safe TTL cache to avoid redundant embedding on repeated
        queries with the same document set.

        Args:
            documents: List of dicts with 'url' and 'content' keys.

        Returns:
            True if a fresh index was built, False if served from cache.
        """
        cache_key = _docs_cache_key(documents)

        # Thread-safe cache check
        with _index_cache_lock:
            entry = _index_cache.get(cache_key)
            if entry and time.time() < entry.expires_at:
                self.index = entry.index
                self.chunks = list(entry.chunks)  # Copy to avoid mutation
                self.chunk_sources = list(entry.chunk_sources)
                logger.debug("Cache hit for index key %s", cache_key[:12])
                return False  # cache hit

        # Build index outside the lock (expensive operation)
        chunks: list[str] = []
        chunk_sources: list[str] = []

        for doc in documents:
            content = doc.get("content", "")
            url = doc.get("url", "")
            if content.strip():
                c, s = self.chunk_text(content, url)
                chunks.extend(c)
                chunk_sources.extend(s)

        if not chunks:
            self.index = None
            self.chunks = []
            self.chunk_sources = []
            return True

        embeddings = self.embed_model.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)

        # Thread-safe cache update
        with _index_cache_lock:
            _index_cache[cache_key] = _IndexCacheEntry(
                index,
                chunks,
                chunk_sources,
                FAISS_CACHE_TTL,
            )

        self.index = index
        self.chunks = chunks
        self.chunk_sources = chunk_sources
        logger.debug("Built fresh index with %d chunks", len(chunks))
        return True  # fresh build

    # ── Retrieval ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        """Retrieve the top_k most similar chunks.

        Args:
            query: Search query string.
            top_k: Number of chunks to retrieve. Defaults to config.RAG_TOP_K.

        Returns:
            List of dicts with 'text', 'source', and 'score' keys.
        """
        top_k = top_k or RAG_TOP_K
        if not self.index or self.index.ntotal == 0:
            return []

        query_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))

        results: list[dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append(
                    {
                        "text": self.chunks[idx],
                        "source": self.chunk_sources[idx],
                        "score": float(distances[0][i]),
                    }
                )
        return results

    # ── Generation (blocking) ────────────────────────────────────────────────
    def generate_answer(self, query: str, context: str) -> str:
        """Generate a complete answer (blocking).

        Args:
            query: User's question.
            context: Retrieved context to ground the answer.

        Returns:
            Generated answer string.
        """
        tokens = list(self.generate_answer_stream(query, context))
        return "".join(tokens).strip()

    # ── Generation (streaming) ───────────────────────────────────────────────
    def generate_answer_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        """Yield answer tokens as they are generated.

        Args:
            query: User's question.
            context: Retrieved context to ground the answer.

        Yields:
            Generated tokens as strings.
        """
        prompt = (
            "Use the following context to answer the user's question. "
            "If the context does not contain the answer, say "
            '"I don\'t have enough information to answer that based on the provided sources."\n\n'
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions strictly based on the provided context.",
            },
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in a background thread so we can iterate the streamer
        thread = threading.Thread(target=lambda: self.generator.generate(**gen_kwargs))
        thread.start()

        yield from streamer

        thread.join()
