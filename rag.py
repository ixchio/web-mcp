import hashlib
import time
import threading
import faiss
import numpy as np
import torch
import nltk
from typing import List, Dict, Any, Tuple, Optional, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_MODEL, GEN_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    RAG_TOP_K, FAISS_CACHE_TTL,
)

# Download sentence tokenizer data (runs once)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Index Cache
# ─────────────────────────────────────────────────────────────────────────────
class _IndexCacheEntry:
    __slots__ = ("index", "chunks", "chunk_sources", "expires_at")

    def __init__(self, index, chunks, chunk_sources, ttl):
        self.index = index
        self.chunks = chunks
        self.chunk_sources = chunk_sources
        self.expires_at = time.time() + ttl


_index_cache: Dict[str, _IndexCacheEntry] = {}


def _docs_cache_key(documents: List[Dict[str, str]]) -> str:
    """Deterministic hash of document URLs to identify identical index sets."""
    urls = sorted(d.get("url", "") for d in documents)
    return hashlib.sha256("|".join(urls).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self,
                 embed_model_name: str = None,
                 gen_model_name: str = None):
        """Initialize models for retrieval and generation.
        Model names are read from config unless overridden."""
        embed_model_name = embed_model_name or EMBED_MODEL
        gen_model_name = gen_model_name or GEN_MODEL

        # 1. Embedding model
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        # State (set by build_index)
        self.index = None
        self.chunks: List[str] = []
        self.chunk_sources: List[str] = []

        # 2. Generator LM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
        )

    # ── Sentence-aware chunking ──────────────────────────────────────────────
    def chunk_text(
        self,
        text: str,
        source_url: str,
        chunk_size: int = None,
        overlap: int = None,
    ) -> Tuple[List[str], List[str]]:
        """Split text into chunks at sentence boundaries.

        Groups sentences until ``chunk_size`` words are reached, then starts
        a new chunk with the last ``overlap`` words of the previous chunk
        prepended for continuity.
        """
        chunk_size = chunk_size or CHUNK_SIZE
        overlap = overlap or CHUNK_OVERLAP

        sentences = nltk.sent_tokenize(text)
        chunks: List[str] = []
        sources: List[str] = []

        current_words: List[str] = []
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
    def build_index(self, documents: List[Dict[str, str]]) -> bool:
        """Chunk documents and create a FAISS index.

        Returns True if a fresh index was built, False if served from cache.
        """
        cache_key = _docs_cache_key(documents)

        # Check cache
        entry = _index_cache.get(cache_key)
        if entry and time.time() < entry.expires_at:
            self.index = entry.index
            self.chunks = entry.chunks
            self.chunk_sources = entry.chunk_sources
            return False  # cache hit

        self.chunks = []
        self.chunk_sources = []

        for doc in documents:
            content = doc.get("content", "")
            url = doc.get("url", "")
            if content.strip():
                c, s = self.chunk_text(content, url)
                self.chunks.extend(c)
                self.chunk_sources.extend(s)

        if not self.chunks:
            return True

        embeddings = self.embed_model.encode(self.chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        # Store in cache
        _index_cache[cache_key] = _IndexCacheEntry(
            self.index, self.chunks, self.chunk_sources, FAISS_CACHE_TTL,
        )
        return True  # fresh build

    # ── Retrieval ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks."""
        top_k = top_k or RAG_TOP_K
        if not self.index or self.index.ntotal == 0:
            return []

        query_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "text": self.chunks[idx],
                    "source": self.chunk_sources[idx],
                    "score": float(distances[0][i]),
                })
        return results

    # ── Generation (blocking) ────────────────────────────────────────────────
    def generate_answer(self, query: str, context: str) -> str:
        """Generate a complete answer (blocking)."""
        tokens = list(self.generate_answer_stream(query, context))
        return "".join(tokens).strip()

    # ── Generation (streaming) ───────────────────────────────────────────────
    def generate_answer_stream(self, query: str, context: str) -> Generator[str, None, None]:
        """Yield answer tokens as they are generated."""
        prompt = (
            "Use the following context to answer the user's question. "
            "If the context does not contain the answer, say "
            "\"I don't have enough information to answer that based on the provided sources.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant answering questions strictly based on the provided context."},
            {"role": "user", "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
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

        for token_text in streamer:
            yield token_text

        thread.join()
