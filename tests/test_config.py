"""Tests for configuration module."""

import os
from unittest import mock


class TestConfigDefaults:
    """Test configuration default values."""

    def test_embed_model_default(self):
        """Test default embedding model."""
        from config import EMBED_MODEL

        assert "MiniLM" in EMBED_MODEL or EMBED_MODEL != ""

    def test_reranker_model_default(self):
        """Test default reranker model."""
        from config import RERANKER_MODEL

        assert "ms-marco" in RERANKER_MODEL or RERANKER_MODEL != ""

    def test_gen_model_default(self):
        """Test default generation model."""
        from config import GEN_MODEL

        assert GEN_MODEL != ""

    def test_chunk_size_default(self):
        """Test default chunk size is reasonable."""
        from config import CHUNK_SIZE

        assert 50 <= CHUNK_SIZE <= 1000

    def test_chunk_overlap_default(self):
        """Test default chunk overlap is reasonable."""
        from config import CHUNK_OVERLAP

        assert 0 <= CHUNK_OVERLAP <= 200

    def test_rag_top_k_default(self):
        """Test default RAG top_k is reasonable."""
        from config import RAG_TOP_K

        assert 1 <= RAG_TOP_K <= 50

    def test_rerank_top_k_default(self):
        """Test default rerank top_k is reasonable."""
        from config import RERANK_TOP_K

        assert 1 <= RERANK_TOP_K <= 20

    def test_faiss_cache_ttl_default(self):
        """Test default FAISS cache TTL is reasonable."""
        from config import FAISS_CACHE_TTL

        assert 0 <= FAISS_CACHE_TTL <= 3600


class TestConfigEnvOverride:
    """Test configuration environment variable overrides."""

    def test_env_helper_strips_whitespace(self):
        """Test _env helper strips whitespace."""
        from config import _env

        with mock.patch.dict(os.environ, {"TEST_VAR": "  value  "}):
            assert _env("TEST_VAR", "default") == "value"

    def test_env_helper_returns_default(self):
        """Test _env helper returns default for missing var."""
        from config import _env

        assert _env("NONEXISTENT_VAR_12345", "default") == "default"

    def test_int_helper_parses_valid_int(self):
        """Test _int helper parses valid integer."""
        from config import _int

        with mock.patch.dict(os.environ, {"TEST_INT": "42"}):
            assert _int("TEST_INT", 0) == 42

    def test_int_helper_returns_default_for_invalid(self):
        """Test _int helper returns default for invalid value."""
        from config import _int

        with mock.patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert _int("TEST_INT", 99) == 99

    def test_int_helper_returns_default_for_missing(self):
        """Test _int helper returns default for missing var."""
        from config import _int

        assert _int("NONEXISTENT_INT_12345", 123) == 123


class TestConfigExports:
    """Test configuration exports."""

    def test_all_exports_defined(self):
        """Test __all__ contains expected exports."""
        from config import __all__

        expected = [
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
        for item in expected:
            assert item in __all__
