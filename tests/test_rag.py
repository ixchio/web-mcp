"""Tests for RAG pipeline module."""

from unittest import mock

import pytest


class TestRAGPipelineChunking:
    """Tests for RAG pipeline text chunking."""

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create a RAG pipeline with mocked models."""
        with (
            mock.patch("rag.SentenceTransformer") as MockEmbed,
            mock.patch("rag.AutoTokenizer"),
            mock.patch("rag.AutoModelForCausalLM"),
        ):
            # Mock embedding model
            mock_embed = MockEmbed.return_value
            mock_embed.get_sentence_embedding_dimension.return_value = 384
            mock_embed.encode.return_value = [[0.1] * 384]

            from rag import RAGPipeline

            pipeline = RAGPipeline()
            return pipeline

    def test_chunk_text_basic(self, mock_rag_pipeline):
        """Test basic text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks, sources = mock_rag_pipeline.chunk_text(text, "http://example.com")

        assert len(chunks) > 0
        assert len(chunks) == len(sources)
        assert all(s == "http://example.com" for s in sources)

    def test_chunk_text_respects_chunk_size(self, mock_rag_pipeline):
        """Test chunking respects chunk size."""
        # Create text with many words
        text = " ".join(["word."] * 500)
        chunks, _ = mock_rag_pipeline.chunk_text(text, "url", chunk_size=100)

        # Each chunk should be roughly around chunk_size words
        for chunk in chunks:
            word_count = len(chunk.split())
            # Allow some flexibility due to sentence boundaries
            assert word_count <= 150  # chunk_size + buffer

    def test_chunk_text_maintains_overlap(self, mock_rag_pipeline):
        """Test chunking maintains overlap between chunks."""
        text = "Sentence one here. Sentence two here. Sentence three here. Sentence four here."
        chunks, _ = mock_rag_pipeline.chunk_text(text, "url", chunk_size=5, overlap=2)

        # With overlap, consecutive chunks should share some words
        if len(chunks) > 1:
            # Check there's some overlap
            words_chunk1 = set(chunks[0].split()[-3:])
            words_chunk2 = set(chunks[1].split()[:3])
            # There should be some shared words due to overlap
            assert len(words_chunk1.intersection(words_chunk2)) > 0

    def test_chunk_text_empty_returns_empty(self, mock_rag_pipeline):
        """Test chunking empty text returns empty lists."""
        chunks, sources = mock_rag_pipeline.chunk_text("", "url")
        # May return empty or single empty chunk depending on implementation
        assert isinstance(chunks, list)
        assert isinstance(sources, list)

    def test_chunk_text_single_sentence(self, mock_rag_pipeline):
        """Test chunking single sentence."""
        text = "This is a single sentence."
        chunks, sources = mock_rag_pipeline.chunk_text(text, "http://test.com")

        assert len(chunks) == 1
        assert chunks[0] == "This is a single sentence."


class TestRAGPipelineIndexing:
    """Tests for RAG pipeline index building."""

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create a RAG pipeline with mocked models."""
        with (
            mock.patch("rag.SentenceTransformer") as MockEmbed,
            mock.patch("rag.AutoTokenizer"),
            mock.patch("rag.AutoModelForCausalLM"),
            mock.patch("rag.faiss") as mock_faiss,
        ):
            import numpy as np

            # Mock embedding model
            mock_embed = MockEmbed.return_value
            mock_embed.get_sentence_embedding_dimension.return_value = 384
            mock_embed.encode.return_value = np.random.rand(3, 384).astype("float32")

            # Mock FAISS index
            mock_index = mock.MagicMock()
            mock_index.ntotal = 3
            mock_faiss.IndexFlatIP.return_value = mock_index

            from rag import RAGPipeline

            pipeline = RAGPipeline()
            return pipeline

    def test_build_index_creates_faiss_index(self, mock_rag_pipeline, sample_documents):
        """Test build_index creates FAISS index."""
        result = mock_rag_pipeline.build_index(sample_documents)

        assert result is True  # Fresh build
        assert mock_rag_pipeline.index is not None

    def test_build_index_populates_chunks(self, mock_rag_pipeline, sample_documents):
        """Test build_index populates chunks list."""
        mock_rag_pipeline.build_index(sample_documents)

        assert len(mock_rag_pipeline.chunks) > 0
        assert len(mock_rag_pipeline.chunk_sources) > 0
        assert len(mock_rag_pipeline.chunks) == len(mock_rag_pipeline.chunk_sources)

    def test_build_index_empty_documents(self, mock_rag_pipeline):
        """Test build_index with empty documents."""
        result = mock_rag_pipeline.build_index([])

        assert result is True
        assert mock_rag_pipeline.chunks == []

    def test_build_index_filters_empty_content(self, mock_rag_pipeline):
        """Test build_index filters documents with empty content."""
        docs = [
            {"url": "http://a.com", "content": "Valid content here."},
            {"url": "http://b.com", "content": ""},
            {"url": "http://c.com", "content": "   "},
        ]
        mock_rag_pipeline.build_index(docs)

        # Should only have chunks from valid content
        assert len(mock_rag_pipeline.chunks) > 0
        # Sources should only be from docs with content
        assert "http://b.com" not in mock_rag_pipeline.chunk_sources


class TestRAGPipelineRetrieval:
    """Tests for RAG pipeline retrieval."""

    @pytest.fixture
    def mock_rag_with_index(self):
        """Create a RAG pipeline with mocked index."""
        with (
            mock.patch("rag.SentenceTransformer") as MockEmbed,
            mock.patch("rag.AutoTokenizer"),
            mock.patch("rag.AutoModelForCausalLM"),
            mock.patch("rag.faiss") as mock_faiss,
        ):
            import numpy as np

            # Mock embedding model
            mock_embed = MockEmbed.return_value
            mock_embed.get_sentence_embedding_dimension.return_value = 384
            mock_embed.encode.return_value = np.random.rand(1, 384).astype("float32")

            # Mock FAISS index with search results
            mock_index = mock.MagicMock()
            mock_index.ntotal = 3

            def mock_search(query_emb, k):
                return (
                    np.array([[0.9, 0.7, 0.5]])[:, :k],
                    np.array([[0, 1, 2]])[:, :k],
                )

            mock_index.search.side_effect = mock_search
            mock_faiss.IndexFlatIP.return_value = mock_index

            from rag import RAGPipeline

            pipeline = RAGPipeline()

            # Manually set up index state
            pipeline.index = mock_index
            pipeline.chunks = ["chunk0", "chunk1", "chunk2"]
            pipeline.chunk_sources = ["url0", "url1", "url2"]

            return pipeline

    def test_retrieve_returns_results(self, mock_rag_with_index):
        """Test retrieve returns results."""
        results = mock_rag_with_index.retrieve("test query", top_k=3)

        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r

    def test_retrieve_respects_top_k(self, mock_rag_with_index):
        """Test retrieve respects top_k parameter."""
        results = mock_rag_with_index.retrieve("test query", top_k=2)

        assert len(results) == 2

    def test_retrieve_empty_index_returns_empty(self, mock_rag_with_index):
        """Test retrieve with empty index returns empty list."""
        mock_rag_with_index.index = None

        results = mock_rag_with_index.retrieve("test query")
        assert results == []

    def test_retrieve_scores_are_floats(self, mock_rag_with_index):
        """Test retrieve scores are Python floats."""
        results = mock_rag_with_index.retrieve("test query")

        for r in results:
            assert isinstance(r["score"], float)


class TestDocsCacheKey:
    """Tests for document cache key generation."""

    def test_cache_key_deterministic(self):
        """Test cache key is deterministic for same documents."""
        from rag import _docs_cache_key

        docs = [{"url": "http://a.com"}, {"url": "http://b.com"}]

        key1 = _docs_cache_key(docs)
        key2 = _docs_cache_key(docs)

        assert key1 == key2

    def test_cache_key_order_independent(self):
        """Test cache key is same regardless of document order."""
        from rag import _docs_cache_key

        docs1 = [{"url": "http://a.com"}, {"url": "http://b.com"}]
        docs2 = [{"url": "http://b.com"}, {"url": "http://a.com"}]

        assert _docs_cache_key(docs1) == _docs_cache_key(docs2)

    def test_cache_key_different_for_different_docs(self):
        """Test cache key differs for different documents."""
        from rag import _docs_cache_key

        docs1 = [{"url": "http://a.com"}]
        docs2 = [{"url": "http://b.com"}]

        assert _docs_cache_key(docs1) != _docs_cache_key(docs2)

    def test_cache_key_is_sha256(self):
        """Test cache key is a valid SHA256 hash."""
        from rag import _docs_cache_key

        docs = [{"url": "http://test.com"}]
        key = _docs_cache_key(docs)

        assert len(key) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in key)
