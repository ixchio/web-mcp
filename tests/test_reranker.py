"""Tests for cross-encoder reranker module."""
import sys
from unittest import mock

import pytest

# Mock sentence_transformers before importing reranker
mock_cross_encoder = mock.MagicMock()
sys.modules["sentence_transformers"] = mock.MagicMock()
sys.modules["sentence_transformers"].CrossEncoder = mock_cross_encoder


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    @pytest.fixture
    def mock_reranker(self):
        """Create a reranker with mocked model."""
        # Reset the mock for each test
        mock_cross_encoder.reset_mock()
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.1, 0.7]
        mock_cross_encoder.return_value = mock_model
        
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        return reranker

    def test_rerank_returns_sorted_results(self, mock_reranker, sample_chunks):
        """Test rerank returns results sorted by score descending."""
        query = "What is the speed of light?"
        results = mock_reranker.rerank(query, sample_chunks)
        
        # Check sorted by score descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_returns_tuples(self, mock_reranker, sample_chunks):
        """Test rerank returns list of (text, score) tuples."""
        query = "test query"
        results = mock_reranker.rerank(query, sample_chunks)
        
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], str)
            assert isinstance(result[1], float)

    def test_rerank_with_top_k(self, mock_reranker, sample_chunks):
        """Test rerank respects top_k parameter."""
        query = "test query"
        results = mock_reranker.rerank(query, sample_chunks, top_k=2)
        
        assert len(results) == 2

    def test_rerank_empty_documents(self, mock_reranker):
        """Test rerank handles empty document list."""
        results = mock_reranker.rerank("query", [])
        assert results == []

    def test_rerank_filters_empty_strings(self, mock_reranker):
        """Test rerank filters out empty strings."""
        with mock.patch.object(mock_reranker.model, "predict", return_value=[0.5, 0.3]):
            docs = ["valid doc", "", "another doc", None, "   "]
            results = mock_reranker.rerank("query", docs)
            
            # Should only have valid docs
            texts = [r[0] for r in results]
            assert "" not in texts
            assert "   " not in texts

    def test_rerank_invalid_query_raises(self, mock_reranker):
        """Test rerank raises ValueError for invalid query."""
        with pytest.raises(ValueError, match="non-empty string"):
            mock_reranker.rerank("", ["doc1", "doc2"])
        
        with pytest.raises(ValueError, match="non-empty string"):
            mock_reranker.rerank("   ", ["doc1", "doc2"])

    def test_rerank_top_k_none_returns_all(self, mock_reranker, sample_chunks):
        """Test rerank with top_k=None returns all results."""
        results = mock_reranker.rerank("query", sample_chunks, top_k=None)
        assert len(results) == len(sample_chunks)

    def test_rerank_top_k_zero_returns_empty(self, mock_reranker, sample_chunks):
        """Test rerank with top_k=0 returns empty list."""
        results = mock_reranker.rerank("query", sample_chunks, top_k=0)
        assert results == []

    def test_rerank_scores_are_python_floats(self, mock_reranker, sample_chunks):
        """Test scores are converted to Python floats (not numpy)."""
        import numpy as np
        
        # Mock returns numpy floats
        with mock.patch.object(
            mock_reranker.model, "predict", 
            return_value=np.array([0.9, 0.3, 0.1, 0.7])
        ):
            results = mock_reranker.rerank("query", sample_chunks)
            
            for _, score in results:
                assert type(score) == float  # Must be Python float, not np.float


class TestCrossEncoderRerankerInit:
    """Tests for CrossEncoderReranker initialization."""

    def test_init_with_default_model(self):
        """Test initialization uses default model from config."""
        mock_cross_encoder.reset_mock()
        from reranker import CrossEncoderReranker
        from config import RERANKER_MODEL
        
        # Force reimport by clearing from sys.modules
        if "reranker" in sys.modules:
            del sys.modules["reranker"]
        
        from reranker import CrossEncoderReranker
        CrossEncoderReranker()
        mock_cross_encoder.assert_called_with(RERANKER_MODEL)

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        mock_cross_encoder.reset_mock()
        
        if "reranker" in sys.modules:
            del sys.modules["reranker"]
            
        from reranker import CrossEncoderReranker
        CrossEncoderReranker(model_name="custom/model")
        mock_cross_encoder.assert_called_with("custom/model")

    def test_init_failure_raises_runtime_error(self):
        """Test initialization failure raises RuntimeError."""
        mock_cross_encoder.reset_mock()
        mock_cross_encoder.side_effect = Exception("Load failed")
        
        if "reranker" in sys.modules:
            del sys.modules["reranker"]
            
        from reranker import CrossEncoderReranker
        
        with pytest.raises(RuntimeError, match="Failed to load reranker"):
            CrossEncoderReranker()
        
        # Reset side_effect for other tests
        mock_cross_encoder.side_effect = None
