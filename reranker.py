"""
Cross-encoder reranker module.

Provides a CrossEncoderReranker class that reranks document candidates
using a cross-encoder model for improved relevance scoring.
"""
import logging
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

__all__ = ["CrossEncoderReranker"]

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker using a cross-encoder model for query-document relevance scoring."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier. Defaults to config.RERANKER_MODEL.

        Raises:
            RuntimeError: If model loading fails.
        """
        model_name = model_name or RERANKER_MODEL
        try:
            self.model = CrossEncoder(model_name)
            logger.info("Loaded reranker model: %s", model_name)
        except Exception as e:
            logger.error("Failed to load reranker model %s: %s", model_name, e)
            raise RuntimeError(f"Failed to load reranker model: {e}") from e

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank a list of documents against a query.

        Args:
            query: The search query.
            documents: List of document strings to rank.
            top_k: Number of top documents to return. Returns all if None.

        Returns:
            List of tuples (document_text, score) sorted by score descending.

        Raises:
            ValueError: If query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        if not documents:
            return []

        # Filter out empty documents
        valid_docs = [doc for doc in documents if doc and isinstance(doc, str)]
        if not valid_docs:
            return []

        pairs = [[query, doc] for doc in valid_docs]
        scores = self.model.predict(pairs)

        # Convert numpy types to Python floats for JSON serialization
        scored_docs: List[Tuple[str, float]] = [
            (doc, float(score)) for doc, score in zip(valid_docs, scores)
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            if top_k <= 0:
                return []
            scored_docs = scored_docs[:top_k]

        return scored_docs
