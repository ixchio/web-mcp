from typing import List, Tuple
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL


class CrossEncoderReranker:
    def __init__(self, model_name: str = None):
        """Initialize the cross-encoder reranker.
        Model is read from config.RERANKER_MODEL unless overridden.
        """
        self.model = CrossEncoder(model_name or RERANKER_MODEL)

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """Rerank a list of documents against a query.

        Args:
            query: The search query.
            documents: List of document strings to rank.
            top_k: Optional number of top documents to return. Returns all if None.

        Returns:
            List of tuples (document_text, score) sorted by score descending.
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        return scored_docs
