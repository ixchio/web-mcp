from typing import List, Tuple
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder reranker.
        Uses ms-marco-MiniLM-L-6-v2 by default, which is highly efficient and effective.
        """
        self.model = CrossEncoder(model_name)

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
            
        # CrossEncoder expects a list of [query, document] pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Pair documents with their scores
        scored_docs = list(zip(documents, scores))
        
        # Sort by score descending (higher score means more relevant)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
            
        return scored_docs
