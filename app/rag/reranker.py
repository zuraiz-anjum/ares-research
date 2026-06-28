"""Cross-encoder reranker for RAG.

Uses FlashRank (ms-marco-MiniLM-L-12-v2) to score candidate chunks against
the user's query.  The model is ~4 MB and loads once into a module-level
singleton; subsequent calls are fast (pure ONNX inference, no GPU needed).

Usage pattern (retrieve-then-rerank):
  candidates = chromadb.query(top_n=20)   # fast approximate ANN search
  top_chunks  = rerank(query, candidates, k=6)  # slow but accurate scoring
"""

import logging

logger = logging.getLogger(__name__)

_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache")
        logger.info("reranker_loaded model=ms-marco-MiniLM-L-12-v2")
    return _ranker


def rerank(query: str, chunks: list[str], k: int) -> list[str]:
    """Score chunks against query and return the top-k by relevance score."""
    if not chunks:
        return []
    try:
        from flashrank import RerankRequest
        ranker = _get_ranker()
        passages = [{"id": i, "text": c} for i, c in enumerate(chunks)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        top = [r["text"] for r in results[:k]]
        logger.info(f"rerank candidates={len(chunks)} returned={len(top)}")
        return top
    except Exception:
        logger.warning("rerank_failed falling back to embedding order", exc_info=True)
        return chunks[:k]
