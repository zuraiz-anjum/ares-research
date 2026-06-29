"""ChromaDB-backed vector store for uploaded documents.

Each document is chunked before upload and stored with doc_id metadata so
multiple documents can coexist in the same collection.  The collection is lazy-
initialised on first use and reused across requests (thread-safe in ChromaDB).
"""

import logging
import os

logger = logging.getLogger(__name__)

CHROMA_PATH = "ares_vectorstore"
_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = client.get_or_create_collection(
        name="ares_documents",
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"chroma_collection_ready items={_collection.count()}")
    return _collection


def add_document(doc_id: str, chunks: list[str], filename: str) -> int:
    """Embed and store document chunks. Returns the number of chunks stored."""
    col = _get_collection()
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename} for _ in chunks]
    col.add(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info(f"rag_stored doc_id={doc_id} chunks={len(chunks)}")
    return len(chunks)


_FETCH_K = 20   # candidates fetched from ChromaDB before reranking


def search(doc_id: str, query: str, k: int = 6) -> list[str]:
    """Retrieve-then-rerank: fetch up to 20 candidates by embedding similarity,
    then score with a cross-encoder and return the top-k."""
    col = _get_collection()
    existing = col.get(where={"doc_id": doc_id})
    n = len(existing["ids"])
    if n == 0:
        return []

    fetch_k = min(_FETCH_K, n)
    results = col.query(
        query_texts=[query],
        n_results=fetch_k,
        where={"doc_id": doc_id},
    )
    candidates = results["documents"][0] if results["documents"] else []
    if not candidates:
        return []

    from app.rag.reranker import rerank
    return rerank(query, candidates, k)


def search_multi(doc_ids: list[str], query: str, k: int = 8) -> list[str]:
    """Search across multiple documents and return top-k re-ranked chunks.

    Results from all docs are pooled before re-ranking so the best passages
    from any file float to the top.
    """
    col = _get_collection()
    all_candidates: list[str] = []
    for doc_id in doc_ids:
        existing = col.get(where={"doc_id": doc_id})
        if not existing["ids"]:
            continue
        fetch_k = min(_FETCH_K, len(existing["ids"]))
        results = col.query(
            query_texts=[query],
            n_results=fetch_k,
            where={"doc_id": doc_id},
        )
        candidates = results["documents"][0] if results["documents"] else []
        all_candidates.extend(candidates)

    if not all_candidates:
        return []

    from app.rag.reranker import rerank
    return rerank(query, all_candidates, k)


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to doc_id from the collection."""
    col = _get_collection()
    col.delete(where={"doc_id": doc_id})
    logger.info(f"rag_deleted doc_id={doc_id}")
