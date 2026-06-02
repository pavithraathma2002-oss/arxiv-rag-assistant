"""
retriever.py — Hybrid search: BM25 (sparse) + ChromaDB (dense) + cross-encoder reranking.
"""

from typing import Optional
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PATH = "./chroma_db"

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def hybrid_search(
    query: str,
    collection_name: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    return_scores: bool = False,
) -> list[dict]:
    """
    Hybrid retrieval pipeline:
    1. Dense retrieval via ChromaDB (cosine similarity)
    2. BM25 sparse retrieval over same corpus
    3. Reciprocal Rank Fusion to merge scores
    4. Cross-encoder reranking (optional)
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(collection_name)

    # Fetch all documents for BM25
    all_docs = collection.get(include=["documents", "metadatas"])
    all_texts = all_docs["documents"]
    all_ids = all_docs["ids"]
    all_meta = all_docs["metadatas"]

    # ── 1. Dense retrieval ──────────────────────────────────────────────────
    embedder = get_embedder()
    query_embedding = embedder.encode(query).tolist()

    fetch_k = min(top_k * 3, len(all_texts))
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    dense_ids = dense_results["ids"][0]
    dense_distances = dense_results["distances"][0]
    # Convert cosine distance to similarity score (0-1)
    dense_scores = {id_: 1 - dist for id_, dist in zip(dense_ids, dense_distances)}

    if not use_hybrid:
        # Dense only
        candidates = []
        for id_, doc, meta in zip(
            dense_results["ids"][0],
            dense_results["documents"][0],
            dense_results["metadatas"][0],
        ):
            candidates.append({
                "id": id_,
                "text": doc,
                "page": meta.get("page", "?"),
                "dense_score": dense_scores.get(id_, 0),
                "bm25_score": 0,
                "rerank_score": 0,
            })
        candidates = candidates[:top_k]

    else:
        # ── 2. BM25 sparse retrieval ────────────────────────────────────────
        tokenized_corpus = [doc.lower().split() for doc in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores_raw = bm25.get_scores(query.lower().split())

        # Map id -> bm25 score
        bm25_id_scores = {}
        for i, id_ in enumerate(all_ids):
            bm25_id_scores[id_] = float(bm25_scores_raw[i])

        # Normalize BM25 scores to 0-1
        max_bm25 = max(bm25_id_scores.values()) or 1
        bm25_id_scores = {k: v / max_bm25 for k, v in bm25_id_scores.items()}

        # ── 3. Reciprocal Rank Fusion ───────────────────────────────────────
        # Get top BM25 candidates
        bm25_top_ids = sorted(bm25_id_scores, key=bm25_id_scores.get, reverse=True)[:fetch_k]

        # Union of dense + BM25 candidates
        candidate_ids = list(set(dense_ids) | set(bm25_top_ids))

        def rrf_score(rank, k=60):
            return 1 / (k + rank)

        dense_rank = {id_: i for i, id_ in enumerate(dense_ids)}
        bm25_rank = {id_: i for i, id_ in enumerate(bm25_top_ids)}

        fused = {}
        for id_ in candidate_ids:
            dr = dense_rank.get(id_, fetch_k)
            br = bm25_rank.get(id_, fetch_k)
            fused[id_] = rrf_score(dr) + rrf_score(br)

        # Sort by fused score
        sorted_ids = sorted(fused, key=fused.get, reverse=True)[:top_k * 2]

        # Build candidate list
        id_to_text = dict(zip(all_ids, all_texts))
        id_to_meta = dict(zip(all_ids, all_meta))

        candidates = []
        for id_ in sorted_ids:
            if id_ in id_to_text:
                candidates.append({
                    "id": id_,
                    "text": id_to_text[id_],
                    "page": id_to_meta[id_].get("page", "?"),
                    "dense_score": dense_scores.get(id_, 0),
                    "bm25_score": bm25_id_scores.get(id_, 0),
                    "rerank_score": 0,
                    "fused_score": fused.get(id_, 0),
                })

    # ── 4. Cross-encoder reranking ──────────────────────────────────────────
    if use_reranker and candidates:
        reranker = get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs).tolist()

        for c, score in zip(candidates, rerank_scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:top_k]
