"""
Core pipeline orchestrator.

Wires together embedding, sparse index, dense index, fusion, reranking,
and semantic caching into a single query flow with per-stage telemetry.

Pipeline stages:
  1. Embed query
  2. Semantic cache lookup
  3. Parallel sparse (BM25) + dense (vector) retrieval
  4. Score fusion (Convex or RRF)
  5. Cross-encoder reranking
  6. Cache write-back
  7. Return results with latency breakdown
"""

from __future__ import annotations

import time

from neuroflow.cache.semantic_cache import SemanticCache
from neuroflow.config.settings import AppConfig, load_config
from neuroflow.core.embedding import EmbeddingEngine
from neuroflow.models.schema import (
    DocumentChunk,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    SearchMode,
)
from neuroflow.retrieval.bm25 import BM25Index
from neuroflow.retrieval.dense import DenseVectorIndex
from neuroflow.retrieval.fusion import convex_fusion
from neuroflow.retrieval.reranker import CrossEncoderReranker
from neuroflow.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRAGEngine:
    """
    Stateful engine managing the full retrieval pipeline.
    Thread-safe for concurrent reads after indexing completes.
    """

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

        self.embedder = EmbeddingEngine(
            model_name=self.config.embedding.model_name,
            dimension=self.config.embedding.dimension,
            use_local_fallback=self.config.embedding.use_local_fallback,
        )
        self.cache = SemanticCache(
            similarity_threshold=self.config.cache.similarity_threshold,
            max_entries=self.config.cache.max_entries,
        )
        self.bm25 = BM25Index(
            k1=self.config.retrieval.bm25_k1,
            b=self.config.retrieval.bm25_b,
        )
        self.dense = DenseVectorIndex(dimension=self.embedder.dimension)
        self.reranker = CrossEncoderReranker(
            model_name=self.config.reranker.model_name,
            use_local_fallback=self.config.reranker.use_local_fallback,
        )

        self._chunks: list[DocumentChunk] = []

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)

    def ingest(self, chunks: list[DocumentChunk], clear_existing: bool = False) -> int:
        """
        Index documents across both retrieval engines.
        Returns the number of newly indexed chunks.
        """
        if clear_existing:
            self._chunks = []

        self._chunks.extend(chunks)
        texts = [c.text for c in self._chunks]

        # Generate embeddings
        t0 = time.perf_counter()
        embeddings = self.embedder.encode_batch(texts)
        embed_ms = (time.perf_counter() - t0) * 1000

        # Store embeddings on chunks for downstream access
        for chunk, emb in zip(self._chunks, embeddings):
            chunk.embedding = emb.tolist()

        # Build indices
        t1 = time.perf_counter()
        self.bm25.build(self._chunks)
        self.dense.build(self._chunks, embeddings)
        index_ms = (time.perf_counter() - t1) * 1000

        # Invalidate cache on corpus change
        self.cache.clear()

        logger.info(
            f"Ingested {len(chunks)} chunks (total: {len(self._chunks)}). "
            f"Embedding: {embed_ms:.1f}ms, Indexing: {index_ms:.1f}ms"
        )
        return len(chunks)

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute the full hybrid retrieval pipeline."""
        t_start = time.perf_counter()
        breakdown: dict[str, float] = {}

        # --- Stage 1: Embed query ---
        t0 = time.perf_counter()
        q_vec = self.embedder.encode_single(request.query)
        breakdown["embed_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # --- Stage 2: Semantic cache check ---
        if request.enable_cache:
            t0 = time.perf_counter()
            cache_hit = self.cache.lookup(q_vec)
            breakdown["cache_lookup_ms"] = round((time.perf_counter() - t0) * 1000, 3)

            if cache_hit is not None:
                entry, sim = cache_hit
                total_ms = round((time.perf_counter() - t_start) * 1000, 3)
                return QueryResponse(
                    query=request.query,
                    mode=request.mode,
                    results=entry.results[:request.top_k],
                    total_candidates=len(entry.results),
                    cached=True,
                    cache_similarity=round(sim, 4),
                    latency_ms=total_ms,
                    breakdown_ms=breakdown,
                )

        # --- Stage 3: Candidate retrieval ---
        pool_size = max(request.top_k * self.config.retrieval.candidate_pool_multiplier, 15)
        t0 = time.perf_counter()

        dense_hits: list[tuple[int, float]] = []
        sparse_hits: list[tuple[int, float]] = []

        if request.mode in (SearchMode.HYBRID, SearchMode.DENSE_ONLY):
            dense_hits = self.dense.search(q_vec, top_k=pool_size)

        if request.mode in (SearchMode.HYBRID, SearchMode.SPARSE_ONLY):
            sparse_hits = self.bm25.search(request.query, top_k=pool_size)

        breakdown["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # --- Stage 4: Score fusion ---
        t0 = time.perf_counter()
        dense_dict = {idx: score for idx, score in dense_hits}
        sparse_dict = {idx: score for idx, score in sparse_hits}

        if request.mode == SearchMode.DENSE_ONLY:
            fused = dense_dict
        elif request.mode == SearchMode.SPARSE_ONLY:
            fused = sparse_dict
        else:
            fused = convex_fusion(dense_dict, sparse_dict, alpha=request.alpha)

        # Sort by fused score
        fused_ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:pool_size]
        breakdown["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # Build candidate results
        candidates: list[RetrievalResult] = []
        for idx, score in fused_ranked:
            candidates.append(RetrievalResult(
                chunk=self._chunks[idx],
                final_score=round(score, 4),
                dense_score=round(dense_dict.get(idx, 0.0), 4),
                sparse_score=round(sparse_dict.get(idx, 0.0), 4),
            ))

        total_candidates = len(candidates)

        # --- Stage 5: Cross-encoder reranking ---
        if request.enable_rerank and candidates:
            t0 = time.perf_counter()
            rerank_chunks = [c.chunk for c in candidates]
            ranked_pairs = self.reranker.rerank(request.query, rerank_chunks, top_k=request.top_k)

            # Rebuild results with reranker scores while preserving provenance
            candidate_lookup = {c.chunk.id: c for c in candidates}
            reranked: list[RetrievalResult] = []
            for chunk, r_score in ranked_pairs:
                orig = candidate_lookup.get(chunk.id)
                reranked.append(RetrievalResult(
                    chunk=chunk,
                    final_score=round(r_score, 4),
                    dense_score=orig.dense_score if orig else None,
                    sparse_score=orig.sparse_score if orig else None,
                    rerank_score=round(r_score, 4),
                ))

            final_results = reranked
            breakdown["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        else:
            final_results = candidates[:request.top_k]

        # --- Stage 6: Cache write-back ---
        if request.enable_cache:
            self.cache.store(request.query, q_vec, final_results)

        total_ms = round((time.perf_counter() - t_start) * 1000, 3)
        return QueryResponse(
            query=request.query,
            mode=request.mode,
            results=final_results,
            total_candidates=total_candidates,
            cached=False,
            latency_ms=total_ms,
            breakdown_ms=breakdown,
        )
