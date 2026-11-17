"""
Dense vector index with brute-force cosine similarity search.

Vectors are stored as a contiguous normalized matrix for cache-friendly
sequential access. Query time is O(N) via a single matrix-vector dot product,
which is efficient for corpora up to ~100K chunks on modern hardware.

For larger corpora, this would be replaced with an approximate nearest
neighbor index (HNSW, IVF-PQ) — but the interface contract stays identical.
"""

from __future__ import annotations

import numpy as np

from neuroflow.models.schema import DocumentChunk
from neuroflow.utils.logging import get_logger

logger = get_logger(__name__)


class DenseVectorIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._chunks: list[DocumentChunk] = []
        self._matrix: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Build the index from pre-computed embeddings.
        Vectors are L2-normalized at index time for dot-product cosine equivalence.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        if len(embeddings) > 0 and embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) != index dimension ({self.dimension})")

        self._chunks = chunks

        if len(embeddings) == 0:
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)
            return

        # L2-normalize row-wise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        self._matrix = (embeddings / norms).astype(np.float32)

        logger.info(f"Dense index built: {len(chunks)} vectors, dim={self.dimension}")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Brute-force cosine search via matrix-vector dot product.

        Since both matrix rows and query are L2-normalized:
            dot(M[i], q) = cos(M[i], q)

        Returns list of (doc_index, cosine_similarity) sorted descending.
        Cosine similarity values are in [-1, 1]; we remap to [0, 1] for fusion.
        """
        if self.corpus_size == 0:
            return []

        # Normalize query vector
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-10:
            return []
        q_hat = query_vec / q_norm

        # Single BLAS-accelerated dot product
        similarities = self._matrix @ q_hat  # shape: (N,)

        # Remap [-1, 1] -> [0, 1] and clip numerical float32 drift
        normalized = np.clip((similarities + 1.0) * 0.5, 0.0, 1.0)

        top_indices = np.argpartition(normalized, -min(top_k, len(normalized)))[-top_k:]
        top_indices = top_indices[np.argsort(normalized[top_indices])[::-1]]

        return [(int(idx), float(normalized[idx])) for idx in top_indices]
