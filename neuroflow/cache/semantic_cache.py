"""
In-memory semantic cache using vectorized cosine similarity.

Cache entries store the query embedding alongside the retrieval results.
On a new query, the cache performs a matrix-vector dot product against
all stored query embeddings. If the best match exceeds the similarity
threshold, the cached results are returned directly — bypassing the
entire retrieval and reranking pipeline.

Eviction policy: LFU (Least Frequently Used) with age tiebreaker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from neuroflow.models.schema import RetrievalResult
from neuroflow.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    query: str
    query_vector: np.ndarray
    results: list[RetrievalResult]
    created_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.monotonic)


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92, max_entries: int = 2048):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []
        self._matrix: np.ndarray | None = None  # stacked query vectors for batch comparison
        self._dirty: bool = True  # flag to rebuild matrix after mutations

        # Observability counters
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def _rebuild_matrix(self) -> None:
        """Reconstruct the comparison matrix from current entries."""
        if self._entries:
            self._matrix = np.vstack([e.query_vector for e in self._entries])
        else:
            self._matrix = None
        self._dirty = False

    def lookup(self, query_vec: np.ndarray) -> tuple[CacheEntry, float] | None:
        """
        Check cache for a semantically similar query.

        Performs vectorized cosine similarity via dot product (vectors are
        pre-normalized). Returns (entry, similarity) or None on miss.
        """
        if not self._entries:
            self._misses += 1
            return None

        if self._dirty:
            self._rebuild_matrix()

        if self._matrix is None:
            self._misses += 1
            return None

        # Normalize query
        qn = np.linalg.norm(query_vec)
        if qn < 1e-10:
            self._misses += 1
            return None
        q_hat = query_vec / qn

        # Batch cosine similarity
        sims = self._matrix @ q_hat
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            entry = self._entries[best_idx]
            entry.hit_count += 1
            entry.last_accessed = time.monotonic()
            self._hits += 1
            return entry, best_sim

        self._misses += 1
        return None

    def store(self, query: str, query_vec: np.ndarray, results: list[RetrievalResult]) -> None:
        """Add a new entry to the cache, evicting if at capacity."""
        # Normalize before storage
        qn = np.linalg.norm(query_vec)
        if qn > 1e-10:
            query_vec = query_vec / qn

        if len(self._entries) >= self.max_entries:
            self._evict()

        self._entries.append(CacheEntry(
            query=query,
            query_vector=query_vec,
            results=results,
        ))
        self._dirty = True

    def _evict(self) -> None:
        """Remove the least valuable entry (lowest hits, then oldest)."""
        if not self._entries:
            return
        # Sort by (hit_count ASC, created_at ASC) and remove the first
        victim_idx = min(
            range(len(self._entries)),
            key=lambda i: (self._entries[i].hit_count, self._entries[i].created_at),
        )
        self._entries.pop(victim_idx)
        self._evictions += 1
        self._dirty = True

    def clear(self) -> None:
        self._entries.clear()
        self._matrix = None
        self._dirty = True
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / total, 4) if total > 0 else 0.0,
            "evictions": self._evictions,
            "threshold": self.threshold,
        }
