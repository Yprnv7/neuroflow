"""Unit tests for the semantic cache."""

import numpy as np
import pytest

from neuroflow.cache.semantic_cache import SemanticCache


class TestSemanticCache:
    def test_miss_on_empty_cache(self):
        cache = SemanticCache(similarity_threshold=0.9)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cache.lookup(vec) is None
        assert cache.stats()["misses"] == 1

    def test_hit_on_identical_vector(self):
        cache = SemanticCache(similarity_threshold=0.9)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cache.store("test query", vec, [])
        result = cache.lookup(vec)
        assert result is not None
        entry, sim = result
        assert entry.query == "test query"
        assert sim >= 0.99

    def test_hit_on_similar_vector(self):
        cache = SemanticCache(similarity_threshold=0.85)
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.95, 0.1, 0.0], dtype=np.float32)  # close to v1
        cache.store("original", v1, [])
        result = cache.lookup(v2)
        assert result is not None

    def test_miss_on_dissimilar_vector(self):
        cache = SemanticCache(similarity_threshold=0.9)
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal
        cache.store("original", v1, [])
        assert cache.lookup(v2) is None

    def test_eviction_at_capacity(self):
        cache = SemanticCache(similarity_threshold=0.99, max_entries=2)
        v1 = np.array([1, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0], dtype=np.float32)
        v3 = np.array([0, 0, 1], dtype=np.float32)

        cache.store("q1", v1, [])
        cache.store("q2", v2, [])
        assert cache.stats()["entries"] == 2

        cache.store("q3", v3, [])
        assert cache.stats()["entries"] == 2  # one was evicted
        assert cache.stats()["evictions"] == 1

    def test_lfu_eviction_preserves_popular(self):
        cache = SemanticCache(similarity_threshold=0.99, max_entries=2)
        v1 = np.array([1, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0], dtype=np.float32)

        cache.store("popular", v1, [])
        # Simulate hits
        cache.lookup(v1)
        cache.lookup(v1)
        cache.lookup(v1)

        cache.store("unpopular", v2, [])

        # Adding a third should evict "unpopular" (fewer hits)
        v3 = np.array([0, 0, 1], dtype=np.float32)
        cache.store("new", v3, [])

        # "popular" should survive
        result = cache.lookup(v1)
        assert result is not None
        assert result[0].query == "popular"

    def test_clear_resets_everything(self):
        cache = SemanticCache()
        v = np.array([1, 0, 0], dtype=np.float32)
        cache.store("q", v, [])
        cache.lookup(v)
        cache.clear()
        s = cache.stats()
        assert s["entries"] == 0
        assert s["hits"] == 0
        assert s["misses"] == 0

    def test_hit_ratio_calculation(self):
        cache = SemanticCache(similarity_threshold=0.99)
        v = np.array([1, 0, 0], dtype=np.float32)
        cache.store("q", v, [])
        cache.lookup(v)  # hit
        cache.lookup(v)  # hit
        cache.lookup(np.array([0, 1, 0], dtype=np.float32))  # miss
        s = cache.stats()
        assert s["hit_ratio"] == pytest.approx(2 / 3, abs=0.01)
