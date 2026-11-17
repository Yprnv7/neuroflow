"""Unit tests for the dense vector index."""

import numpy as np
import pytest

from neuroflow.models.schema import DocumentChunk
from neuroflow.retrieval.dense import DenseVectorIndex


class TestDenseVectorIndex:
    def test_build_and_search_basic(self):
        idx = DenseVectorIndex(dimension=4)
        chunks = [
            DocumentChunk(id="a", text="alpha"),
            DocumentChunk(id="b", text="beta"),
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        idx.build(chunks, embeddings)

        # Query aligned with 'a'
        results = idx.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert results[0][0] == 0  # chunk 'a' is closest
        assert results[0][1] > results[1][1]

    def test_search_returns_normalized_scores(self):
        idx = DenseVectorIndex(dimension=3)
        chunks = [DocumentChunk(id="x", text="x")]
        embeddings = np.array([[0.6, 0.8, 0.0]], dtype=np.float32)
        idx.build(chunks, embeddings)

        results = idx.search(np.array([0.6, 0.8, 0.0]), top_k=1)
        # Identical vector -> cosine sim = 1.0 -> normalized to 1.0
        assert 0.99 <= results[0][1] <= 1.0

    def test_empty_index_returns_nothing(self):
        idx = DenseVectorIndex(dimension=4)
        idx.build([], np.empty((0, 4), dtype=np.float32))
        assert idx.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5) == []

    def test_zero_query_vector(self):
        idx = DenseVectorIndex(dimension=3)
        chunks = [DocumentChunk(id="a", text="a")]
        idx.build(chunks, np.array([[1, 0, 0]], dtype=np.float32))
        results = idx.search(np.zeros(3, dtype=np.float32), top_k=1)
        assert results == []

    def test_dimension_mismatch_raises(self):
        idx = DenseVectorIndex(dimension=4)
        chunks = [DocumentChunk(id="a", text="a")]
        embeddings = np.array([[1, 0, 0]], dtype=np.float32)  # dim=3 not 4
        with pytest.raises(ValueError):
            idx.build(chunks, embeddings)
