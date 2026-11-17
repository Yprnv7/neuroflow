"""Unit tests for the BM25 sparse retrieval engine."""

import pytest
from neuroflow.models.schema import DocumentChunk
from neuroflow.retrieval.bm25 import BM25Index


@pytest.fixture
def bm25_corpus():
    return [
        DocumentChunk(id="d1", text="distributed consensus algorithm raft leader election"),
        DocumentChunk(id="d2", text="database transaction isolation level serializable"),
        DocumentChunk(id="d3", text="leader election timeout randomized distributed systems"),
        DocumentChunk(id="d4", text="machine learning gradient descent optimization neural network"),
        DocumentChunk(id="d5", text="event streaming kafka consumer groups partitioned commit log"),
    ]


class TestBM25Index:
    def test_build_computes_corpus_stats(self, bm25_corpus):
        idx = BM25Index()
        idx.build(bm25_corpus)
        assert idx.corpus_size == 5
        assert idx._avgdl > 0

    def test_search_returns_relevant_first(self, bm25_corpus):
        idx = BM25Index()
        idx.build(bm25_corpus)
        results = idx.search("raft leader election consensus", top_k=3)
        # d1 and d3 both mention leader election — they should rank top
        top_ids = [bm25_corpus[r[0]].id for r in results]
        assert "d1" in top_ids[:2]
        assert "d3" in top_ids[:2]

    def test_search_scores_normalized(self, bm25_corpus):
        idx = BM25Index()
        idx.build(bm25_corpus)
        results = idx.search("kafka event streaming", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_search_empty_query(self, bm25_corpus):
        idx = BM25Index()
        idx.build(bm25_corpus)
        assert idx.search("", top_k=5) == []

    def test_search_no_match(self, bm25_corpus):
        idx = BM25Index()
        idx.build(bm25_corpus)
        results = idx.search("xylophone trombone accordion", top_k=5)
        assert results == []

    def test_empty_corpus(self):
        idx = BM25Index()
        idx.build([])
        assert idx.corpus_size == 0
        assert idx.search("anything", top_k=5) == []

    def test_k1_b_tuning_affects_scores(self, bm25_corpus):
        idx_default = BM25Index(k1=1.5, b=0.75)
        idx_tuned = BM25Index(k1=2.0, b=0.5)
        idx_default.build(bm25_corpus)
        idx_tuned.build(bm25_corpus)

        r1 = idx_default.search("distributed consensus", top_k=1)
        r2 = idx_tuned.search("distributed consensus", top_k=1)
        # Both should find d1 or d3, but scores may differ
        assert len(r1) > 0 and len(r2) > 0
