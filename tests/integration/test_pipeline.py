"""Integration tests: full pipeline and API endpoints."""

import pytest
from fastapi.testclient import TestClient

from neuroflow.api.server import app
from neuroflow.core.engine import HybridRAGEngine
from neuroflow.evals.metrics import run_evaluation
from neuroflow.models.schema import DocumentChunk, QueryRequest, SearchMode

# ---- Full pipeline tests ----

class TestHybridEngineIntegration:
    @pytest.fixture
    def engine(self):
        e = HybridRAGEngine()
        e.ingest([
            DocumentChunk(id="ds-1", text="Raft consensus uses leader election and log replication for distributed state machine safety"),
            DocumentChunk(id="ds-2", text="PostgreSQL achieves concurrency via MVCC allowing readers and writers to operate without blocking"),
            DocumentChunk(id="ir-1", text="BM25 is a probabilistic ranking function for information retrieval based on term frequency"),
            DocumentChunk(id="ml-1", text="Neural networks learn representations through backpropagation and gradient descent optimization"),
            DocumentChunk(id="de-1", text="Kafka provides fault-tolerant distributed event streaming with partitioned commit logs"),
        ])
        return e

    def test_hybrid_retrieval_ranks_correctly(self, engine):
        resp = engine.query(QueryRequest(query="distributed consensus leader election", top_k=2, enable_rerank=False))
        assert resp.results[0].chunk.id == "ds-1"
        assert not resp.cached

    def test_dense_only_mode(self, engine):
        resp = engine.query(QueryRequest(query="gradient descent", top_k=2, mode=SearchMode.DENSE_ONLY))
        assert resp.mode == SearchMode.DENSE_ONLY
        assert len(resp.results) > 0

    def test_sparse_only_mode(self, engine):
        resp = engine.query(QueryRequest(query="kafka event streaming", top_k=2, mode=SearchMode.SPARSE_ONLY))
        assert resp.mode == SearchMode.SPARSE_ONLY
        ids = [r.chunk.id for r in resp.results]
        assert "de-1" in ids

    def test_cache_hit_on_repeat_query(self, engine):
        q = QueryRequest(query="Raft distributed consensus", top_k=3)
        r1 = engine.query(q)
        assert not r1.cached
        r2 = engine.query(q)
        assert r2.cached
        assert r2.cache_similarity is not None
        assert r2.latency_ms <= r1.latency_ms

    def test_cache_disabled(self, engine):
        q = QueryRequest(query="something new", top_k=2, enable_cache=False)
        r1 = engine.query(q)
        r2 = engine.query(q)
        assert not r1.cached
        assert not r2.cached

    def test_latency_breakdown_populated(self, engine):
        resp = engine.query(QueryRequest(query="BM25 ranking function"))
        assert "embed_ms" in resp.breakdown_ms
        assert "retrieval_ms" in resp.breakdown_ms
        assert "fusion_ms" in resp.breakdown_ms

    def test_alpha_variation_affects_results(self, engine):
        r_dense = engine.query(QueryRequest(query="leader election", alpha=1.0, enable_rerank=False, enable_cache=False))
        r_sparse = engine.query(QueryRequest(query="leader election", alpha=0.0, enable_rerank=False, enable_cache=False))
        # Different alpha values should produce different score orderings
        s_dense = [r.final_score for r in r_dense.results]
        s_sparse = [r.final_score for r in r_sparse.results]
        assert s_dense != s_sparse

    def test_evaluation_harness(self, engine):
        eval_data = [
            {"query": "distributed consensus raft", "relevant_ids": ["ds-1"]},
            {"query": "kafka event streaming logs", "relevant_ids": ["de-1"]},
        ]
        metrics = run_evaluation(
            eval_data,
            lambda q, k: engine.query(QueryRequest(query=q, top_k=k, enable_cache=False)),
            k=3,
        )
        assert metrics["mrr"] > 0.0
        assert metrics["queries_evaluated"] == 2


# ---- API endpoint tests ----

class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        with TestClient(app) as c:
            yield c

    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["corpus_size"] > 0

    def test_search_endpoint(self, client):
        r = client.post("/api/v1/search", json={
            "query": "How does Raft handle leader election?",
            "top_k": 3,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) > 0
        assert "latency_ms" in data
        assert data["results"][0]["chunk"]["id"] is not None

    def test_search_cache_hit(self, client):
        payload = {"query": "semantic caching for query deduplication", "top_k": 2}
        r1 = client.post("/api/v1/search", json=payload)
        r2 = client.post("/api/v1/search", json=payload)
        assert r1.json()["cached"] is False
        assert r2.json()["cached"] is True

    def test_cache_stats_endpoint(self, client):
        # Fire a query first to populate stats
        client.post("/api/v1/search", json={"query": "stats test"})
        r = client.get("/api/v1/cache/stats")
        assert r.status_code == 200
        assert "hits" in r.json()

    def test_cache_clear_endpoint(self, client):
        r = client.delete("/api/v1/cache")
        assert r.status_code == 200

    def test_ingest_endpoint(self, client):
        r = client.post("/api/v1/ingest", json={
            "documents": [
                {"id": "new-1", "text": "A freshly ingested document about graph algorithms and traversals."}
            ],
        })
        assert r.status_code == 201
        data = r.json()
        assert data["indexed_count"] == 1
        assert data["total_corpus_size"] > 1
