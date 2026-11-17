"""FastAPI application and REST endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from neuroflow.config.settings import load_config
from neuroflow.core.engine import HybridRAGEngine
from neuroflow.models.schema import (
    CacheStats,
    DocumentChunk,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

config = load_config()
engine = HybridRAGEngine(config)


# --- Sample corpus seeded at startup for demo / health verification ---
_SEED_CORPUS = [
    DocumentChunk(
        id="raft-consensus",
        text="The Raft consensus algorithm divides time into terms. Each term begins with a leader election. "
             "Candidates request votes with randomized timeouts to prevent split votes. The elected leader "
             "replicates log entries to followers and commits once a majority acknowledges. This ensures "
             "linearizable reads and state machine safety even under network partitions.",
        metadata={"domain": "distributed_systems", "topic": "consensus"},
    ),
    DocumentChunk(
        id="hnsw-vector-search",
        text="Hierarchical Navigable Small World (HNSW) graphs enable approximate nearest neighbor search "
             "in logarithmic time. Nodes are inserted into multiple layers, each sparser than the one below. "
             "Search begins at the topmost layer and greedily descends, using the small-world property "
             "to quickly converge on the neighborhood of the query vector.",
        metadata={"domain": "information_retrieval", "topic": "ann_search"},
    ),
    DocumentChunk(
        id="bm25-ranking",
        text="BM25 Okapi is a bag-of-words retrieval function that ranks documents by term frequency "
             "and inverse document frequency, penalizing documents that are excessively long relative "
             "to the corpus average. It remains a strong baseline in information retrieval benchmarks, "
             "often competitive with early neural rankers on keyword-heavy queries.",
        metadata={"domain": "information_retrieval", "topic": "sparse_ranking"},
    ),
    DocumentChunk(
        id="semantic-caching",
        text="Semantic caching exploits the observation that many user queries are paraphrases of each other. "
             "By comparing embedding-space cosine similarity of incoming queries against previously answered ones, "
             "a cache can serve stored results when similarity exceeds a calibrated threshold, bypassing "
             "expensive retrieval and generation passes entirely.",
        metadata={"domain": "ai_engineering", "topic": "latency_optimization"},
    ),
    DocumentChunk(
        id="kafka-event-streaming",
        text="Apache Kafka models topics as partitioned append-only commit logs. Producers write records "
             "to partition leaders; consumers read in order within each partition. Consumer groups enable "
             "parallel processing with automatic rebalancing. Exactly-once semantics are achieved via "
             "idempotent producers and transactional reads.",
        metadata={"domain": "data_engineering", "topic": "event_streaming"},
    ),
    DocumentChunk(
        id="cross-encoder-reranking",
        text="Cross-encoder rerankers pass the query and document jointly through a transformer, "
             "computing full token-level cross-attention. Unlike bi-encoders that embed independently, "
             "cross-encoders capture fine-grained lexical interactions, achieving higher precision "
             "at the cost of O(N) inference passes over the candidate pool.",
        metadata={"domain": "information_retrieval", "topic": "neural_reranking"},
    ),
    DocumentChunk(
        id="feature-drift-monitoring",
        text="Feature drift occurs when the distribution of input features changes between training "
             "and production. Statistical tests like the Kolmogorov-Smirnov test and Population Stability "
             "Index quantify drift magnitude. Continuous monitoring triggers retraining pipelines when "
             "drift exceeds operational thresholds.",
        metadata={"domain": "mlops", "topic": "data_quality"},
    ),
    DocumentChunk(
        id="wal-durability",
        text="Write-ahead logging ensures database durability by writing mutation records to a sequential "
             "log file before applying them to the in-memory data structure. On crash recovery, the WAL "
             "is replayed to restore the last consistent state. This separates the durability guarantee "
             "from the potentially expensive random I/O of updating the main storage.",
        metadata={"domain": "databases", "topic": "crash_recovery"},
    ),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.ingest(_SEED_CORPUS)
    yield


app = FastAPI(
    title="NeuroFlow RAG Engine",
    description="High-Throughput Hybrid Retrieval-Augmented Generation Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "neuroflow",
        "corpus_size": engine.corpus_size,
    }


@app.post("/api/v1/ingest", response_model=IngestResponse, status_code=201)
def ingest_documents(req: IngestRequest):
    import time
    t0 = time.perf_counter()
    count = engine.ingest(req.documents, clear_existing=req.clear_existing)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return IngestResponse(
        indexed_count=count,
        total_corpus_size=engine.corpus_size,
        indexing_time_ms=elapsed,
    )


@app.post("/api/v1/search", response_model=QueryResponse)
def search(req: QueryRequest):
    if engine.corpus_size == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Ingest documents first.")
    return engine.query(req)


@app.get("/api/v1/cache/stats", response_model=CacheStats)
def cache_stats():
    return CacheStats(**engine.cache.stats())


@app.delete("/api/v1/cache")
def clear_cache():
    engine.cache.clear()
    return {"status": "cleared"}
