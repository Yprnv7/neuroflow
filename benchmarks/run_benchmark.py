"""
Automated benchmark harness.

Measures ingestion throughput, uncached query latency, cached query latency,
and retrieval quality metrics across a synthetic corpus.
"""

import time
import numpy as np

from neuroflow.core.engine import HybridRAGEngine
from neuroflow.models.schema import DocumentChunk, QueryRequest, SearchMode
from neuroflow.evals.metrics import run_evaluation


TOPICS = [
    "distributed consensus and leader election in fault-tolerant replicated state machines",
    "approximate nearest neighbor search using hierarchical navigable small world graphs",
    "relational database query optimization through cost-based plan selection",
    "real-time event stream processing with windowed aggregations and watermarks",
    "gradient descent optimization for deep neural network training convergence",
]


def generate_corpus(n: int) -> list[DocumentChunk]:
    chunks = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        chunks.append(DocumentChunk(
            id=f"bench-{i:04d}",
            text=f"Document {i} provides a detailed technical discussion of {topic}. "
                 f"This chunk covers implementation patterns, trade-offs, and benchmarks "
                 f"relevant to production systems operating at scale.",
            metadata={"topic_idx": i % len(TOPICS), "index": i},
        ))
    return chunks


def main():
    print("=" * 70)
    print("  NEUROFLOW ENGINE — PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)

    corpus_size = 500
    n_queries = 100
    engine = HybridRAGEngine()

    # 1. Ingestion benchmark
    print(f"\n[1/5] Ingesting {corpus_size} document chunks...")
    corpus = generate_corpus(corpus_size)
    t0 = time.perf_counter()
    engine.ingest(corpus)
    ingest_ms = (time.perf_counter() - t0) * 1000
    print(f"  → {ingest_ms:.1f} ms ({corpus_size / (ingest_ms / 1000):.0f} chunks/sec)")

    # 2. Uncached hybrid query latency
    print(f"\n[2/5] Benchmarking uncached HYBRID queries ({n_queries} queries)...")
    uncached_latencies = []
    for i in range(n_queries):
        topic = TOPICS[i % len(TOPICS)]
        resp = engine.query(QueryRequest(
            query=f"Explain {topic}",
            top_k=5,
            enable_cache=False,
            enable_rerank=True,
        ))
        uncached_latencies.append(resp.latency_ms)
    uncached = np.array(uncached_latencies)

    # 3. Cached query latency
    print(f"[3/5] Benchmarking CACHED queries ({n_queries} queries)...")
    # Seed the cache
    engine.query(QueryRequest(query="Tell me about distributed consensus", top_k=5, enable_cache=True))
    cached_latencies = []
    for _ in range(n_queries):
        resp = engine.query(QueryRequest(
            query="Explain distributed consensus mechanisms",
            top_k=5,
            enable_cache=True,
        ))
        cached_latencies.append(resp.latency_ms)
    cached = np.array(cached_latencies)

    # 4. Mode comparison
    print("[4/5] Comparing retrieval modes (HYBRID vs DENSE_ONLY vs SPARSE_ONLY)...")
    mode_latencies: dict[str, list[float]] = {"hybrid": [], "dense_only": [], "sparse_only": []}
    for i in range(50):
        topic = TOPICS[i % len(TOPICS)]
        for mode_name, mode_enum in [("hybrid", SearchMode.HYBRID), ("dense_only", SearchMode.DENSE_ONLY), ("sparse_only", SearchMode.SPARSE_ONLY)]:
            resp = engine.query(QueryRequest(query=topic, top_k=5, mode=mode_enum, enable_cache=False, enable_rerank=False))
            mode_latencies[mode_name].append(resp.latency_ms)

    # 5. Retrieval quality evaluation
    print("[5/5] Running retrieval quality evaluation...")
    eval_dataset = []
    for t_idx, topic in enumerate(TOPICS):
        expected = [f"bench-{i:04d}" for i in range(t_idx, corpus_size, len(TOPICS))]
        eval_dataset.append({"query": topic, "relevant_ids": expected})

    metrics = run_evaluation(
        eval_dataset,
        lambda q, k: engine.query(QueryRequest(query=q, top_k=k, enable_cache=False)),
        k=5,
    )

    # --- Report ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n  Ingestion")
    print(f"    Corpus size     : {corpus_size} chunks")
    print(f"    Ingestion time  : {ingest_ms:.1f} ms")

    print("\n  Uncached Query Latency (Hybrid + Rerank)")
    print(f"    P50             : {np.percentile(uncached, 50):.3f} ms")
    print(f"    P95             : {np.percentile(uncached, 95):.3f} ms")
    print(f"    P99             : {np.percentile(uncached, 99):.3f} ms")

    print("\n  Cached Query Latency")
    print(f"    P50             : {np.percentile(cached, 50):.3f} ms")
    print(f"    P95             : {np.percentile(cached, 95):.3f} ms")
    speedup = np.mean(uncached) / np.mean(cached) if np.mean(cached) > 0 else float("inf")
    print(f"    Speedup         : {speedup:.1f}x")

    print("\n  Mode Comparison (P50 latency, no rerank)")
    for mode_name, lats in mode_latencies.items():
        print(f"    {mode_name:14s}  : {np.percentile(lats, 50):.3f} ms")

    print("\n  Retrieval Quality")
    print(f"    MRR             : {metrics['mrr']:.4f}")
    print(f"    MAP             : {metrics['map']:.4f}")
    print(f"    NDCG@5          : {metrics['ndcg@5']:.4f}")
    print(f"    Precision@5     : {metrics['precision@5']:.4f}")
    print(f"    Recall@5        : {metrics['recall@5']:.4f}")

    print("\n  Cache Stats")
    stats = engine.cache.stats()
    print(f"    Hit ratio       : {stats['hit_ratio'] * 100:.1f}%")
    print(f"    Total entries   : {stats['entries']}")
    print(f"    Evictions       : {stats['evictions']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
