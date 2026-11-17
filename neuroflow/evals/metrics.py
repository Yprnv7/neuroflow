"""
Retrieval quality metrics for empirical evaluation.

Implements standard Information Retrieval evaluation metrics:
  - Mean Reciprocal Rank (MRR)
  - Precision@K
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - Mean Average Precision (MAP)

Also provides a benchmark harness that runs a full evaluation dataset
through the engine and aggregates metrics with latency percentiles.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from neuroflow.models.schema import QueryResponse


class RetrievalMetrics:
    """Stateless metric computation utilities."""

    @staticmethod
    def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if k <= 0:
            return 0.0
        top = retrieved[:k]
        hits = sum(1 for doc_id in top if doc_id in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 0.0
        top = retrieved[:k]
        hits = sum(1 for doc_id in top if doc_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def mrr(retrieved: list[str], relevant: set[str]) -> float:
        """Mean Reciprocal Rank: 1/rank of the first relevant document."""
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def average_precision(retrieved: list[str], relevant: set[str]) -> float:
        """
        Average Precision: mean of precision values at each relevant rank.
        AP = (1/|R|) * sum_{k: d_k relevant} P@k
        """
        if not relevant:
            return 0.0
        hits = 0
        sum_precision = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                hits += 1
                sum_precision += hits / rank
        return sum_precision / len(relevant) if relevant else 0.0

    @staticmethod
    def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain.
        DCG@K = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
        NDCG = DCG / IDCG
        """
        if not relevant or k <= 0:
            return 0.0

        top = retrieved[:k]

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(top):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)

        # Ideal DCG: all relevant docs ranked first
        ideal_hits = min(len(relevant), k)
        idcg = sum((2.0 ** 1.0 - 1.0) / math.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(
    eval_dataset: list[dict],
    query_fn: Callable[[str, int], QueryResponse],
    k: int = 5,
) -> dict[str, float]:
    """
    Run a full evaluation benchmark.

    Args:
        eval_dataset: list of {"query": str, "relevant_ids": list[str]}
        query_fn: callable(query, top_k) -> QueryResponse
        k: evaluation depth

    Returns:
        Aggregated metrics dict with means and latency percentiles.
    """
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []
    aps = []
    latencies = []

    for item in eval_dataset:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        response = query_fn(query, k)
        retrieved = [r.chunk.id for r in response.results]

        precisions.append(RetrievalMetrics.precision_at_k(retrieved, relevant, k))
        recalls.append(RetrievalMetrics.recall_at_k(retrieved, relevant, k))
        mrrs.append(RetrievalMetrics.mrr(retrieved, relevant))
        ndcgs.append(RetrievalMetrics.ndcg_at_k(retrieved, relevant, k))
        aps.append(RetrievalMetrics.average_precision(retrieved, relevant))
        latencies.append(response.latency_ms)

    lat = np.array(latencies)
    return {
        f"precision@{k}": round(float(np.mean(precisions)), 4),
        f"recall@{k}": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
        "map": round(float(np.mean(aps)), 4),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(lat, 99)), 3),
        "queries_evaluated": len(eval_dataset),
    }
