"""Unit tests for retrieval metrics."""

import pytest

from neuroflow.evals.metrics import RetrievalMetrics


class TestRetrievalMetrics:
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"b", "d"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 2) == 0.5
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 5) == 0.4

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"b", "d"}
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 2) == 0.5
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 5) == 1.0

    def test_mrr_first_hit_at_rank_3(self):
        retrieved = ["x", "y", "z", "target"]
        relevant = {"z"}
        assert RetrievalMetrics.mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_mrr_no_hit(self):
        assert RetrievalMetrics.mrr(["a", "b"], {"z"}) == 0.0

    def test_average_precision(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        # rank 1: hit, P@1 = 1/1; rank 3: hit, P@3 = 2/3
        # AP = (1/2) * (1 + 2/3) = 5/6
        ap = RetrievalMetrics.average_precision(retrieved, relevant)
        assert ap == pytest.approx(5 / 6, abs=0.001)

    def test_ndcg_perfect_ranking(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)

    def test_ndcg_reversed_ranking(self):
        retrieved = ["x", "a"]
        relevant = {"a"}
        # DCG = 0/log2(2) + 1/log2(3) = 0.631
        # IDCG = 1/log2(2) = 1.0
        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevant, 2)
        assert 0.5 < ndcg < 0.7

    def test_empty_relevant_set(self):
        assert RetrievalMetrics.precision_at_k(["a"], set(), 1) == 0.0
        assert RetrievalMetrics.recall_at_k(["a"], set(), 1) == 0.0
        assert RetrievalMetrics.ndcg_at_k(["a"], set(), 1) == 0.0
