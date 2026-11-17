"""Unit tests for score fusion strategies."""

from neuroflow.retrieval.fusion import convex_fusion, reciprocal_rank_fusion


class TestConvexFusion:
    def test_pure_dense(self):
        dense = {0: 0.9, 1: 0.5}
        sparse = {0: 0.3, 1: 0.8}
        fused = convex_fusion(dense, sparse, alpha=1.0)
        assert fused[0] == 0.9
        assert fused[1] == 0.5

    def test_pure_sparse(self):
        dense = {0: 0.9, 1: 0.5}
        sparse = {0: 0.3, 1: 0.8}
        fused = convex_fusion(dense, sparse, alpha=0.0)
        assert fused[0] == 0.3
        assert fused[1] == 0.8

    def test_balanced_fusion(self):
        dense = {0: 1.0}
        sparse = {0: 0.5}
        fused = convex_fusion(dense, sparse, alpha=0.5)
        assert fused[0] == 0.75

    def test_disjoint_candidate_sets(self):
        dense = {0: 0.8}
        sparse = {1: 0.6}
        fused = convex_fusion(dense, sparse, alpha=0.5)
        assert fused[0] == 0.4  # 0.5 * 0.8 + 0.5 * 0.0
        assert fused[1] == 0.3  # 0.5 * 0.0 + 0.5 * 0.6


class TestRRF:
    def test_basic_rrf(self):
        dense_ranked = [(0, 0.9), (1, 0.7)]
        sparse_ranked = [(1, 0.8), (0, 0.5)]
        scores = reciprocal_rank_fusion(dense_ranked, sparse_ranked, k=60)
        # Both docs appear in both lists, so both get contributions
        assert 0 in scores
        assert 1 in scores

    def test_single_system(self):
        dense_ranked = [(0, 0.9)]
        sparse_ranked = []
        scores = reciprocal_rank_fusion(dense_ranked, sparse_ranked, k=60)
        assert scores[0] == 1.0 / (60 + 1)
