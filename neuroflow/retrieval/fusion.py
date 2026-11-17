"""
Reciprocal Rank Fusion (RRF) and Convex Combination strategies
for merging sparse and dense retrieval candidate lists.

RRF is parameter-free and robust to score distribution differences.
Convex combination requires score normalization but allows explicit
control via the alpha parameter.
"""

from __future__ import annotations

from enum import Enum


class FusionStrategy(str, Enum):
    CONVEX = "convex"
    RRF = "rrf"


def convex_fusion(
    dense_scores: dict[int, float],
    sparse_scores: dict[int, float],
    alpha: float = 0.6,
) -> dict[int, float]:
    """
    Weighted linear combination: score = alpha * dense + (1 - alpha) * sparse

    Both score dicts must be pre-normalized to [0, 1].
    Alpha controls the bias:
      - alpha > 0.5: favor semantic/conceptual matching
      - alpha < 0.5: favor exact keyword/term matching
    """
    all_ids = set(dense_scores) | set(sparse_scores)
    fused = {}
    for idx in all_ids:
        d = dense_scores.get(idx, 0.0)
        s = sparse_scores.get(idx, 0.0)
        fused[idx] = alpha * d + (1.0 - alpha) * s
    return fused


def reciprocal_rank_fusion(
    dense_ranked: list[tuple[int, float]],
    sparse_ranked: list[tuple[int, float]],
    k: int = 60,
) -> dict[int, float]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF(d) = sum_r [ 1 / (k + rank_r(d)) ]

    where r iterates over the retrieval systems and rank_r(d) is the
    1-based rank of document d in system r. k=60 is the standard constant
    that dampens the influence of high-ranked documents.

    Advantage: no score normalization needed; purely rank-based.
    """
    scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    for rank, (idx, _) in enumerate(sparse_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    return scores
