"""
Cross-encoder reranker for precision-critical second-stage ranking.

Unlike bi-encoders that encode query and document independently and compare
via dot product, a cross-encoder concatenates [query, document] and passes
them together through the full transformer attention stack. This captures
fine-grained token-level interactions at the cost of O(N) forward passes.

The fallback scorer uses a combination of:
  - Jaccard coefficient over token sets
  - Positional term proximity in the document
  - Length-normalized term overlap
"""

from __future__ import annotations

import math
import re

from neuroflow.models.schema import DocumentChunk
from neuroflow.utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b")


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_local_fallback: bool = False,
    ):
        self.model_name = model_name
        self._model = None
        self._fallback = use_local_fallback

        if not use_local_fallback:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model_name)
                logger.info(f"Cross-encoder loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Cross-encoder unavailable ({e}), using lexical fallback")
                self._fallback = True
        else:
            logger.info("Using lexical proximity fallback reranker")

    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank candidates by query-document relevance.
        Returns list of (chunk, score) sorted by descending relevance.
        """
        if not candidates:
            return []

        if not self._fallback and self._model is not None:
            return self._transformer_rerank(query, candidates, top_k)
        return self._lexical_rerank(query, candidates, top_k)

    def _transformer_rerank(
        self, query: str, candidates: list[DocumentChunk], top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        pairs = [[query, c.text] for c in candidates]
        raw_scores = self._model.predict(pairs)
        # Sigmoid normalization to [0, 1]
        scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _lexical_rerank(
        self, query: str, candidates: list[DocumentChunk], top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Fallback scoring using:
        1. Jaccard coefficient: |Q ∩ D| / |Q ∪ D|
        2. Coverage ratio: what fraction of query terms appear in doc
        3. Proximity bonus: consecutive query term matches in doc
        """
        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not q_tokens:
            return [(c, 0.0) for c in candidates[:top_k]]

        scored = []
        for chunk in candidates:
            d_tokens_list = _TOKEN_RE.findall(chunk.text.lower())
            d_tokens = set(d_tokens_list)

            if not d_tokens:
                scored.append((chunk, 0.0))
                continue

            # Jaccard coefficient
            intersection = q_tokens & d_tokens
            union = q_tokens | d_tokens
            jaccard = len(intersection) / len(union) if union else 0.0

            # Coverage: fraction of query terms found
            coverage = len(intersection) / len(q_tokens)

            # Proximity: count adjacent pairs of query terms in doc text
            proximity = 0.0
            for i in range(len(d_tokens_list) - 1):
                if d_tokens_list[i] in q_tokens and d_tokens_list[i + 1] in q_tokens:
                    proximity += 1.0
            proximity_norm = min(1.0, proximity / max(len(q_tokens), 1))

            # Weighted combination
            score = 0.4 * jaccard + 0.4 * coverage + 0.2 * proximity_norm
            scored.append((chunk, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
