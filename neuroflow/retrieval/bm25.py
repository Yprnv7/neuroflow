"""
BM25 Okapi probabilistic retrieval engine.

Implements the full BM25 Okapi ranking formula:
    score(q, d) = sum_t[ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl)) ]

where:
    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    N = corpus size
    df(t) = document frequency of term t
    tf(t,d) = term frequency of t in document d
    |d| = document length in tokens
    avgdl = average document length across the corpus

Score normalization to [0, 1] is applied post-retrieval for downstream
fusion compatibility with dense cosine scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter
import numpy as np

from neuroflow.models.schema import DocumentChunk
from neuroflow.utils.logging import get_logger

logger = get_logger(__name__)

# Precompiled tokenization pattern — alphanumeric sequences of 2+ chars
_TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] = []
        self._n: int = 0
        self._avgdl: float = 0.0
        self._doc_lens: np.ndarray = np.array([], dtype=np.float32)
        self._tf: list[Counter] = []          # per-doc term frequencies
        self._df: Counter = Counter()          # global document frequencies
        self._idf_cache: dict[str, float] = {} # precomputed IDF values

    @property
    def corpus_size(self) -> int:
        return self._n

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build inverted index from a list of document chunks."""
        self._chunks = chunks
        self._n = len(chunks)
        if self._n == 0:
            return

        self._tf = []
        self._df = Counter()
        lengths = []

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            tf = Counter(tokens)
            self._tf.append(tf)
            lengths.append(len(tokens))
            # Document frequency: count each unique term once per doc
            for term in tf:
                self._df[term] += 1

        self._doc_lens = np.array(lengths, dtype=np.float32)
        self._avgdl = float(np.mean(self._doc_lens)) if self._n > 0 else 1.0

        # Precompute IDF for every term in the vocabulary
        self._idf_cache = {}
        for term, df in self._df.items():
            self._idf_cache[term] = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

        logger.info(f"BM25 index built: {self._n} docs, {len(self._df)} unique terms, avgdl={self._avgdl:.1f}")

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Score and rank documents against the query.

        Returns list of (doc_index, normalized_score) sorted descending.
        Scores are min-max normalized to [0, 1] for fusion compatibility.
        """
        if self._n == 0:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # Accumulate BM25 score per document
        scores = np.zeros(self._n, dtype=np.float64)

        for term in q_tokens:
            idf = self._idf_cache.get(term)
            if idf is None:
                continue  # term not in corpus vocabulary

            for doc_idx in range(self._n):
                tf = self._tf[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                scores[doc_idx] += idf * (numerator / denominator)

        # Min-max normalization to [0, 1]
        max_s = scores.max()
        if max_s > 0:
            scores /= max_s

        # Argsort descending, filter zero scores
        ranked = np.argsort(scores)[::-1]
        results = [(int(idx), float(scores[idx])) for idx in ranked[:top_k] if scores[idx] > 0]
        return results
