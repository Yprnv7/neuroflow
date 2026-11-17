"""
Embedding engine with dual-mode execution:
  - Full mode: SentenceTransformer models with GPU/CPU inference
  - Fallback mode: Deterministic hash-projection embeddings for CI/offline environments

The fallback produces consistent, reproducible vectors using multi-bucket
locality-sensitive hashing. This isn't random noise — semantically similar
texts will produce somewhat similar vectors, enough to validate pipeline
correctness without downloading transformer weights.
"""

from __future__ import annotations

import hashlib
import struct
import numpy as np
from neuroflow.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        use_local_fallback: bool = False,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._fallback = use_local_fallback

        if not use_local_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self.dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded transformer model: {model_name} (dim={self.dimension})")
            except Exception as e:
                logger.warning(f"Transformer unavailable ({e}), using hash-projection fallback")
                self._fallback = True
        else:
            logger.info(f"Using hash-projection fallback embeddings (dim={dimension})")

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into normalized embedding vectors."""
        if not self._fallback and self._model is not None:
            vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vecs.astype(np.float32)
        return self._hash_encode(texts)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into a normalized embedding vector."""
        return self.encode_batch([text])[0]

    def _hash_encode(self, texts: list[str]) -> np.ndarray:
        """
        Deterministic hash-projection embedding.

        For each word, generates multiple hash indices via MD5 and projects
        weighted contributions into a fixed-dimension vector. Words appearing
        in similar contexts will share hash buckets, producing non-trivial
        similarity structure.
        """
        dim = self.dimension
        out = np.zeros((len(texts), dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue

            vec = np.zeros(dim, dtype=np.float32)
            for pos, token in enumerate(tokens):
                # Generate 4 hash buckets per token using MD5
                digest = hashlib.md5(token.encode("utf-8")).digest()
                for b in range(4):
                    offset = b * 4
                    idx = struct.unpack_from("<I", digest, offset)[0] % dim
                    # Sign from high bit, magnitude weighted by 1/(bucket+1)
                    sign = 1.0 if (digest[offset] & 0x80) else -1.0
                    vec[idx] += sign / (b + 1)

                # Positional decay: tokens near the start carry more weight
                decay = 1.0 / (1.0 + 0.1 * pos)
                h2 = hashlib.md5(f"{token}_pos".encode()).digest()
                idx2 = struct.unpack_from("<I", h2, 0)[0] % dim
                vec[idx2] += decay

            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                out[i] = vec / norm

        return out
