"""Application configuration with environment variable overrides."""

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    normalize: bool = True
    use_local_fallback: bool = True  # hash-projection fallback for CI/offline


@dataclass(frozen=True)
class CacheConfig:
    similarity_threshold: float = 0.92
    max_entries: int = 2048
    ttl_seconds: int = 3600


@dataclass(frozen=True)
class RetrievalConfig:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    default_alpha: float = 0.6  # dense weight in hybrid fusion
    candidate_pool_multiplier: int = 3  # top_k * this = candidate pool


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_local_fallback: bool = True  # lexical overlap fallback for CI/offline


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass(frozen=True)
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config() -> AppConfig:
    """Load config with optional environment variable overrides."""
    return AppConfig(
        embedding=EmbeddingConfig(
            model_name=os.getenv("NF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            dimension=int(os.getenv("NF_EMBED_DIM", "384")),
            use_local_fallback=os.getenv("NF_EMBED_FALLBACK", "true").lower() == "true",
        ),
        cache=CacheConfig(
            similarity_threshold=float(os.getenv("NF_CACHE_THRESHOLD", "0.92")),
            max_entries=int(os.getenv("NF_CACHE_MAX", "2048")),
            ttl_seconds=int(os.getenv("NF_CACHE_TTL", "3600")),
        ),
        retrieval=RetrievalConfig(
            bm25_k1=float(os.getenv("NF_BM25_K1", "1.5")),
            bm25_b=float(os.getenv("NF_BM25_B", "0.75")),
            default_alpha=float(os.getenv("NF_ALPHA", "0.6")),
        ),
        reranker=RerankerConfig(
            use_local_fallback=os.getenv("NF_RERANK_FALLBACK", "true").lower() == "true",
        ),
        server=ServerConfig(
            host=os.getenv("NF_HOST", "0.0.0.0"),
            port=int(os.getenv("NF_PORT", "8000")),
            log_level=os.getenv("NF_LOG_LEVEL", "info"),
        ),
    )
