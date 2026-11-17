"""Structured domain models using Pydantic v2 with strict validation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


class SearchMode(str, Enum):
    HYBRID = "hybrid"
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"


class DocumentChunk(BaseModel):
    """An atomic unit of indexed content with metadata."""
    id: str = Field(..., min_length=1, description="Unique chunk identifier")
    text: str = Field(..., min_length=1, description="Raw text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata (source, page, tags)")
    embedding: list[float] | None = Field(default=None, exclude=True, description="Dense embedding vector (populated during indexing)")


class RetrievalResult(BaseModel):
    """A single retrieved chunk with score provenance."""
    chunk: DocumentChunk
    final_score: float = Field(..., description="Post-fusion/post-rerank combined score")
    dense_score: float | None = Field(default=None, description="Raw cosine similarity from vector index")
    sparse_score: float | None = Field(default=None, description="Normalized BM25 Okapi score")
    rerank_score: float | None = Field(default=None, description="Cross-encoder relevance score")


class QueryRequest(BaseModel):
    """Incoming search request with tunable pipeline controls."""
    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of final results")
    alpha: float = Field(default=0.6, ge=0.0, le=1.0, description="Dense vs sparse weight (1.0 = pure dense)")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Retrieval strategy")
    enable_rerank: bool = Field(default=True, description="Apply cross-encoder reranking pass")
    enable_cache: bool = Field(default=True, description="Check/write semantic cache")


class QueryResponse(BaseModel):
    """Full response with telemetry."""
    query: str
    mode: SearchMode
    results: list[RetrievalResult]
    total_candidates: int = Field(description="Candidates before reranking/truncation")
    cached: bool = False
    cache_similarity: float | None = None
    latency_ms: float = Field(description="Total end-to-end latency")
    breakdown_ms: dict[str, float] = Field(default_factory=dict, description="Per-stage latency breakdown")


class IngestRequest(BaseModel):
    """Batch document ingestion request."""
    documents: list[DocumentChunk] = Field(..., min_length=1)
    clear_existing: bool = Field(default=False, description="Wipe current index before ingesting")


class IngestResponse(BaseModel):
    """Ingestion acknowledgement."""
    indexed_count: int
    total_corpus_size: int
    indexing_time_ms: float


class CacheStats(BaseModel):
    """Semantic cache observability."""
    entries: int
    hits: int
    misses: int
    hit_ratio: float
    evictions: int
    threshold: float
