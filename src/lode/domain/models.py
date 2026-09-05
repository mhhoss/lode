"""Domain models — the stable language of the Lode engine.

All models are frozen dataclasses: immutable, slot-optimized, and
dependency-free. Collections use tuple for immutability contracts.
metadata fields stay as plain dict — open-ended caller context.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------
type ChunkId = str
type DocumentId = str
type ChunkScores = dict[ChunkId, float]
type Embedding = tuple[float, ...]
type Embeddings = tuple[Embedding, ...]
type Filters = Mapping[str, str]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RetrievalMode(StrEnum):
    """
    Controls which retrieval backend(s) are used.
    HYBRID runs both and fuses via RRF.
    DENSE / SPARSE exist for debugging and A/B comparison.
    """

    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Document:
    """
    A single ingested document normalized to plain text.
    Parsing raw bytes (PDF, DOCX, HTML) is DocumentParser's job.
    """

    id: DocumentId
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """
    A contiguous slice of a Document after chunking.
    chunk_index preserves reading order for context assembly.
    Embeddings are never stored here — that's VectorStore's job.
    """

    id: ChunkId
    document_id: DocumentId
    content: str
    chunk_index: int  # Context Assembly
    # embedding: Store embedding only in Vector Store Adapters
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Source:
    """
    A retrieved chunk with its fused score, ready to cite.
    Flows unchanged from raw retrieval through RRF to the final response.
    """

    chunk_id: ChunkId
    document_id: DocumentId
    content: str
    score: float
    retrieval_mode: RetrievalMode
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """
    Incoming query as the engine sees it.
    Tenant identity is resolved upstream — the domain only sees a query.
    """

    query: str
    top_k: int = 5
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """Output of the retrieval subsystem — sources ranked by fused score."""

    sources: tuple[Source, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Example:
    # metadata={
    #   "cache_hit": True,
    #   "latency_ms": 123,
    #   "model": "...",
    #   "retrieved_chunks": 15,
    # }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of ingesting a single document."""

    document_id: DocumentId
    chunk_count: int
    success: bool
    error: str | None = None

    def __post_init__(self) -> None:

        if self.success and self.error is not None:
            raise ValueError("Successful ingestion cannot carry an error.")
        if not self.success and self.error is None:
            raise ValueError("Failed ingestion must provide an error message.")


@dataclass(frozen=True, slots=True)
class IngestionStats:
    """Aggregate stats for a batch ingestion — for monitoring only."""

    total_documents: int
    successful: int
    failed: int
    total_chunks: int

