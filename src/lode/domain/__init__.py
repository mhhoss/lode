"""Pure domain layer — zero external dependencies."""

from .exceptions import (
    EmbeddingError,
    IngestionError,
    LodeError,
    PartialIngestionError,
    ProviderAuthError,
    ProviderError,
    RetrievalError,
    SparseStoreError,
    StorageError,
    VectorStoreError,
)
from .interfaces import (
    Chunker,
    EmbeddingProvider,
    Normalizer,
    SparseStore,
    VectorStore,
)
from .models import (
    ChunkId,
    ChunkScores,
    Document,
    DocumentChunk,
    DocumentId,
    Embedding,
    Embeddings,
    Filters,
    IngestionResult,
    IngestionStats,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
    Source,
)
from .services import reciprocal_rank_fusion

__all__ = [
    # Type aliases
    "ChunkId",
    "DocumentId",
    "ChunkScores",
    "Embedding",
    "Embeddings",
    "Filters",
    # Models
    "RetrievalMode",
    "Document",
    "DocumentChunk",
    "Source",
    "RetrievalRequest",
    "RetrievalResponse",
    "IngestionResult",
    "IngestionStats",
    # Interfaces
    "Chunker",
    "EmbeddingProvider",
    "Normalizer",
    "SparseStore",
    "VectorStore",
    # Exceptions
    "LodeError",
    "IngestionError",
    "PartialIngestionError",
    "RetrievalError",
    "StorageError",
    "VectorStoreError",
    "SparseStoreError",
    "ProviderError",
    "ProviderAuthError",
    "EmbeddingError",
    # Services
    "reciprocal_rank_fusion",
]

