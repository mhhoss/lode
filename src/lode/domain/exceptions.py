"""
Lode domain exceptions.

This module defines the canonical error hierarchy for the Lode engine.

Design principles:
- Exceptions represent semantic runtime boundaries of the engine, not implementation details.
- Each error maps to a distinct subsystem responsibility: ingestion, retrieval, storage, and external providers.
- Provider errors are restricted to model-based systems (Embedding), while storage and retrieval backends
  (VectorStore, SparseStore) are modeled separately to preserve observability and debugging clarity.
- Errors are intended for control-flow decisions at the orchestration layer (Lodeto), not for internal algorithmic flow.
- Partial failures are first-class concepts where pipeline stages may succeed with degraded results (e.g., ingestion).

A new type of error is only valuable if it allows the caller to make a different decision.

This hierarchy is intentionally minimal and stable to avoid coupling domain logic to infrastructure specifics.
"""

# Base
class LodeError(Exception):
    """Base class for all domain-specific Lode exceptions."""
    pass


# Core pipeline
class IngestionError(LodeError):
    pass

class PartialIngestionError(IngestionError):
    pass


# Retrieval
class RetrievalError(LodeError):
    pass


# Storage
class StorageError(LodeError):
    pass

class VectorStoreError(StorageError):
    pass

class SparseStoreError(StorageError):
    pass


# Providers
class ProviderError(LodeError):
    pass

class ProviderAuthError(ProviderError):
    pass

class EmbeddingError(ProviderError):
    pass

