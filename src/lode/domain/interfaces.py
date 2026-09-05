"""
Abstract contracts (protocols) for Lode infrastructure.

Design decisions:
- Uses typing.Protocol (structural subtyping), NOT abc.ABC.
  This means any object with the right methods satisfies the contract —
  no inheritance required, no import coupling.
- Every method is async. Even if an implementation is sync internally,
  the contract is async so the engine can await uniformly.
- Return types are domain models only. No infra types leak through.
- Methods are minimal — only what the engine actually needs.
  YAGNI: no bulk_delete, no list_collections, no stats methods.
  Add them when a real use case demands it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, AsyncContextManager, Literal, Protocol

# source of truth: mypy & pyright
from lode.domain.models import (
    Document,
    DocumentChunk,
    Embedding,
    Embeddings,
    Filters,
    Source,
)


class Normalizer(Protocol):
    """Normalize text separately for indexing and querying."""

    async def normalize_index(self, text: str) -> str: ...
    async def normalize_query(self, text: str) -> str: ...


class Chunker(Protocol):
    """Split a normalized Document into DocumentChunks."""

    async def split(
        self,
        document: Document,
    ) -> tuple[DocumentChunk, ...]:
        ...


class UnitOfWork(Protocol):
    """Transactional boundary. Implementations set tenant context internally."""
    def transaction(self, *, tenant_id: str) -> AsyncContextManager[Any]: ...


class VectorStore(Protocol):
    """Dense retrieval backend."""

    async def upsert_chunks(
        self,
        chunks: tuple[DocumentChunk, ...],
        embeddings: Embeddings,
        *,
        tenant_id: str,
        conn: Any | None = None,
    ) -> None: ...

    async def delete_document(
        self,
        document_id: str,
        *,
        tenant_id: str,
        conn: Any | None = None,
    ) -> None: ...

    async def delete_by_metadata(
        self,
        filters: Filters,
        *,
        tenant_id: str,
        conn: Any | None = None,
    ) -> None: ...

    async def list_documents(
        self,
        filters: Filters | None = None,
        *,
        tenant_id: str,
    ) -> tuple[str, ...]: ...

    async def search(
        self,
        query_embedding: Embedding,
        *,
        top_k: int,
        tenant_id: str,
    ) -> tuple[Source, ...]: ...


class SparseStore(Protocol):
    """Sparse retrieval backend. Stateless — tenant_id passed per call."""

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        tenant_id: str,
    ) -> tuple[Source, ...]: ...


class EmbeddingProvider(Protocol):
    """Generate vector embeddings from text."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        mode: Literal["query", "document"] = "document",
    ) -> Embeddings: ...

