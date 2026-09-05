"""
Retrieval & Ingestion Orchestrator.

This is the application layer of the Lode engine. It knows *how* and *when*
to call the domain interfaces, but it knows nothing about *how* they are implemented.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping
from dataclasses import replace
from typing import Any, Final

from lode.domain.exceptions import (
    EmbeddingError,
    IngestionError,
    RetrievalError,
)
from lode.domain.interfaces import (
    Chunker,
    EmbeddingProvider,
    Normalizer,
    SparseStore,
    UnitOfWork,
    VectorStore,
)
from lode.domain.models import (
    ChunkId,
    Document,
    Filters,
    IngestionResult,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
    Source,
)
from lode.domain.services import reciprocal_rank_fusion

DEFAULT_FUSION_SEARCH_FACTOR: Final = 2

from lode.observ import get_logger

logger = get_logger(__name__)


class RetrievalOrchestrator:
    """
    Coordinates the Hybrid Retrieval pipeline.
    Stateless w.r.t. tenant — tenant_id passed per call.
    """

    def __init__(
        self,
        normalizer: Normalizer,
        chunker: Chunker,
        vector_store: VectorStore,
        sparse_store: SparseStore,
        embedding_provider: EmbeddingProvider,
        client: UnitOfWork,
    ) -> None:
        self._normalizer = normalizer
        self._chunker = chunker
        self._vector_store = vector_store
        self._sparse_store = sparse_store
        self._embedding_provider = embedding_provider
        self._client = client


    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    async def ingest(
        self,
        raw_text: str,
        document_id: str,
        *,
        tenant_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> IngestionResult:
        """Normalize, chunk, embed, and persist a document for a given tenant."""
        logger.info("ingestion_started", extra={"tenant_id": tenant_id, "document_id": document_id})

        try:
            document = Document(
                id=document_id,
                content=raw_text,
                metadata=metadata if metadata is not None else {}
            )

            normalized_content = await self._normalizer.normalize_index(document.content)
            normalized_document = replace(document, content=normalized_content)

            chunks = await self._chunker.split(normalized_document)

            if not chunks:
                return IngestionResult(
                    document_id=document_id,
                    chunk_count=0,
                    success=True,
                )

            chunk_texts = tuple(
                c.content
                for c in chunks
            )

            embeddings = await self._embedding_provider.embed(
                chunk_texts,
                mode="document"
            )

            if not embeddings:
                raise EmbeddingError(
                    "Embedding provider returned no embeddings."
                )

            if len(embeddings) != len(chunks):
                raise EmbeddingError(
                    "Embedding count does not match chunk count."
                )

            async with self._client.transaction(tenant_id=tenant_id) as conn:
                await self._vector_store.upsert_chunks(chunks, embeddings, tenant_id=tenant_id, conn=conn)
            logger.info(
                "ingestion_completed",
                extra={"tenant_id": tenant_id, "document_id": document_id, "chunk_count": len(chunks)},
            )

            return IngestionResult(
                document_id=document_id,
                chunk_count=len(chunks),
                success=True,
            )

        except IngestionError:
            raise

        except Exception as exc:
            logger.error(
                "ingestion_failed",
                extra={"tenant_id": tenant_id, "document_id": document_id, "error": str(exc)},
            )
            raise IngestionError(
                "Ingestion pipeline failed."
            ) from exc


    async def delete_document(self, document_id: str, *, tenant_id: str) -> None:
        """Delete all chunks belonging to a document from both backends."""
        async with self._client.transaction(tenant_id=tenant_id) as conn:
            await self._vector_store.delete_document(document_id, tenant_id=tenant_id, conn=conn)


    async def delete_by_metadata(self, filters: Filters, *, tenant_id: str) -> None:
        async with self._client.transaction(tenant_id=tenant_id) as conn:
            await self._vector_store.delete_by_metadata(filters, tenant_id=tenant_id, conn=conn)


    async def list_documents(
        self,
        filters: Filters | None = None,
        *,
        tenant_id: str,
    ) -> tuple[str, ...]:
        """Vector store is the source of truth — both backends share document_ids."""
        return await self._vector_store.list_documents(filters, tenant_id=tenant_id)



    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> RetrievalResponse:
        """
        Execute hybrid retrieval, fuse results with RRF, and return top_k.
        """
        logger.info(
            "retrieval_started",
            extra={"tenant_id": tenant_id, "mode": request.retrieval_mode.value, "top_k": request.top_k},
        )

        try:
            if request.top_k <= 0:
                raise RetrievalError("top_k must be positive.")

            normalized_query = await self._normalizer.normalize_query(request.query)

            # Prepare concurrent tasks based on RetrievalMode
            tasks: list[Awaitable[tuple[Source, ...]]] = []

            if request.retrieval_mode in (
                RetrievalMode.HYBRID,
                RetrievalMode.DENSE
            ):
                embeddings = await self._embedding_provider.embed(
                    (normalized_query,),
                    mode="query",
                )

                if not embeddings:
                    raise RetrievalError(
                        "Embedding provider returned no embeddings."
                    )

                if len(embeddings) != 1:
                    raise RetrievalError(
                        "Embedding provider returned an invalid query embedding."
                    )

                tasks.append(
                    self._vector_store.search(
                        embeddings[0],
                        top_k=request.top_k * DEFAULT_FUSION_SEARCH_FACTOR,
                        tenant_id=tenant_id,
                    )
                )

            if request.retrieval_mode in (
                RetrievalMode.HYBRID,
                RetrievalMode.SPARSE
            ):
                tasks.append(
                    self._sparse_store.search(
                        normalized_query,
                        top_k=request.top_k * DEFAULT_FUSION_SEARCH_FACTOR,
                        tenant_id=tenant_id,
                    )
                )

            # Fire searches concurrently
            results = await asyncio.gather(*tasks)

            # Extract rankings
            rankings: list[list[ChunkId]] = []
            sources_by_chunk_id: dict[ChunkId, Source] = {}

            for sources in results:
                ranking: list[ChunkId] = []

                for source in sources:
                    ranking.append(source.chunk_id)
                    sources_by_chunk_id[source.chunk_id] = source
                rankings.append(ranking)

            if not rankings:
                return RetrievalResponse(sources=tuple())

            # Apply Reciprocal Rank Fusion
            fused_scores = reciprocal_rank_fusion(rankings)

            # Select the highest-ranked chunk ids.
            sorted_chunk_ids = sorted(
                fused_scores,
                key=lambda chunk_id: fused_scores[chunk_id],
                reverse=True,
            )[: request.top_k]

            # Build final sources with their fused scores
            final_sources = tuple(
                replace(
                    sources_by_chunk_id[chunk_id],
                    score=fused_scores[chunk_id],
                    retrieval_mode=request.retrieval_mode,
                )
                for chunk_id in sorted_chunk_ids
            )

            logger.info(
                "retrieval_completed",
                extra={"tenant_id": tenant_id, "result_count": len(final_sources)},
            )

            return RetrievalResponse(sources=final_sources)

        except RetrievalError:
            raise

        except Exception as exc:
            logger.error(
                "retrieval_failed",
                extra={"tenant_id": tenant_id, "error": str(exc)},
            )
            raise RetrievalError(
                "Retrieval pipeline failed."
            ) from exc

