from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from lode.domain.exceptions import (
    IngestionError,
    RetrievalError,
)
from lode.domain.interfaces import UnitOfWork
from lode.domain.models import (
    DocumentChunk,
    IngestionResult,
    RetrievalMode,
    RetrievalRequest,
    Source,
)
from lode.engine.retrieval.orchestrator import RetrievalOrchestrator

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def normalizer() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def chunker() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def sparse_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def embedding_provider() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def orchestrator(
    normalizer,
    chunker,
    vector_store,
    sparse_store,
    embedding_provider,
    unit_of_work,
):
    return RetrievalOrchestrator(
        normalizer=normalizer,
        chunker=chunker,
        vector_store=vector_store,
        sparse_store=sparse_store,
        embedding_provider=embedding_provider,
        client=unit_of_work,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_constructor_stores_dependencies(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
    chunker: AsyncMock,
    vector_store: AsyncMock,
    sparse_store: AsyncMock,
    embedding_provider: AsyncMock,
    unit_of_work: UnitOfWork,
) -> None:

    assert orchestrator._normalizer is normalizer
    assert orchestrator._chunker is chunker
    assert orchestrator._vector_store is vector_store
    assert orchestrator._sparse_store is sparse_store
    assert orchestrator._embedding_provider is embedding_provider
    assert orchestrator._client is unit_of_work


async def test_ingest_happy_path(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
    chunker: AsyncMock,
    embedding_provider: AsyncMock,
    vector_store: AsyncMock,
) -> None:

    normalizer.normalize_index.return_value = "normalized"

    chunks = (
        DocumentChunk(
            id="c1",
            document_id="doc1",
            content="chunk1",
            chunk_index=0,
        ),
        DocumentChunk(
            id="c2",
            document_id="doc1",
            content="chunk2",
            chunk_index=1,
        ),
    )

    chunker.split.return_value = chunks

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
        (0.3, 0.4),
    )

    result = await orchestrator.ingest(
        raw_text="raw document",
        document_id="doc1",
        tenant_id="tenant-1",
    )

    assert result == IngestionResult(
        document_id="doc1",
        chunk_count=2,
        success=True,
    )

    normalizer.normalize_index.assert_awaited_once()

    chunker.split.assert_awaited_once()

    embedding_provider.embed.assert_awaited_once_with(
        (
            "chunk1",
            "chunk2",
        ),
        mode="document",
    )

    vector_store.upsert_chunks.assert_awaited_once()

    args = vector_store.upsert_chunks.await_args

    assert args.args[0] == chunks
    assert args.args[1] == (
        (0.1, 0.2),
        (0.3, 0.4),
    )

    assert args.kwargs["tenant_id"] == "tenant-1"
    assert "conn" in args.kwargs


async def test_ingest_returns_success_when_chunker_returns_no_chunks(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
    chunker: AsyncMock,
    embedding_provider: AsyncMock,
    vector_store: AsyncMock,
) -> None:

    normalizer.normalize_index.return_value = "normalized"

    chunker.split.return_value = ()

    result = await orchestrator.ingest(
        raw_text="hello",
        document_id="doc",
        tenant_id="tenant-1",
    )

    assert result == IngestionResult(
        document_id="doc",
        chunk_count=0,
        success=True,
    )

    embedding_provider.embed.assert_not_called()

    vector_store.upsert_chunks.assert_not_called()


async def test_ingest_raises_when_embedding_provider_returns_no_embeddings(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
    chunker: AsyncMock,
    embedding_provider: AsyncMock,
) -> None:

    normalizer.normalize_index.return_value = "normalized"

    chunker.split.return_value = (
        DocumentChunk(
            id="c1",
            document_id="doc",
            content="chunk",
            chunk_index=0,
        ),
    )

    embedding_provider.embed.return_value = ()

    with pytest.raises(
        IngestionError,
        match="Ingestion pipeline failed",
    ):
        await orchestrator.ingest(
            raw_text="text",
            document_id="doc",
            tenant_id="tenant-1",
        )


async def test_ingest_raises_when_embedding_count_does_not_match_chunks(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
    chunker: AsyncMock,
    embedding_provider: AsyncMock,
) -> None:

    normalizer.normalize_index.return_value = "normalized"

    chunker.split.return_value = (
        DocumentChunk(
            id="c1",
            document_id="doc",
            content="a",
            chunk_index=0,
        ),
        DocumentChunk(
            id="c2",
            document_id="doc",
            content="b",
            chunk_index=1,
        ),
    )

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
    )

    with pytest.raises(
        IngestionError,
        match="Ingestion pipeline failed",
    ):
        await orchestrator.ingest(
            raw_text="text",
            document_id="doc",
            tenant_id="tenant-1",
        )


async def test_ingest_wraps_normalizer_failure(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
) -> None:

    normalizer.normalize_index.side_effect = RuntimeError(
        "boom",
    )

    with pytest.raises(
        IngestionError,
        match="Ingestion pipeline failed",
    ):
        await orchestrator.ingest(
            raw_text="hello",
            document_id="doc",
            tenant_id="tenant-1",
        )


async def test_ingest_wraps_vector_store_failure(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
    chunker: AsyncMock,
    embedding_provider: AsyncMock,
    vector_store: AsyncMock,
) -> None:

    normalizer.normalize_index.return_value = "normalized"

    chunks = (
        DocumentChunk(
            id="c1",
            document_id="doc",
            content="chunk",
            chunk_index=0,
        ),
    )

    chunker.split.return_value = chunks

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
    )

    vector_store.upsert_chunks.side_effect = RuntimeError(
        "db error",
    )

    with pytest.raises(
        IngestionError,
        match="Ingestion pipeline failed",
    ):
        await orchestrator.ingest(
            raw_text="hello",
            document_id="doc",
            tenant_id="tenant-1",
        )


async def test_ingest_preserves_document_metadata(
    orchestrator: RetrievalOrchestrator,
    normalizer: AsyncMock,
    chunker: AsyncMock,
    embedding_provider: AsyncMock,
) -> None:

    normalizer.normalize_index.return_value = "normalized"

    metadata = {
        "source": "pdf",
        "lang": "fa",
    }

    chunker.split.return_value = (
        DocumentChunk(
            id="c1",
            document_id="doc",
            content="chunk",
            chunk_index=0,
            metadata=metadata,
        ),
    )

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
    )

    await orchestrator.ingest(
        raw_text="hello",
        document_id="doc",
        tenant_id="tenant-1",
        metadata=metadata,
    )

    chunker.split.assert_awaited_once()

    document = chunker.split.await_args.args[0]

    assert document.metadata == metadata


async def test_delete_document_calls_vector_store(
    orchestrator: RetrievalOrchestrator,
    vector_store: AsyncMock,
) -> None:

    await orchestrator.delete_document(
        "doc-1",
        tenant_id="tenant-1",
    )

    vector_store.delete_document.assert_awaited_once()

    args = vector_store.delete_document.await_args

    assert args.args == (
        "doc-1",
    )

    assert args.kwargs["tenant_id"] == "tenant-1"
    assert "conn" in args.kwargs


async def test_delete_document_propagates_vector_failure(
    orchestrator: RetrievalOrchestrator,
    vector_store: AsyncMock,
) -> None:

    vector_store.delete_document.side_effect = RuntimeError(
        "boom",
    )

    with pytest.raises(RuntimeError):
        await orchestrator.delete_document(
            "doc",
            tenant_id="tenant-1",
        )


async def test_delete_by_metadata_calls_vector_store(
    orchestrator: RetrievalOrchestrator,
    vector_store: AsyncMock,
) -> None:

    filters = {
        "lang": "fa",
        "source": "pdf",
    }

    await orchestrator.delete_by_metadata(
        filters,
        tenant_id="tenant-1",
    )

    vector_store.delete_by_metadata.assert_awaited_once()

    args = vector_store.delete_by_metadata.await_args

    assert args.args == (filters,)
    assert args.kwargs["tenant_id"] == "tenant-1"
    assert "conn" in args.kwargs


async def test_delete_by_metadata_propagates_vector_failure(
    orchestrator: RetrievalOrchestrator,
    vector_store: AsyncMock,
) -> None:

    vector_store.delete_by_metadata.side_effect = RuntimeError(
        "boom",
    )

    with pytest.raises(RuntimeError):
        await orchestrator.delete_by_metadata(
            {
                "lang": "fa",
            },
            tenant_id="tenant-1",
        )


async def test_list_documents_returns_vector_store_result(
    orchestrator: RetrievalOrchestrator,
    vector_store: AsyncMock,
) -> None:

    vector_store.list_documents.return_value = (
        "doc1",
        "doc2",
    )

    result = await orchestrator.list_documents(
        tenant_id="tenant-1",
    )

    assert result == (
        "doc1",
        "doc2",
    )

    vector_store.list_documents.assert_awaited_once_with(
        None,
        tenant_id="tenant-1",
    )


async def test_list_documents_passes_filters(
    orchestrator: RetrievalOrchestrator,
    vector_store: AsyncMock,
) -> None:

    filters = {
        "lang": "fa",
    }

    await orchestrator.list_documents(
        filters,
        tenant_id="tenant-1",
    )

    vector_store.list_documents.assert_awaited_once_with(
        filters,
        tenant_id="tenant-1",
    )


async def test_list_documents_propagates_store_failure(
    orchestrator: RetrievalOrchestrator,
    vector_store: AsyncMock,
) -> None:

    vector_store.list_documents.side_effect = RuntimeError(
        "boom",
    )

    with pytest.raises(RuntimeError):
        await orchestrator.list_documents(
            tenant_id="tenant-1",
        )


async def test_dense_mode_uses_only_vector_search(
    orchestrator: RetrievalOrchestrator,
    embedding_provider: AsyncMock,
    vector_store: AsyncMock,
    sparse_store: AsyncMock,
    normalizer: AsyncMock,
) -> None:

    normalizer.normalize_query.return_value = "normalized"

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
    )

    vector_store.search.return_value = (
        Source(
            chunk_id="1",
            document_id="doc",
            content="chunk",
            score=0.8,
            retrieval_mode=RetrievalMode.DENSE,
            metadata={},
        ),
    )

    request = RetrievalRequest(
        query="hello",
        top_k=1,
        retrieval_mode=RetrievalMode.DENSE,
    )

    await orchestrator.retrieve(
        request,
        tenant_id="tenant-1",
    )

    vector_store.search.assert_awaited_once()

    sparse_store.search.assert_not_called()


async def test_sparse_mode_skips_embedding_generation(
    orchestrator: RetrievalOrchestrator,
    embedding_provider: AsyncMock,
    sparse_store: AsyncMock,
    vector_store: AsyncMock,
    normalizer: AsyncMock,
) -> None:

    normalizer.normalize_query.return_value = "normalized"

    sparse_store.search.return_value = (
        Source(
            chunk_id="1",
            document_id="doc",
            content="chunk",
            score=1.0,
            retrieval_mode=RetrievalMode.SPARSE,
            metadata={},
        ),
    )

    request = RetrievalRequest(
        query="hello",
        top_k=1,
        retrieval_mode=RetrievalMode.SPARSE,
    )

    await orchestrator.retrieve(
        request,
        tenant_id="tenant-1",
    )

    embedding_provider.embed.assert_not_called()

    vector_store.search.assert_not_called()

    sparse_store.search.assert_awaited_once()


async def test_dense_mode_only_uses_vector_store(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
    sparse_store,
) -> None:
    normalizer.normalize_query.return_value = "normalized"

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
    )

    vector_store.search.return_value = ()

    request = RetrievalRequest(
        query="hello",
        top_k=5,
        retrieval_mode=RetrievalMode.DENSE,
    )

    await orchestrator.retrieve(
        request,
        tenant_id=tenant_id,
    )

    vector_store.search.assert_awaited_once()
    sparse_store.search.assert_not_called()


async def test_sparse_mode_only_uses_sparse_store(
    orchestrator,
    tenant_id,
    normalizer,
    sparse_store,
    vector_store,
) -> None:
    normalizer.normalize_query.return_value = "normalized"

    sparse_store.search.return_value = ()

    request = RetrievalRequest(
        query="hello",
        top_k=5,
        retrieval_mode=RetrievalMode.SPARSE,
    )

    await orchestrator.retrieve(
        request,
        tenant_id=tenant_id,
    )

    sparse_store.search.assert_awaited_once()
    vector_store.search.assert_not_called()


async def test_hybrid_mode_uses_both_backends(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
    sparse_store,
) -> None:
    normalizer.normalize_query.return_value = "normalized"

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
    )

    vector_store.search.return_value = ()
    sparse_store.search.return_value = ()

    request = RetrievalRequest(
        query="hello",
        top_k=5,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    await orchestrator.retrieve(
        request,
        tenant_id=tenant_id,
    )

    vector_store.search.assert_awaited_once()
    sparse_store.search.assert_awaited_once()


async def test_retrieve_rejects_empty_query_embedding(
    orchestrator,
    tenant_id,
    embedding_provider,
    normalizer,
) -> None:
    normalizer.normalize_query.return_value = "normalized"

    embedding_provider.embed.return_value = ()

    request = RetrievalRequest(
        query="hello",
        top_k=2,
        retrieval_mode=RetrievalMode.DENSE,
    )

    with pytest.raises(
        RetrievalError,
        match="Embedding provider returned no embeddings",
    ):
        await orchestrator.retrieve(
            request,
            tenant_id=tenant_id,
        )


async def test_retrieve_rejects_multiple_query_embeddings(
    orchestrator,
    tenant_id,
    embedding_provider,
    normalizer,
) -> None:
    normalizer.normalize_query.return_value = "normalized"

    embedding_provider.embed.return_value = (
        (0.1,),
        (0.2,),
    )

    request = RetrievalRequest(
        query="hello",
        top_k=2,
        retrieval_mode=RetrievalMode.DENSE,
    )

    with pytest.raises(
        RetrievalError,
        match="invalid query embedding",
    ):
        await orchestrator.retrieve(
            request,
            tenant_id=tenant_id,
        )


async def test_retrieve_returns_empty_response_when_no_sources(
    orchestrator,
    tenant_id,
    embedding_provider,
    vector_store,
    sparse_store,
    normalizer,
) -> None:
    normalizer.normalize_query.return_value = "normalized"

    embedding_provider.embed.return_value = (
        (0.1,),
    )

    vector_store.search.return_value = ()
    sparse_store.search.return_value = ()

    request = RetrievalRequest(
        query="hello",
        top_k=3,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    response = await orchestrator.retrieve(
        request,
        tenant_id=tenant_id,
    )

    assert response.sources == ()


async def test_query_is_normalized_before_search(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
) -> None:
    normalizer.normalize_query.return_value = "normalized query"

    embedding_provider.embed.return_value = (
        (0.1, 0.2),
    )

    vector_store.search.return_value = ()

    request = RetrievalRequest(
        query="Original Query",
        top_k=3,
        retrieval_mode=RetrievalMode.DENSE,
    )

    await orchestrator.retrieve(
        request,
        tenant_id=tenant_id,
    )

    normalizer.normalize_query.assert_awaited_once_with(
        "Original Query",
    )

    embedding_provider.embed.assert_awaited_once_with(
        ("normalized query",),
        mode="query",
    )


async def test_returns_empty_response_when_no_results(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
) -> None:
    normalizer.normalize_query.return_value = "normalized"

    embedding_provider.embed.return_value = (
        (0.1,),
    )

    vector_store.search.return_value = ()

    request = RetrievalRequest(
        query="hello",
        top_k=5,
        retrieval_mode=RetrievalMode.DENSE,
    )

    response = await orchestrator.retrieve(
        request,
        tenant_id=tenant_id,
    )

    assert response.sources == ()


async def test_rrf_merges_duplicate_chunk_ids(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
    sparse_store,
) -> None:
    normalizer.normalize_query.return_value = "query"

    embedding_provider.embed.return_value = (
        (0.1,),
    )

    shared = Source(
        chunk_id="chunk-1",
        document_id="doc",
        content="shared",
        score=0.9,
        retrieval_mode=RetrievalMode.DENSE,
        metadata={},
    )

    vector_store.search.return_value = (
        shared,
    )

    sparse_store.search.return_value = (
        replace(
            shared,
            retrieval_mode=RetrievalMode.SPARSE,
        ),
    )

    response = await orchestrator.retrieve(
        RetrievalRequest(
            query="hello",
            top_k=5,
            retrieval_mode=RetrievalMode.HYBRID,
        ),
        tenant_id=tenant_id,
    )

    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "chunk-1"


async def test_retrieve_limits_results_to_top_k(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
) -> None:
    normalizer.normalize_query.return_value = "query"

    embedding_provider.embed.return_value = (
        (0.1,),
    )

    vector_store.search.return_value = tuple(
        Source(
            chunk_id=str(i),
            document_id="doc",
            content=str(i),
            score=float(10 - i),
            retrieval_mode=RetrievalMode.DENSE,
            metadata={},
        )
        for i in range(10)
    )

    response = await orchestrator.retrieve(
        RetrievalRequest(
            query="hello",
            top_k=3,
            retrieval_mode=RetrievalMode.DENSE,
        ),
        tenant_id=tenant_id,
    )

    assert len(response.sources) == 3


async def test_response_retrieval_mode_matches_request(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
) -> None:
    normalizer.normalize_query.return_value = "query"

    embedding_provider.embed.return_value = (
        (0.1,),
    )

    vector_store.search.return_value = (
        Source(
            chunk_id="1",
            document_id="doc",
            content="hello",
            score=0.5,
            retrieval_mode=RetrievalMode.DENSE,
            metadata={},
        ),
    )

    response = await orchestrator.retrieve(
        RetrievalRequest(
            query="hello",
            top_k=1,
            retrieval_mode=RetrievalMode.DENSE,
        ),
        tenant_id=tenant_id,
    )

    assert (
        response.sources[0].retrieval_mode
        is RetrievalMode.DENSE
    )


async def test_backend_failure_is_wrapped(
    orchestrator,
    tenant_id,
    normalizer,
    embedding_provider,
    vector_store,
) -> None:
    normalizer.normalize_query.return_value = "query"

    embedding_provider.embed.return_value = (
        (0.1,),
    )

    vector_store.search.side_effect = RuntimeError(
        "database down",
    )

    with pytest.raises(
        RetrievalError,
        match="Retrieval pipeline failed",
    ):
        await orchestrator.retrieve(
            RetrievalRequest(
                query="hello",
                top_k=5,
                retrieval_mode=RetrievalMode.DENSE,
            ),
            tenant_id=tenant_id,
        )


