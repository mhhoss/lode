from __future__ import annotations

import pytest

from lode.domain.models import (
    DocumentChunk,
)


def make_embedding() -> tuple[float, ...]:
    return tuple(0.1 for _ in range(384))

@pytest.mark.asyncio
async def test_vector_upsert_then_search_returns_chunk(
    vector_store,
    tenant_id,
) -> None:

    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        content="سامسونگ گلکسی A55",
        metadata={
            "brand": "samsung",
        },
    )

    embedding = (make_embedding(),)

    await vector_store.upsert_chunks(
        (chunk,),
        embedding,
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        embedding[0],
        top_k=5,
        tenant_id=tenant_id,
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == "chunk-1"
    assert result.document_id == "doc-1"
    assert result.content == "سامسونگ گلکسی A55"
    assert result.metadata["brand"] == "samsung"


@pytest.mark.asyncio
async def test_vector_search_returns_best_match_first(
    vector_store,
    tenant_id,
) -> None:
    chunk1 = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        content="سامسونگ A55",
        metadata={},
    )

    chunk2 = DocumentChunk(
        id="chunk-2",
        document_id="doc-2",
        chunk_index=0,
        content="آیفون 15",
        metadata={},
    )

    emb1 = tuple(1.0 if i == 0 else 0.0 for i in range(384))
    emb2 = tuple(1.0 if i == 1 else 0.0 for i in range(384))

    await vector_store.upsert_chunks(
        (chunk1, chunk2),
        (emb1, emb2),
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        emb1,
        top_k=2,
        tenant_id=tenant_id,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score >= results[1].score


@pytest.mark.asyncio
async def test_vector_search_respects_top_k(
    vector_store,
    tenant_id,
) -> None:
    chunks = tuple(
        DocumentChunk(
            id=f"chunk-{i}",
            document_id=f"doc-{i}",
            chunk_index=0,
            content=f"chunk {i}",
            metadata={},
        )
        for i in range(5)
    )

    embeddings = tuple(make_embedding() for _ in range(5))

    await vector_store.upsert_chunks(
        chunks,
        embeddings,
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        make_embedding(),
        top_k=3,
        tenant_id=tenant_id,
    )

    assert len(results) == 3


@pytest.mark.asyncio
async def test_updated_embedding_affects_search_order(
    vector_store,
    tenant_id,
) -> None:
    emb_old = tuple(1.0 if i == 0 else 0.0 for i in range(384))
    emb_new = tuple(1.0 if i == 1 else 0.0 for i in range(384))

    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        content="Samsung",
        metadata={"v": 1},
    )

    await vector_store.upsert_chunks(
        (chunk,),
        (emb_old,),
        tenant_id=tenant_id,
    )

    await vector_store.search(
        emb_old,
        top_k=1,
        tenant_id=tenant_id,
    )

    updated = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        content="Samsung",
        metadata={"v": 2},
    )

    await vector_store.upsert_chunks(
        (updated,),
        (emb_new,),
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        emb_new,
        top_k=1,
        tenant_id=tenant_id,
    )

    assert len(results) == 1
    assert results[0].metadata["v"] == 2


async def test_delete_document_removes_chunks_from_search(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    chunk = DocumentChunk(
        id="chunk-delete",
        document_id="doc-delete",
        content="iphone 16",
        chunk_index=0,
        metadata={},
    )

    await vector_store.upsert_chunks(
        (chunk,),
        (embedding_a,),
        tenant_id=tenant_id,
    )

    await vector_store.delete_document(
        "doc-delete",
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        embedding_a,
        top_k=10,
        tenant_id=tenant_id,
    )

    assert results == ()


async def test_search_empty_database_returns_empty_tuple(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    results = await vector_store.search(
        embedding_a,
        top_k=10,
        tenant_id=tenant_id,
    )

    assert results == ()


async def test_search_returns_multiple_chunks_from_same_document(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    chunks = (
        DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            content="part one",
            chunk_index=0,
            metadata={},
        ),
        DocumentChunk(
            id="chunk-2",
            document_id="doc-1",
            content="part two",
            chunk_index=1,
            metadata={},
        ),
    )

    embeddings = (
        embedding_a,
        embedding_a,
    )

    await vector_store.upsert_chunks(
        chunks,
        embeddings,
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        embedding_a,
        top_k=10,
        tenant_id=tenant_id,
    )

    ids = {r.chunk_id for r in results}

    assert ids == {"chunk-1", "chunk-2"}


async def test_search_returns_chunks_from_multiple_documents(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    chunks = (
        DocumentChunk(
            id="chunk-a",
            document_id="doc-a",
            content="apple",
            chunk_index=0,
            metadata={},
        ),
        DocumentChunk(
            id="chunk-b",
            document_id="doc-b",
            content="apple",
            chunk_index=0,
            metadata={},
        ),
    )

    await vector_store.upsert_chunks(
        chunks,
        (embedding_a, embedding_a),
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        embedding_a,
        top_k=10,
        tenant_id=tenant_id,
    )

    docs = {r.document_id for r in results}

    assert docs == {"doc-a", "doc-b"}
