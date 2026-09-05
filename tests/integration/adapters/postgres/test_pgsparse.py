from __future__ import annotations

import pytest

from lode.domain.models import DocumentChunk


async def test_search_finds_matching_chunk_by_content(
    vector_store,
    sparse_store,
    tenant_id,
    embedding_a,
) -> None:
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="سگ و گربه در خانه بازی می‌کنند",
        chunk_index=0,
        metadata={"source": "unit-test"},
    )

    await vector_store.upsert_chunks((chunk,), (embedding_a,), tenant_id=tenant_id)

    results = await sparse_store.search("گربه", top_k=10, tenant_id=tenant_id)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].content == chunk.content
    assert results[0].metadata == {"source": "unit-test"}


async def test_search_returns_empty_for_no_match(
    vector_store,
    sparse_store,
    tenant_id,
    embedding_a,
) -> None:
    chunk = DocumentChunk(
        id="chunk-1", document_id="doc-1", content="سلام دنیا", chunk_index=0, metadata={}
    )

    await vector_store.upsert_chunks((chunk,), (embedding_a,), tenant_id=tenant_id)

    results = await sparse_store.search("nonexistent_term_xyz", top_k=10, tenant_id=tenant_id)

    assert results == ()


async def test_search_respects_top_k(
    vector_store,
    sparse_store,
    tenant_id,
    embedding_a,
) -> None:
    chunks = tuple(
        DocumentChunk(
            id=f"chunk-{i}", document_id=f"doc-{i}", content="تکرار کلمه تست", chunk_index=0, metadata={}
        )
        for i in range(5)
    )
    embeddings = tuple(embedding_a for _ in range(5))

    await vector_store.upsert_chunks(chunks, embeddings, tenant_id=tenant_id)

    results = await sparse_store.search("تست", top_k=2, tenant_id=tenant_id)

    assert len(results) == 2


async def test_search_preserves_metadata(
    vector_store,
    sparse_store,
    tenant_id,
    embedding_a,
) -> None:
    chunk = DocumentChunk(
        id="chunk-meta",
        document_id="doc-meta",
        content="کتاب هوش مصنوعی",
        chunk_index=0,
        metadata={"author": "mohammad", "lang": "fa"},
    )

    await vector_store.upsert_chunks((chunk,), (embedding_a,), tenant_id=tenant_id)

    results = await sparse_store.search(
        "هوش",
        top_k=10,
        tenant_id=tenant_id,
    )

    assert len(results) == 1
    assert results[0].metadata == {"author": "mohammad", "lang": "fa"}


async def test_search_empty_query_returns_empty(
    sparse_store,
    tenant_id,
) -> None:
    results = await sparse_store.search(
        "",
        top_k=10,
        tenant_id=tenant_id,
    )

    assert results == ()


async def test_search_rejects_zero_top_k(
    sparse_store,
    tenant_id,
) -> None:
    with pytest.raises(ValueError):
        await sparse_store.search(
            "hello",
            top_k=0,
            tenant_id=tenant_id,
        )


async def test_search_rejects_negative_top_k(
    sparse_store,
    tenant_id,
) -> None:
    with pytest.raises(ValueError):
        await sparse_store.search(
            "hello",
            top_k=-1,
            tenant_id=tenant_id,
        )


async def test_search_ranks_better_match_first(
    vector_store,
    sparse_store,
    tenant_id,
    embedding_a,
) -> None:
    chunks = (
        DocumentChunk(
            id="c1",
            document_id="d1",
            content="گربه",
            chunk_index=0,
            metadata={},
        ),
        DocumentChunk(
            id="c2",
            document_id="d2",
            content="گربه گربه گربه",
            chunk_index=0,
            metadata={},
        ),
        DocumentChunk(
            id="c3",
            document_id="d3",
            content="گربه سگ پرنده",
            chunk_index=0,
            metadata={},
        ),
    )

    embeddings = (embedding_a, embedding_a, embedding_a)

    await vector_store.upsert_chunks(
        chunks,
        embeddings,
        tenant_id=tenant_id,
    )

    results = await sparse_store.search(
        "گربه",
        top_k=3,
        tenant_id=tenant_id,
    )

    assert len(results) == 3

    assert results[0].chunk_id == "c2"
