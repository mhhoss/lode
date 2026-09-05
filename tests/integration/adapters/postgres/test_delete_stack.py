from __future__ import annotations

from lode.domain.models import (
    DocumentChunk,
)


async def test_delete_by_metadata_removes_matching_chunks(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    chunk = DocumentChunk(
        id="chunk-meta-1",
        document_id="doc-meta",
        content="test",
        chunk_index=0,
        metadata={"category": "shoes"},
    )

    await vector_store.upsert_chunks(
        (chunk,),
        (embedding_a,),
        tenant_id=tenant_id,
    )

    await vector_store.delete_by_metadata(
        {"category": "shoes"},
        tenant_id=tenant_id,
    )

    documents = await vector_store.list_documents(
        tenant_id=tenant_id,
    )

    assert documents == ()




async def test_delete_document_removes_only_target_document(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    chunk1 = DocumentChunk(
        id="chunk-1",
        document_id="doc-a",
        content="apple",
        chunk_index=0,
        metadata={},
    )

    chunk2 = DocumentChunk(
        id="chunk-2",
        document_id="doc-b",
        content="banana",
        chunk_index=0,
        metadata={},
    )

    await vector_store.upsert_chunks(
        (chunk1, chunk2),
        (embedding_a, embedding_a),
        tenant_id=tenant_id,
    )

    await vector_store.delete_document(
        "doc-a",
        tenant_id=tenant_id,
    )

    documents = await vector_store.list_documents(
        tenant_id=tenant_id,
    )

    assert documents == ("doc-b",)


async def test_delete_by_metadata_keeps_non_matching_chunks(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    shoe = DocumentChunk(
        id="shoe",
        document_id="doc-shoe",
        content="shoe",
        chunk_index=0,
        metadata={"category": "shoes"},
    )

    phone = DocumentChunk(
        id="phone",
        document_id="doc-phone",
        content="phone",
        chunk_index=0,
        metadata={"category": "phones"},
    )

    await vector_store.upsert_chunks(
        (shoe, phone),
        (embedding_a, embedding_a),
        tenant_id=tenant_id,
    )

    await vector_store.delete_by_metadata(
        {"category": "shoes"},
        tenant_id=tenant_id,
    )

    documents = await vector_store.list_documents(
        tenant_id=tenant_id,
    )

    assert documents == ("doc-phone",)

