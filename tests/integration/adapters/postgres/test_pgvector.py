from __future__ import annotations

from lode.domain.models import DocumentChunk


async def test_upsert_inserts_single_chunk(
    vector_store,
    tenant_id,
    embedding_a,
) -> None:
    # Arrange
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="hello world",
        chunk_index=0,
        metadata={},
    )

    # Act
    await vector_store.upsert_chunks(
        (chunk,),
        (embedding_a,),
        tenant_id=tenant_id,
    )

    # Assert
    documents = await vector_store.list_documents(
        tenant_id=tenant_id,
    )

    assert documents == ("doc-1",)


async def test_upsert_updates_existing_chunk(
    vector_store,
    tenant_id,
    embedding_a,
    embedding_b,
) -> None:
    # Arrange
    original = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="old content",
        chunk_index=0,
        metadata={"version": 1},
    )

    updated = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="new content",
        chunk_index=0,
        metadata={"version": 2},
    )

    # Act
    await vector_store.upsert_chunks(
        (original,),
        (embedding_a,),
        tenant_id=tenant_id,
    )

    await vector_store.upsert_chunks(
        (updated,),
        (embedding_b,),
        tenant_id=tenant_id,
    )

    results = await vector_store.search(
        embedding_b,
        top_k=10,
        tenant_id=tenant_id,
    )

    documents = await vector_store.list_documents(
    tenant_id=tenant_id,
    )

    # Assert
    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == "chunk-1"
    assert result.content == "new content"
    assert result.metadata == {"version": 2}

    assert documents == ("doc-1",)


async def test_upsert_within_explicit_transaction_commits_on_success(
    vector_store,
    postgres_client,
    tenant_id,
    embedding_a,
) -> None:
    chunk = DocumentChunk(
        id="chunk-tx", document_id="doc-tx", content="in transaction", chunk_index=0, metadata={}
    )

    async with postgres_client.transaction(tenant_id=tenant_id) as conn:
        await vector_store.upsert_chunks((chunk,), (embedding_a,), tenant_id=tenant_id, conn=conn)

    documents = await vector_store.list_documents(tenant_id=tenant_id)
    assert documents == ("doc-tx",)
