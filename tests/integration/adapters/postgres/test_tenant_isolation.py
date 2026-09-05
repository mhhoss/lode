"""RLS isolation test — the single most important integration test in Lode."""

from __future__ import annotations

from lode.domain.models import DocumentChunk


async def test_tenant_cannot_see_other_tenants_chunks(
    vector_store,
    tenant_id,
    other_tenant_id,
    embedding_a,
    embedding_b,
) -> None:
    chunk_a = DocumentChunk(
        id="chunk-a1",
        document_id="doc-a",
        content="secret data for tenant A",
        chunk_index=0,
        metadata={},
    )
    chunk_b = DocumentChunk(
        id="chunk-b1",
        document_id="doc-b",
        content="secret data for tenant B",
        chunk_index=0,
        metadata={},
    )

    await vector_store.upsert_chunks((chunk_a,), (embedding_a,), tenant_id=tenant_id)
    await vector_store.upsert_chunks((chunk_b,), (embedding_b,), tenant_id=other_tenant_id)

    docs_a = await vector_store.list_documents(tenant_id=tenant_id)
    docs_b = await vector_store.list_documents(tenant_id=other_tenant_id)

    assert docs_a == ("doc-a",)
    assert docs_b == ("doc-b",)
    assert "doc-b" not in docs_a
    assert "doc-a" not in docs_b


async def test_search_respects_tenant_boundary(
    vector_store,
    tenant_id,
    other_tenant_id,
    embedding_a,
) -> None:
    chunk_a = DocumentChunk(
        id="chunk-a1", document_id="doc-a", content="apple", chunk_index=0, metadata={}
    )
    chunk_b = DocumentChunk(
        id="chunk-b1", document_id="doc-b", content="apple", chunk_index=0, metadata={}
    )

    await vector_store.upsert_chunks((chunk_a,), (embedding_a,), tenant_id=tenant_id)
    await vector_store.upsert_chunks((chunk_b,), (embedding_a,), tenant_id=other_tenant_id)

    results = await vector_store.search(embedding_a, top_k=10, tenant_id=tenant_id)

    result_ids = {r.chunk_id for r in results}
    assert "chunk-a1" in result_ids
    assert "chunk-b1" not in result_ids


