from __future__ import annotations

from lode.domain.models import RetrievalMode, RetrievalRequest


async def test_ingest_persists_chunks_with_real_pipeline(
    orchestrator,
    vector_store,
    tenant_id,
) -> None:
    result = await orchestrator.ingest(
        raw_text=(
            "کفش مشکی چرم مردانه، سایز ۴۰ تا ۴۴. "
            "قیمت چهارصد و پنجاه هزار تومان. "
            "برای اطلاعات بیشتر تماس بگیرید."
        ),
        document_id="doc-shoe-1",
        tenant_id=tenant_id,
        metadata={"phone": "0912xxxxxxx"},
    )

    assert result.success
    assert result.chunk_count > 0

    documents = await vector_store.list_documents(tenant_id=tenant_id)
    assert "doc-shoe-1" in documents


async def test_ingest_empty_text_returns_zero_chunks(
    orchestrator,
    tenant_id,
) -> None:
    result = await orchestrator.ingest(
        raw_text="   ",
        document_id="doc-empty",
        tenant_id=tenant_id,
    )

    assert result.success
    assert result.chunk_count == 0


async def test_ingest_is_isolated_per_tenant(
    orchestrator,
    vector_store,
    tenant_id,
    other_tenant_id,
) -> None:
    await orchestrator.ingest(
        raw_text="محصول تست برای مستأجر یک",
        document_id="doc-tenant-a",
        tenant_id=tenant_id,
    )
    await orchestrator.ingest(
        raw_text="محصول تست برای مستأجر دو",
        document_id="doc-tenant-b",
        tenant_id=other_tenant_id,
    )

    docs_a = await vector_store.list_documents(tenant_id=tenant_id)
    docs_b = await vector_store.list_documents(tenant_id=other_tenant_id)

    assert "doc-tenant-a" in docs_a
    assert "doc-tenant-b" not in docs_a
    assert "doc-tenant-b" in docs_b
    assert "doc-tenant-a" not in docs_b


async def test_reingesting_same_document_id_updates_not_duplicates(
    orchestrator,
    tenant_id,
) -> None:
    await orchestrator.ingest(
        raw_text="نسخه اول محصول",
        document_id="doc-versioned",
        tenant_id=tenant_id,
    )
    result = await orchestrator.ingest(
        raw_text="نسخه دوم محصول با متن کاملا متفاوت و طولانی‌تر",
        document_id="doc-versioned",
        tenant_id=tenant_id,
    )

    assert result.success

    response = await orchestrator.retrieve(
        RetrievalRequest(query="نسخه دوم", top_k=10, retrieval_mode=RetrievalMode.SPARSE),
        tenant_id=tenant_id,
    )

    matched = [s for s in response.sources if s.document_id == "doc-versioned"]
    assert len(matched) >= 1
    assert any("نسخه دوم" in s.content for s in matched)


async def test_delete_then_reingest_restores_document(
    orchestrator,
    tenant_id,
) -> None:
    await orchestrator.ingest(
        raw_text="نسخه اول محصول",
        document_id="doc-recreate",
        tenant_id=tenant_id,
    )

    await orchestrator.delete_document(
        "doc-recreate",
        tenant_id=tenant_id,
    )

    await orchestrator.ingest(
        raw_text="نسخه جدید محصول بعد از حذف",
        document_id="doc-recreate",
        tenant_id=tenant_id,
    )

    response = await orchestrator.retrieve(
        RetrievalRequest(
            query="نسخه جدید",
            top_k=5,
            retrieval_mode=RetrievalMode.HYBRID,
        ),
        tenant_id=tenant_id,
    )

    assert any(
        s.document_id == "doc-recreate"
        and "نسخه جدید" in s.content
        for s in response.sources
    )


async def test_metadata_survives_ingestion_and_retrieval(
    orchestrator,
    tenant_id,
) -> None:
    metadata = {
        "brand": "Samsung",
        "category": "phone",
        "price": 1200,
    }

    await orchestrator.ingest(
        raw_text="گوشی سامسونگ گلکسی",
        document_id="doc-meta",
        tenant_id=tenant_id,
        metadata=metadata,
    )

    response = await orchestrator.retrieve(
        RetrievalRequest(
            query="سامسونگ",
            top_k=3,
            retrieval_mode=RetrievalMode.HYBRID,
        ),
        tenant_id=tenant_id,
    )

    source = next(
        s
        for s in response.sources
        if s.document_id == "doc-meta"
    )

    assert source.metadata == metadata


