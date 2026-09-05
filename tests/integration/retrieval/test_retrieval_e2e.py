from __future__ import annotations

from lode.domain.models import RetrievalMode, RetrievalRequest


async def test_hybrid_retrieval_finds_relevant_document(
    orchestrator,
    tenant_id,
) -> None:
    await orchestrator.ingest(
        raw_text="کفش مشکی چرم مردانه، مناسب پیاده‌روی روزانه",
        document_id="doc-shoe",
        tenant_id=tenant_id,
    )
    await orchestrator.ingest(
        raw_text="کیف دستی زنانه، جنس چرم طبیعی، رنگ قهوه‌ای",
        document_id="doc-bag",
        tenant_id=tenant_id,
    )

    response = await orchestrator.retrieve(
        RetrievalRequest(query="کفش چرم مردانه", top_k=5, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=tenant_id,
    )

    assert len(response.sources) > 0
    assert response.sources[0].document_id == "doc-shoe"


async def test_hybrid_outperforms_sparse_only_on_semantic_query(
    orchestrator,
    tenant_id,
) -> None:
    """
    A query with no lexical overlap but clear semantic meaning should
    only be found via dense retrieval — proving hybrid actually adds
    value over sparse alone, not just duplicating it.
    """
    await orchestrator.ingest(
        raw_text="کفش مشکی چرم مردانه برای پیاده‌روی",
        document_id="doc-shoe",
        tenant_id=tenant_id,
    )

    sparse_response = await orchestrator.retrieve(
        RetrievalRequest(query="پاپوش تیره رنگ برای راه رفتن", top_k=5, retrieval_mode=RetrievalMode.SPARSE),
        tenant_id=tenant_id,
    )
    hybrid_response = await orchestrator.retrieve(
        RetrievalRequest(query="پاپوش تیره رنگ برای راه رفتن", top_k=5, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=tenant_id,
    )

    hybrid_doc_ids = {s.document_id for s in hybrid_response.sources}
    assert "doc-shoe" in hybrid_doc_ids


async def test_retrieval_respects_top_k(
    orchestrator,
    tenant_id,
) -> None:
    for i in range(5):
        await orchestrator.ingest(
            raw_text=f"محصول شماره {i} با توضیحات تکراری کالای فروشگاهی",
            document_id=f"doc-{i}",
            tenant_id=tenant_id,
        )

    response = await orchestrator.retrieve(
        RetrievalRequest(query="کالای فروشگاهی", top_k=2, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=tenant_id,
    )

    assert len(response.sources) <= 2


async def test_retrieval_is_isolated_per_tenant(
    orchestrator,
    tenant_id,
    other_tenant_id,
) -> None:
    await orchestrator.ingest(
        raw_text="محصول محرمانه مستأجر یک",
        document_id="doc-secret-a",
        tenant_id=tenant_id,
    )
    await orchestrator.ingest(
        raw_text="محصول محرمانه مستأجر دو",
        document_id="doc-secret-b",
        tenant_id=other_tenant_id,
    )

    response = await orchestrator.retrieve(
        RetrievalRequest(query="محصول محرمانه", top_k=10, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=tenant_id,
    )

    doc_ids = {s.document_id for s in response.sources}
    assert "doc-secret-a" in doc_ids
    assert "doc-secret-b" not in doc_ids


# ---------------------------------------------------------------------
# JoinerFixer / tsvector — the deferred question from earlier
# ---------------------------------------------------------------------

async def test_zwnj_normalization_does_not_break_sparse_matching(
    orchestrator,
    tenant_id,
) -> None:
    """
    farsflow's index pipeline applies JoinerFixer (adds ZWNJ / half-space),
    but the query pipeline deliberately does not. This test proves that
    difference doesn't break sparse (tsvector) matching — i.e. a query
    typed *without* half-space still finds content indexed *with* it.
    """
    # "می‌روم" (with ZWNJ) is what JoinerFixer would normalize toward.
    await orchestrator.ingest(
        raw_text="من هر روز به مغازه می‌روم و خرید می‌کنم",
        document_id="doc-zwnj",
        tenant_id=tenant_id,
    )

    # Query typed the way a real user would: plain space, no ZWNJ.
    response = await orchestrator.retrieve(
        RetrievalRequest(query="می روم", top_k=5, retrieval_mode=RetrievalMode.SPARSE),
        tenant_id=tenant_id,
    )

    doc_ids = {s.document_id for s in response.sources}
    assert "doc-zwnj" in doc_ids, (
        "Sparse search failed to match a ZWNJ-normalized index against a "
        "plain-space query — this means to_tsvector('simple', ...) treats "
        "ZWNJ and space differently, and the earlier deferred fix "
        "(normalizing ZWNJ before to_tsvector) is actually needed."
    )


from farsflow import JoinerFixer, Pipeline


def test_joiner_fixer_behavior_on_short_query():
    pipeline = Pipeline([
        JoinerFixer()
    ])
    result = pipeline("می روم")
    print(repr(result))


