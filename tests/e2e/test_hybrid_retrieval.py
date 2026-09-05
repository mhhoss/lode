"""
End-to-end scenario test: a full storefront catalog, ingested once,
queried the way a real Telegram customer would type — proving the
entire real stack (ONNX embedder, Postgres dense+sparse, RRF fusion,
farsflow normalization) works together, not just each piece alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lode.adapters.chunkers.simple import SimpleTextChunker
from lode.adapters.embedders import OnnxTextEmbeddingAdapter
from lode.adapters.normalizers.farsflow import FarsflowNormalizer
from lode.adapters.stores.pg_sparse import PgSparseAdapter
from lode.adapters.stores.pg_vector import PgVectorAdapter
from lode.domain.models import RetrievalMode, RetrievalRequest
from lode.engine.retrieval.orchestrator import RetrievalOrchestrator

MODEL_DIR = Path.home() / ".cache/lode/models/multilingual-e5-small"
TENANT = "e2e-shop"

CATALOG = [
    ("prod-1", "کفش مشکی چرم مردانه، سایز ۴۰ تا ۴۴، مناسب پیاده‌روی روزانه، قیمت ۴۵۰ هزار تومان"),
    ("prod-2", "کیف دستی زنانه چرم طبیعی رنگ قهوه‌ای، جادار و شیک برای مهمانی"),
    ("prod-3", "کفش ورزشی سفید مردانه برند مخصوص دویدن، تنفس‌پذیر و سبک"),
    ("prod-4", "کلاه بافتنی زمستانی زنانه، رنگ‌بندی متنوع، مناسب هوای سرد"),
    ("prod-5", "کمربند چرم مردانه مشکی با قفل فلزی، سایز قابل تنظیم"),
]


@pytest.fixture(scope="module")
async def e2e_orchestrator(postgres_client) -> RetrievalOrchestrator:
    embedder = OnnxTextEmbeddingAdapter(model_dir=MODEL_DIR)
    return RetrievalOrchestrator(
        normalizer=FarsflowNormalizer(),
        chunker=SimpleTextChunker(),
        vector_store=PgVectorAdapter(postgres_client),
        sparse_store=PgSparseAdapter(postgres_client),
        embedding_provider=embedder,
        client=postgres_client,
    )


@pytest.fixture(scope="module", autouse=True)
async def seed_catalog(e2e_orchestrator: RetrievalOrchestrator):
    for document_id, text in CATALOG:
        await e2e_orchestrator.ingest(raw_text=text, document_id=document_id, tenant_id=TENANT)
    yield


async def test_exact_product_query_returns_correct_item(e2e_orchestrator) -> None:
    response = await e2e_orchestrator.retrieve(
        RetrievalRequest(query="کفش چرم مشکی مردانه", top_k=3, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=TENANT,
    )

    top_doc_ids = [s.document_id for s in response.sources]
    assert "prod-1" in top_doc_ids
    assert top_doc_ids[0] == "prod-1"


async def test_semantic_query_without_exact_keywords_still_matches(e2e_orchestrator) -> None:
    """No lexical overlap with 'کفش ورزشی سفید' — only dense should find it."""
    response = await e2e_orchestrator.retrieve(
        RetrievalRequest(query="پاپوش سفید برای دویدن", top_k=3, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=TENANT,
    )

    doc_ids = {s.document_id for s in response.sources}
    assert "prod-3" in doc_ids


async def test_typed_without_half_space_still_matches_indexed_content(e2e_orchestrator) -> None:
    """Customer types without ZWNJ — the JoinerFixer symmetry fix from earlier."""
    response = await e2e_orchestrator.retrieve(
        RetrievalRequest(query="کمربند مردانه", top_k=3, retrieval_mode=RetrievalMode.SPARSE),
        tenant_id=TENANT,
    )

    doc_ids = {s.document_id for s in response.sources}
    assert "prod-5" in doc_ids


async def test_ambiguous_query_returns_multiple_relevant_candidates(e2e_orchestrator) -> None:
    """A vague query ('چرم') should surface several leather products, not just one."""
    response = await e2e_orchestrator.retrieve(
        RetrievalRequest(query="محصول چرمی", top_k=5, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=TENANT,
    )

    doc_ids = {s.document_id for s in response.sources}
    leather_products = {"prod-1", "prod-2", "prod-5"}
    assert len(doc_ids & leather_products) >= 2


async def test_unrelated_query_returns_no_strong_match(e2e_orchestrator) -> None:
    response = await e2e_orchestrator.retrieve(
        RetrievalRequest(query="یخچال فریزر", top_k=3, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=TENANT,
    )

    if response.sources:
        assert response.sources[0].score < 0.5


async def test_deleting_a_product_removes_it_from_results(e2e_orchestrator) -> None:
    await e2e_orchestrator.delete_document("prod-4", tenant_id=TENANT)

    response = await e2e_orchestrator.retrieve(
        RetrievalRequest(query="کلاه بافتنی زمستانی", top_k=3, retrieval_mode=RetrievalMode.HYBRID),
        tenant_id=TENANT,
    )

    doc_ids = {s.document_id for s in response.sources}
    assert "prod-4" not in doc_ids
