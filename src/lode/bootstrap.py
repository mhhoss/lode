from __future__ import annotations

from pathlib import Path

from lode.adapters.chunkers import SimpleTextChunker
from lode.adapters.embedders import OnnxTextEmbeddingAdapter
from lode.adapters.normalizers import FarsflowNormalizer
from lode.adapters.stores import (
    PgSparseAdapter,
    PgVectorAdapter,
)
from lode.engine.retrieval import RetrievalOrchestrator
from lode.infra.postgres import PostgresClient


async def build_lode(
    *,
    database_url: str,
    model_dir: Path,
) -> RetrievalOrchestrator:
    """
    Build a fully configured RetrievalOrchestrator.

    This is the Composition Root of Lode.
    External applications should only call this function.
    """

    postgres = await PostgresClient.create(
        dsn=database_url,
    )

    normalizer = FarsflowNormalizer()

    chunker = SimpleTextChunker()

    embedding_provider = OnnxTextEmbeddingAdapter(
        model_dir=model_dir,
    )

    vector_store = PgVectorAdapter(
        postgres,
    )

    sparse_store = PgSparseAdapter(
        postgres,
    )

    orchestrator = RetrievalOrchestrator(
        normalizer=normalizer,
        chunker=chunker,
        vector_store=vector_store,
        sparse_store=sparse_store,
        embedding_provider=embedding_provider,
        client=postgres,
    )

    return orchestrator
