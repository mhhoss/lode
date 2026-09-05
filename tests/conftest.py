from __future__ import annotations

import os
from pathlib import Path

import pytest

from lode.adapters.chunkers.simple import SimpleTextChunker
from lode.adapters.embedders import OnnxTextEmbeddingAdapter
from lode.adapters.normalizers.farsflow import FarsflowNormalizer
from lode.adapters.stores.pg_sparse import PgSparseAdapter
from lode.adapters.stores.pg_vector import PgVectorAdapter
from lode.engine.retrieval.orchestrator import RetrievalOrchestrator
from lode.infra.postgres.client import PostgresClient

MODEL_DIR = Path.home() / ".cache/lode/models/multilingual-e5-small"

TEST_DSN = os.getenv(
    "LODE_TEST_DATABASE_URL",
    "postgresql://lode:lode_dev_password@localhost:5432/lode_test",
)

MIGRATIONS_DIR = (
    Path(__file__).parent.parent
    / "src" / "lode" / "infra" / "postgres" / "migrations"
)

TENANT_A = "test-tenant-a"
TENANT_B = "test-tenant-b"


# ---------------------------------------------------------------------
# PostgreSQL — session-scoped client, migrations applied once
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
async def postgres_client() -> PostgresClient:
    client = await PostgresClient.create(TEST_DSN, min_size=1, max_size=4)
    await client.run_migrations(MIGRATIONS_DIR)
    yield client
    await client.close()


# ---------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------

@pytest.fixture
def tenant_id() -> str:
    return TENANT_A


@pytest.fixture
def other_tenant_id() -> str:
    """A second tenant, used by RLS/isolation tests."""
    return TENANT_B


# ---------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------

@pytest.fixture
def vector_store(postgres_client: PostgresClient) -> PgVectorAdapter:
    return PgVectorAdapter(postgres_client)


@pytest.fixture
def sparse_store(postgres_client: PostgresClient) -> PgSparseAdapter:
    return PgSparseAdapter(postgres_client)


# ---------------------------------------------------------------------
# Make embedding
# ---------------------------------------------------------------------

def make_embedding(*, first: float = 1.0, second: float = 0.0) -> tuple[float, ...]:
    """
    Build a 384-dim embedding for tests — matches lode_chunks.embedding dimension.
    Only the first two dimensions vary between test vectors; the rest are zero.
    """
    return (first, second) + (0.0,) * 382


@pytest.fixture
def embedding_a() -> tuple[float, ...]:
    return make_embedding(first=1.0, second=0.0)


@pytest.fixture
def embedding_b() -> tuple[float, ...]:
    return make_embedding(first=0.0, second=1.0)


@pytest.fixture(scope="session")
def embedder() -> OnnxTextEmbeddingAdapter:
    return OnnxTextEmbeddingAdapter(model_dir=MODEL_DIR)


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

@pytest.fixture
def orchestrator(
    postgres_client: PostgresClient,
    vector_store: PgVectorAdapter,
    sparse_store: PgSparseAdapter,
    embedder: OnnxTextEmbeddingAdapter,
) -> RetrievalOrchestrator:
    return RetrievalOrchestrator(
        normalizer=FarsflowNormalizer(),
        chunker=SimpleTextChunker(),
        vector_store=vector_store,
        sparse_store=sparse_store,
        embedding_provider=embedder,
        client=postgres_client,
    )



