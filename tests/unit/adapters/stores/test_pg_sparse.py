from unittest.mock import AsyncMock

import pytest

from lode.adapters import PgSparseAdapter
from lode.domain import RetrievalMode
from lode.infra.postgres.client import PostgresClient

# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------

def test_constructor_stores_dependencies(
    postgres_client: PostgresClient,
) -> None:
    adapter = PgSparseAdapter(postgres_client)

    assert adapter._client is postgres_client
    assert adapter._table_name == "lode_chunks"


def test_constructor_accepts_custom_table_name(
    postgres_client: PostgresClient,
) -> None:
    adapter = PgSparseAdapter(
        postgres_client,
        table_name="custom_chunks",
    )

    assert adapter._table_name == "custom_chunks"


@pytest.mark.parametrize(
    "table_name",
    [
        "",
        "123table",
        "table-name",
        "table name",
        "table;",
        "table--",
        "table$",
    ],
)
def test_constructor_rejects_invalid_table_name(
    postgres_client: PostgresClient,
    table_name: str,
) -> None:
    with pytest.raises(ValueError, match="Invalid table name"):
        PgSparseAdapter(
            postgres_client,
            table_name=table_name,
        )


# ---------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_rejects_non_positive_top_k(
    postgres_client,
) -> None:
    adapter = PgSparseAdapter(postgres_client)

    with pytest.raises(ValueError, match="top_k must be positive"):
        await adapter.search(
            "query",
            top_k=0,
            tenant_id="tenant-1",
        )


@pytest.mark.asyncio
async def test_search_calls_fetch_with_expected_arguments(
    postgres_client,
) -> None:
    postgres_client.fetch = AsyncMock(return_value=[])

    adapter = PgSparseAdapter(postgres_client)

    await adapter.search(
        "black shoes",
        top_k=5,
        tenant_id="tenant-1",
    )

    postgres_client.fetch.assert_awaited_once()

    sql, query, tenant_id, top_k = postgres_client.fetch.await_args.args

    assert "ts_rank_cd" in sql
    assert "plainto_tsquery" in sql
    assert "search_vector" in sql

    assert query == "black shoes"
    assert tenant_id == "tenant-1"
    assert top_k == 5


@pytest.mark.asyncio
async def test_search_maps_records_to_sources(
    postgres_client,
) -> None:
    postgres_client.fetch = AsyncMock(
        return_value=[
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "black leather shoes",
                "score": 0.92,
                "metadata": {
                    "price": "120"
                },
            }
        ]
    )

    adapter = PgSparseAdapter(postgres_client)

    sources = await adapter.search(
        "black shoes",
        top_k=3,
        tenant_id="tenant-1",
    )

    assert len(sources) == 1

    source = sources[0]

    assert source.chunk_id == "chunk-1"
    assert source.document_id == "doc-1"
    assert source.content == "black leather shoes"
    assert source.score == 0.92
    assert source.metadata == {"price": "120"}
    assert source.retrieval_mode is RetrievalMode.SPARSE


@pytest.mark.asyncio
async def test_search_replaces_none_metadata_with_empty_dict(
    postgres_client,
) -> None:
    postgres_client.fetch = AsyncMock(
        return_value=[
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "text",
                "score": 0.5,
                "metadata": None,
            }
        ]
    )

    adapter = PgSparseAdapter(postgres_client)

    sources = await adapter.search(
        "query",
        top_k=2,
        tenant_id="tenant-1",
    )

    assert sources[0].metadata == {}


@pytest.mark.asyncio
async def test_search_preserves_database_order(
    postgres_client,
) -> None:
    postgres_client.fetch = AsyncMock(
        return_value=[
            {
                "id": "chunk-2",
                "document_id": "doc-2",
                "content": "second",
                "score": 0.9,
                "metadata": {},
            },
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "first",
                "score": 0.8,
                "metadata": {},
            },
        ]
    )

    adapter = PgSparseAdapter(postgres_client)

    sources = await adapter.search(
        "query",
        top_k=2,
        tenant_id="tenant-1",
    )

    assert [s.chunk_id for s in sources] == [
        "chunk-2",
        "chunk-1",
    ]


