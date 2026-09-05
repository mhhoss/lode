from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lode.adapters.stores.pg_vector import PgVectorAdapter
from lode.domain.models import DocumentChunk, RetrievalMode

# ---------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------

def test_accepts_default_table_name() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client=client)

    assert adapter._table_name == "lode_chunks"
    assert adapter._client is client


def test_accepts_custom_table_name() -> None:
    adapter = PgVectorAdapter(client=AsyncMock(), table_name="custom_chunks")
    assert adapter._table_name == "custom_chunks"


@pytest.mark.parametrize(
    "table_name",
    ["", "1chunks", "-chunks", "chunks-table", "chunks table",
     "chunks;", "chunks--", "chunks$", "drop table", "../chunks"],
)
def test_rejects_invalid_table_names(table_name: str) -> None:
    with pytest.raises(ValueError, match="Invalid table name"):
        PgVectorAdapter(client=AsyncMock(), table_name=table_name)


# ---------------------------------------------------------------------
# _to_halfvec
# ---------------------------------------------------------------------

def test_halfvec_conversion() -> None:
    result = PgVectorAdapter._to_halfvec((1.0, 2.5, 3.75))
    assert result == "[1.0,2.5,3.75]"


def test_halfvec_empty_embedding() -> None:
    result = PgVectorAdapter._to_halfvec(())
    assert result == "[]"


def test_halfvec_negative_values() -> None:
    result = PgVectorAdapter._to_halfvec((-1.0, -2.25, 3.0))
    assert result == "[-1.0,-2.25,3.0]"


def test_halfvec_preserves_precision() -> None:
    embedding = (0.123456789, -9.87654321)
    result = PgVectorAdapter._to_halfvec(embedding)
    assert result == "[0.123456789,-9.87654321]"


# ---------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> tuple[DocumentChunk, ...]:
    return (
        DocumentChunk(id="c1", document_id="doc1", content="chunk one", chunk_index=0, metadata={"lang": "fa"}),
        DocumentChunk(id="c2", document_id="doc1", content="chunk two", chunk_index=1, metadata={"lang": "fa"}),
    )


@pytest.fixture
def sample_embeddings():
    return ((1.0, 2.0), (3.0, 4.0))


async def test_upsert_chunks_empty_returns_without_db_call() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks((), (), tenant_id="tenant")

    client.executemany.assert_not_called()


async def test_upsert_chunks_requires_matching_lengths(sample_chunks) -> None:
    adapter = PgVectorAdapter(AsyncMock())

    with pytest.raises(ValueError, match="Chunk count does not match embedding count"):
        await adapter.upsert_chunks(sample_chunks, ((1.0, 2.0),), tenant_id="tenant")


async def test_upsert_chunks_calls_executemany(sample_chunks, sample_embeddings) -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks(sample_chunks, sample_embeddings, tenant_id="tenantA")

    client.executemany.assert_awaited_once()
    _, kwargs = client.executemany.await_args
    assert kwargs["tenant_id"] == "tenantA"


async def test_upsert_chunks_sql_contains_insert(sample_chunks, sample_embeddings) -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks(sample_chunks, sample_embeddings, tenant_id="tenant")

    sql = client.executemany.await_args.args[0]

    assert f"INSERT INTO {adapter._table_name}" in sql
    assert "ON CONFLICT (tenant_id, id)" in sql
    assert "embedding" in sql


async def test_upsert_chunks_passes_correct_number_of_rows(sample_chunks, sample_embeddings) -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks(sample_chunks, sample_embeddings, tenant_id="tenant")

    args = client.executemany.await_args.args[1]
    assert len(args) == 2


async def test_upsert_chunks_serializes_metadata(sample_chunks, sample_embeddings) -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks(sample_chunks, sample_embeddings, tenant_id="tenant")

    rows = client.executemany.await_args.args[1]
    assert rows[0][5] == {"lang": "fa"}


async def test_upsert_chunks_converts_embeddings_to_halfvec(sample_chunks, sample_embeddings) -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks(sample_chunks, sample_embeddings, tenant_id="tenant")

    rows = client.executemany.await_args.args[1]
    assert rows[0][6] == "[1.0,2.0]"
    assert rows[1][6] == "[3.0,4.0]"


async def test_upsert_chunks_preserves_chunk_order(sample_chunks, sample_embeddings) -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks(sample_chunks, sample_embeddings, tenant_id="tenant")

    rows = client.executemany.await_args.args[1]
    assert rows[0][0] == "c1"
    assert rows[1][0] == "c2"


async def test_upsert_chunks_passes_tenant_to_every_row(sample_chunks, sample_embeddings) -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.upsert_chunks(sample_chunks, sample_embeddings, tenant_id="tenantXYZ")

    rows = client.executemany.await_args.args[1]
    assert all(row[1] == "tenantXYZ" for row in rows)


async def test_upsert_chunks_uses_given_connection_instead_of_client() -> None:
    """When conn is provided, client.executemany must NOT be called."""
    client = AsyncMock()
    conn = AsyncMock()
    adapter = PgVectorAdapter(client)
    chunks = (DocumentChunk(id="c1", document_id="doc1", content="x", chunk_index=0, metadata={}),)

    await adapter.upsert_chunks(chunks, ((1.0,),), tenant_id="tenant", conn=conn)

    conn.executemany.assert_awaited_once()
    client.executemany.assert_not_called()


# ---------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------

async def test_delete_document_calls_execute() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_document("doc-1", tenant_id="tenantA")

    client.execute.assert_awaited_once()


async def test_delete_document_passes_correct_parameters() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_document("document42", tenant_id="tenantABC")

    args = client.execute.await_args.args
    kwargs = client.execute.await_args.kwargs
    sql, tenant, document = args

    assert "DELETE FROM" in sql
    assert tenant == "tenantABC"
    assert document == "document42"
    assert kwargs["tenant_id"] == "tenantABC"


async def test_delete_document_uses_given_connection() -> None:
    client = AsyncMock()
    conn = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_document("doc-1", tenant_id="tenant", conn=conn)

    conn.execute.assert_awaited_once()
    client.execute.assert_not_called()


# ---------------------------------------------------------------------
# delete_by_metadata
# ---------------------------------------------------------------------

async def test_delete_by_metadata_empty_filters_returns() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_by_metadata({}, tenant_id="tenant")

    client.execute.assert_not_called()


@pytest.mark.parametrize("key", ["", "bad-key", "bad key", "bad$key", "../etc"])
async def test_delete_by_metadata_rejects_invalid_keys(key: str) -> None:
    adapter = PgVectorAdapter(AsyncMock())

    with pytest.raises(ValueError, match="Invalid metadata key"):
        await adapter.delete_by_metadata({key: "value"}, tenant_id="tenant")


async def test_delete_by_metadata_single_filter() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_by_metadata({"lang": "fa"}, tenant_id="tenant")

    sql, tenant, value = client.execute.await_args.args
    assert "metadata->>'lang'" in sql
    assert tenant == "tenant"
    assert value == "fa"


async def test_delete_by_metadata_multiple_filters() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_by_metadata({"lang": "fa", "type": "manual"}, tenant_id="tenant")

    sql, *values = client.execute.await_args.args
    assert "metadata->>'lang'" in sql
    assert "metadata->>'type'" in sql
    assert values == ["tenant", "fa", "manual"]


async def test_delete_by_metadata_always_filters_by_tenant() -> None:
    client = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_by_metadata({"lang": "fa"}, tenant_id="tenantX")

    sql = client.execute.await_args.args[0]
    assert "tenant_id = $1" in sql


async def test_delete_by_metadata_uses_given_connection() -> None:
    client = AsyncMock()
    conn = AsyncMock()
    adapter = PgVectorAdapter(client)

    await adapter.delete_by_metadata({"lang": "fa"}, tenant_id="tenant", conn=conn)

    conn.execute.assert_awaited_once()
    client.execute.assert_not_called()


# ---------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------

async def test_list_documents_without_filters() -> None:
    client = AsyncMock()
    client.fetch.return_value = [{"document_id": "doc1"}, {"document_id": "doc2"}]
    adapter = PgVectorAdapter(client)

    result = await adapter.list_documents(tenant_id="tenant")

    assert result == ("doc1", "doc2")
    client.fetch.assert_awaited_once()


async def test_list_documents_with_single_filter() -> None:
    client = AsyncMock()
    client.fetch.return_value = []
    adapter = PgVectorAdapter(client)

    await adapter.list_documents({"lang": "fa"}, tenant_id="tenant")

    sql, tenant, value = client.fetch.await_args.args
    assert "metadata->>'lang'" in sql
    assert tenant == "tenant"
    assert value == "fa"


async def test_list_documents_with_multiple_filters() -> None:
    client = AsyncMock()
    client.fetch.return_value = []
    adapter = PgVectorAdapter(client)

    await adapter.list_documents({"lang": "fa", "type": "manual"}, tenant_id="tenant")

    sql, *values = client.fetch.await_args.args
    assert "metadata->>'lang'" in sql
    assert "metadata->>'type'" in sql
    assert values == ["tenant", "fa", "manual"]


@pytest.mark.parametrize("key", ["", "bad-key", "bad key", "$lang", "../etc"])
async def test_list_documents_rejects_invalid_keys(key: str) -> None:
    adapter = PgVectorAdapter(AsyncMock())

    with pytest.raises(ValueError, match="Invalid metadata key"):
        await adapter.list_documents({key: "value"}, tenant_id="tenant")


async def test_list_documents_returns_empty_tuple() -> None:
    client = AsyncMock()
    client.fetch.return_value = []
    adapter = PgVectorAdapter(client)

    result = await adapter.list_documents(tenant_id="tenant")
    assert result == ()


async def test_list_documents_always_filters_by_tenant() -> None:
    client = AsyncMock()
    client.fetch.return_value = []
    adapter = PgVectorAdapter(client)

    await adapter.list_documents(tenant_id="tenantABC")

    sql = client.fetch.await_args.args[0]
    assert "tenant_id = $1" in sql


async def test_list_documents_orders_by_document_id() -> None:
    client = AsyncMock()
    client.fetch.return_value = []
    adapter = PgVectorAdapter(client)

    await adapter.list_documents(tenant_id="tenant")

    sql = client.fetch.await_args.args[0]
    assert "ORDER BY document_id" in sql


# ---------------------------------------------------------------------
# search
# ---------------------------------------------------------------------

async def test_search_requires_positive_top_k() -> None:
    adapter = PgVectorAdapter(AsyncMock())

    with pytest.raises(ValueError, match="top_k must be positive"):
        await adapter.search(
            (1.0, 2.0),
            top_k=0,
            tenant_id="tenant",
        )


async def test_search_calls_fetch() -> None:
    client = AsyncMock()
    client.fetch.return_value = []

    adapter = PgVectorAdapter(client)

    await adapter.search(
        (1.0, 2.0),
        top_k=5,
        tenant_id="tenantA",
    )

    client.fetch.assert_awaited_once()


async def test_search_passes_correct_parameters() -> None:
    client = AsyncMock()
    client.fetch.return_value = []

    adapter = PgVectorAdapter(client)

    await adapter.search(
        (1.0, 2.0),
        top_k=7,
        tenant_id="tenantXYZ",
    )

    sql, embedding, tenant, top_k = client.fetch.await_args.args

    assert tenant == "tenantXYZ"
    assert top_k == 7
    assert embedding == "[1.0,2.0]"


async def test_search_sql_contains_expected_clauses() -> None:
    client = AsyncMock()
    client.fetch.return_value = []

    adapter = PgVectorAdapter(client)

    await adapter.search(
        (1.0,),
        top_k=3,
        tenant_id="tenant",
    )

    sql = client.fetch.await_args.args[0]

    assert "SELECT" in sql
    assert "embedding <=>" in sql
    assert "ORDER BY embedding <=>" in sql
    assert "LIMIT $3" in sql
    assert "tenant_id = $2" in sql


async def test_search_maps_rows_to_sources() -> None:
    client = AsyncMock()

    client.fetch.return_value = [
        {
            "id": "chunk1",
            "document_id": "doc1",
            "content": "hello",
            "score": 0.91,
            "metadata": {"lang": "fa"},
        }
    ]

    adapter = PgVectorAdapter(client)

    result = await adapter.search(
        (1.0,),
        top_k=5,
        tenant_id="tenant",
    )

    assert len(result) == 1

    source = result[0]

    assert source.chunk_id == "chunk1"
    assert source.document_id == "doc1"
    assert source.content == "hello"
    assert source.score == 0.91
    assert source.metadata == {"lang": "fa"}
    assert source.retrieval_mode is RetrievalMode.DENSE


async def test_search_replaces_null_metadata_with_empty_dict() -> None:
    client = AsyncMock()

    client.fetch.return_value = [
        {
            "id": "chunk1",
            "document_id": "doc1",
            "content": "hello",
            "score": 0.75,
            "metadata": None,
        }
    ]

    adapter = PgVectorAdapter(client)

    result = await adapter.search(
        (1.0,),
        top_k=5,
        tenant_id="tenant",
    )

    assert result[0].metadata == {}


async def test_search_preserves_database_order() -> None:
    client = AsyncMock()

    client.fetch.return_value = [
        {
            "id": "c1",
            "document_id": "doc1",
            "content": "first",
            "score": 0.9,
            "metadata": {},
        },
        {
            "id": "c2",
            "document_id": "doc2",
            "content": "second",
            "score": 0.8,
            "metadata": {},
        },
    ]

    adapter = PgVectorAdapter(client)

    result = await adapter.search(
        (1.0,),
        top_k=2,
        tenant_id="tenant",
    )

    assert tuple(source.chunk_id for source in result) == ("c1", "c2")


async def test_search_propagates_database_errors() -> None:
    client = AsyncMock()
    client.fetch.side_effect = RuntimeError("database exploded")

    adapter = PgVectorAdapter(client)

    with pytest.raises(RuntimeError, match="database exploded"):
        await adapter.search(
            (1.0,),
            top_k=5,
            tenant_id="tenant",
        )


