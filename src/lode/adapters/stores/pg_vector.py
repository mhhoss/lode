"""
PostgreSQL + pgvector adapter for Dense Retrieval.
"""

from __future__ import annotations

import re

from lode.domain.interfaces import VectorStore
from lode.domain.models import (
    DocumentChunk,
    Embedding,
    Embeddings,
    Filters,
    RetrievalMode,
    Source,
)
from lode.infra.postgres.client import AsyncPGConnection, PostgresClient

_SAFE_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_SAFE_METADATA_KEY = re.compile(r"^[a-zA-Z0-9_]+$")


class PgVectorAdapter(VectorStore):
    """
    VectorStore implementation backed by PostgreSQL + pgvector halfvec.

    Stateless — tenant_id is passed per call, not bound at construction.
    """

    def __init__(
        self,
        client: PostgresClient,
        *,
        table_name: str = "lode_chunks",
    ) -> None:
        if not _SAFE_TABLE_NAME.match(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")
        self._client = client
        self._table_name = table_name

    async def upsert_chunks(
        self,
        chunks: tuple[DocumentChunk, ...],
        embeddings: Embeddings,
        *,
        tenant_id: str,
        conn: AsyncPGConnection | None = None,
    ) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count does not match embedding count.")

        sql = f"""
            INSERT INTO {self._table_name}
                (id, tenant_id, document_id, chunk_index, content, metadata, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7::halfvec)
            ON CONFLICT (tenant_id, id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                chunk_index = EXCLUDED.chunk_index,
                content     = EXCLUDED.content,
                metadata    = EXCLUDED.metadata,
                embedding   = EXCLUDED.embedding;
        """

        args = [
            (
                chunk.id,
                tenant_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.content,
                chunk.metadata,
                self._to_halfvec(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        if conn is not None:
            await conn.executemany(sql, args)
        else:
            await self._client.executemany(sql, args, tenant_id=tenant_id)

    async def delete_document(
        self,
        document_id: str,
        *,
        tenant_id: str,
        conn: AsyncPGConnection | None = None,
    ) -> None:
        sql = f"DELETE FROM {self._table_name} WHERE tenant_id = $1 AND document_id = $2;"
        if conn is not None:
            await conn.execute(sql, tenant_id, document_id)
        else:
            await self._client.execute(sql, tenant_id, document_id, tenant_id=tenant_id)

    async def delete_by_metadata(
        self,
        filters: Filters,
        *,
        tenant_id: str,
        conn: AsyncPGConnection | None = None,
    ) -> None:
        if not filters:
            return

        conditions: list[str] = ["tenant_id = $1"]
        values: list[str] = [tenant_id]
        parameter = 2

        for key, value in filters.items():
            if not _SAFE_METADATA_KEY.fullmatch(key):
                raise ValueError(f"Invalid metadata key: {key!r}")
            conditions.append(f"metadata->>'{key}' = ${parameter}")
            values.append(value)
            parameter += 1

        sql = f"DELETE FROM {self._table_name} WHERE {' AND '.join(conditions)};"

        if conn is not None:
            await conn.execute(sql, *values)
        else:
            await self._client.execute(sql, *values, tenant_id=tenant_id)

    async def list_documents(
        self,
        filters: Filters | None = None,
        *,
        tenant_id: str,
    ) -> tuple[str, ...]:
        values: list[str] = [tenant_id]
        conditions = ["tenant_id = $1"]
        parameter = 2

        if filters:
            for key, value in filters.items():
                if not _SAFE_METADATA_KEY.fullmatch(key):
                    raise ValueError(f"Invalid metadata key: {key!r}")
                conditions.append(f"metadata->>'{key}' = ${parameter}")
                values.append(value)
                parameter += 1

        sql = f"""
            SELECT DISTINCT document_id
            FROM {self._table_name}
            WHERE {" AND ".join(conditions)}
            ORDER BY document_id;
        """

        rows = await self._client.fetch(sql, *values, tenant_id=tenant_id)
        return tuple(row["document_id"] for row in rows)

    async def search(
        self,
        embedding: Embedding,
        *,
        top_k: int,
        tenant_id: str,
    ) -> tuple[Source, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        sql = f"""
            SELECT
                id, document_id, content,
                1 - (embedding <=> $1::halfvec) AS score, metadata
            FROM {self._table_name}
            WHERE tenant_id = $2
            ORDER BY embedding <=> $1::halfvec
            LIMIT $3;
        """

        rows = await self._client.fetch(
            sql,
            self._to_halfvec(embedding),
            tenant_id,
            top_k,
            tenant_id=tenant_id,
        )

        return tuple(
            Source(
                chunk_id=row["id"],
                document_id=row["document_id"],
                content=row["content"],
                score=float(row["score"]),
                retrieval_mode=RetrievalMode.DENSE,
                metadata=row["metadata"] or {},
            )
            for row in rows
        )

    @staticmethod
    def _to_halfvec(embedding: Embedding) -> str:
        return "[" + ",".join(map(str, embedding)) + "]"
