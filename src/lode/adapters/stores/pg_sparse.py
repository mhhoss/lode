"""
PostgreSQL + tsvector adapter for Sparse Retrieval.
"""

from __future__ import annotations

import re

from lode.domain.interfaces import SparseStore
from lode.domain.models import (
    RetrievalMode,
    Source,
)
from lode.infra.postgres.client import PostgresClient

_SAFE_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class PgSparseAdapter(SparseStore):
    """
    SparseStore implementation backed by PostgreSQL Full-Text Search.
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


    async def search(
        self,
        query: str,
        *,
        top_k: int,
        tenant_id: str,
    ) -> tuple[Source, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = query.strip()
        if not query:
            return ()

        sql = f"""
            SELECT
                id, document_id, content,
                ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) AS score,
                metadata
            FROM {self._table_name}
            WHERE tenant_id = $2 AND search_vector @@ plainto_tsquery('simple', $1)
            ORDER BY score DESC
            LIMIT $3;
        """

        rows = await self._client.fetch(
            sql,
            query,
            tenant_id,
            top_k,
            tenant_id=tenant_id
        )

        return tuple(
            Source(
                chunk_id=row["id"],
                document_id=row["document_id"],
                content=row["content"],
                score=float(row["score"]),
                retrieval_mode=RetrievalMode.SPARSE,
                metadata=row["metadata"] or {},
            )
            for row in rows
        )

