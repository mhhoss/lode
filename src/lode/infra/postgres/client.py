"""
Thin wrapper around an asyncpg connection pool.

This module belongs to the infrastructure layer and is unaware of domain concepts.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import asyncpg
from asyncpg import Connection
from asyncpg.pool import PoolConnectionProxy

type AsyncPGConnection = Connection | PoolConnectionProxy


class PostgresClient:
    """Manages the asyncpg connection pool lifecycle."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await conn.set_type_codec(
            "jsonb",
            schema="pg_catalog",
            encoder=json.dumps,
            decoder=json.loads,
            format="text",
        )

    @classmethod
    async def create(cls, dsn: str, **pool_kwargs: Any) -> PostgresClient:
        pool = await asyncpg.create_pool(
            dsn=dsn,
            init=cls._init_connection,
            **pool_kwargs,
        )
        return cls(pool)

    async def run_migrations(self, migrations_dir: Path) -> None:
        """Apply all .sql files in order. No tracking table — idempotent SQL only."""
        async with self._pool.acquire() as conn:
            for file in sorted(migrations_dir.glob("*.sql")):
                await conn.execute(file.read_text())

    @asynccontextmanager
    async def transaction(
        self,
        *,
        tenant_id: str
    ) -> AsyncIterator[asyncpg.Connection]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.tenant_id', $1, true)",
                    tenant_id,
                )
                yield conn

    @asynccontextmanager
    async def connection(
        self,
        *,
        tenant_id: str,
    ) -> AsyncIterator[asyncpg.Connection]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.tenant_id', $1, true)",
                    tenant_id,
                )
                yield conn

    async def execute(
        self,
        query: str,
        *args: Any,
        tenant_id: str
    ) -> str:
        async with self.transaction(tenant_id=tenant_id) as conn:
            return await conn.execute(query, *args)

    async def fetch(
        self,
        query: str,
        *args: Any,
        tenant_id: str
    ) -> list[asyncpg.Record]:
        async with self.connection(tenant_id=tenant_id) as conn:
            return await conn.fetch(query, *args)

    async def fetchval(
        self,
        query: str,
        *args: Any,
        tenant_id: str
    ) -> Any:
        async with self.connection(tenant_id=tenant_id) as conn:
            return await conn.fetchval(query, *args)

    async def executemany(
        self,
        query: str,
        args: Sequence[tuple[Any, ...]],
        *,
        tenant_id: str,
    ) -> None:
        async with self.transaction(tenant_id=tenant_id) as conn:
            await conn.executemany(query, args)

    async def close(self) -> None:
        await self._pool.close()
