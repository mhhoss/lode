from __future__ import annotations

import pytest

from lode.infra.postgres.client import PostgresClient

TENANT_A = "test-tenant-a"


@pytest.fixture(autouse=True)
async def clean_database(postgres_client: PostgresClient):
    async with postgres_client.transaction(tenant_id=TENANT_A) as conn:
        await conn.execute("TRUNCATE TABLE lode_chunks RESTART IDENTITY;")

    yield

    async with postgres_client.transaction(tenant_id=TENANT_A) as conn:
        await conn.execute("TRUNCATE TABLE lode_chunks RESTART IDENTITY;")
