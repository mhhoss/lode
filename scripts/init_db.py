"""One-off script to apply Lode migrations to a running Postgres instance."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from lode.infra.postgres.client import PostgresClient

MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "lode" / "infra" / "postgres" / "migrations"


async def main() -> None:
    dsn = os.environ["LODE_DATABASE_URL"]
    client = await PostgresClient.create(dsn)
    await client.run_migrations(MIGRATIONS_DIR)
    print(f"Migrations applied from {MIGRATIONS_DIR}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
