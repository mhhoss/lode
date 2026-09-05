# Lode

High-performance, reusable Hybrid Retrieval Engine for building retrieval-augmented applications.

## Features

- Hybrid Retrieval (Dense + Sparse)
- PostgreSQL + pgvector
- PostgreSQL Full-Text Search
- Reciprocal Rank Fusion (RRF)
- Local embeddings via ONNX Runtime (E5-style sentence embedding models)
- Async architecture
- Zero framework dependency

## Embeddings

The application/domain layer is dimension-agnostic — `EmbeddingProvider` and
the engine accept any vector size. The PostgreSQL schema is not: it currently
fixes `lode_chunks.embedding` as `halfvec(384)` (`migrations/0001_init.sql`),
matching the current model — a 384-dimensional multilingual E5-family model.
Changing the embedding dimension or model requires a schema migration and a
full re-embedding/re-indexing of existing data; it does not inherently
require any application-code change.

## Requirements

- Python 3.12+
- PostgreSQL
- pgvector
- uv

## Installation

```bash
uv sync
```

## Development

1. Start PostgreSQL:

   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```

2. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

3. Apply database migrations:

   ```bash
   uv run python scripts/init_db.py
   ```

4. Run the unit test suite (no external services required):

   ```bash
   uv run pytest tests/unit
   ```

5. Integration and e2e tests additionally require PostgreSQL running (step 1)
   and a local ONNX embedding model directory cached under
   `~/.cache/lode/models/` (see `scripts/move_cache.sh`):

   ```bash
   uv run pytest tests/integration tests/e2e
   ```
