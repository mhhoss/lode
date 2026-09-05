CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS lode_chunks (
    id            text PRIMARY KEY,
    tenant_id     text NOT NULL,
    document_id   text NOT NULL,
    chunk_index   integer NOT NULL,
    content       text NOT NULL,
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding     halfvec(384),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lode_chunks_tenant ON lode_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lode_chunks_document ON lode_chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS idx_lode_chunks_search_vector ON lode_chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_lode_chunks_embedding ON lode_chunks USING hnsw (embedding halfvec_cosine_ops);