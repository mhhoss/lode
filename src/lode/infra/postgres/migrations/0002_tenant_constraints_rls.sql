ALTER TABLE lode_chunks DROP CONSTRAINT IF EXISTS lode_chunks_pkey;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'lode_chunks_tenant_id_unique'
    ) THEN
        ALTER TABLE lode_chunks ADD CONSTRAINT lode_chunks_tenant_id_unique UNIQUE (tenant_id, id);
    END IF;
END $$;

ALTER TABLE lode_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON lode_chunks;
CREATE POLICY tenant_isolation ON lode_chunks
    USING (tenant_id = current_setting('app.tenant_id', true));