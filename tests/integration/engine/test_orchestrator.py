from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest

from lode.domain.exceptions import IngestionError, VectorStoreError
from lode.domain.interfaces import (
    Chunker,
    EmbeddingProvider,
    Normalizer,
    SparseStore,
    UnitOfWork,
    VectorStore,
)
from lode.domain.models import (
    Document,
    DocumentChunk,
    Embedding,
    Embeddings,
    RetrievalMode,
    RetrievalRequest,
    Source,
)
from lode.engine.retrieval.orchestrator import RetrievalOrchestrator

# ========================================================
# Spy / Stub implementations
# ========================================================

class SpyNormalizer(Normalizer):
    async def normalize_index(self, text: str) -> str:
        return text

    async def normalize_query(self, text: str) -> str:
        return text


class SpyEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, texts: Sequence[str], *, mode: str = "document") -> Embeddings:
        self.calls.append(mode)
        return tuple((0.1, 0.2, 0.3) for _ in texts)


class SpyVectorStore(VectorStore):
    def __init__(self) -> None:
        self.upsert_called_with_embeddings = False
        self.search_called = False
        self._storage: dict[str, tuple[DocumentChunk, Embedding]] = {}
        self.last_search_tenant: str | None = None

    async def upsert_chunks(self, chunks, embeddings, *, tenant_id: str, conn=None) -> None:
        print("UPSERT:", len(chunks))
        for c in chunks:
            print(c.content)
        self.upsert_called_with_embeddings = True
        for c, e in zip(chunks, embeddings):
            self._storage[c.id] = (c, e)

    async def delete_document(self, document_id: str, *, tenant_id: str, conn=None) -> None:
        self._storage = {k: v for k, v in self._storage.items() if v[0].document_id != document_id}

    async def delete_by_metadata(self, filters, *, tenant_id: str, conn=None) -> None:
        pass

    async def list_documents(self, filters=None, *, tenant_id: str) -> tuple[str, ...]:
        return tuple({c.document_id for c, _ in self._storage.values()})

    async def chunk_count(self) -> int:
        return len(self._storage)

    async def search(self, query_embedding, *, top_k: int, tenant_id: str) -> tuple[Source, ...]:
        self.search_called = True
        self.last_search_tenant = tenant_id
        results = list(self._storage.values())[:top_k]
        return tuple(
            Source(chunk_id=c.id, document_id=c.document_id, content=c.content, score=0.99, retrieval_mode=RetrievalMode.DENSE)
            for c, _ in results
        )


class SpySparseStore(SparseStore):
    def __init__(self) -> None:
        self.search_called = False
        self._storage: dict[str, DocumentChunk] = {}
        self.last_search_tenant: str | None = None

    def seed(self, chunks: Sequence[DocumentChunk]) -> None:
        """Test helper — SparseStore has no upsert; seed its view directly."""
        for c in chunks:
            self._storage[c.id] = c

    async def search(self, query: str, *, top_k: int, tenant_id: str) -> tuple[Source, ...]:
        self.search_called = True
        self.last_search_tenant = tenant_id
        results = list(self._storage.values())[-top_k:]
        return tuple(
            Source(chunk_id=c.id, document_id=c.document_id, content=c.content, score=0.85, retrieval_mode=RetrievalMode.SPARSE)
            for c in results
        )


class SpyChunker(Chunker):
    async def split(self, document: Document) -> tuple[DocumentChunk, ...]:
        words = document.content.split()
        return tuple(
            DocumentChunk(id=str(uuid4()), document_id=document.id, content=w, chunk_index=i, metadata={})
            for i, w in enumerate(words)
        )


class EmptyChunker(Chunker):
    async def split(self, document: Document) -> tuple[DocumentChunk, ...]:
        return tuple()


class FailingVectorStore(VectorStore):
    async def upsert_chunks(self, chunks, embeddings, *, tenant_id: str, conn=None) -> None:
        raise VectorStoreError("Database connection lost")

    async def delete_document(self, document_id: str, *, tenant_id: str, conn=None) -> None:
        pass

    async def delete_by_metadata(self, filters, *, tenant_id: str, conn=None) -> None:
        pass

    async def list_documents(self, filters=None, *, tenant_id: str) -> tuple[str, ...]:
        return tuple()

    async def search(self, query_embedding, *, top_k: int, tenant_id: str) -> tuple[Source, ...]:
        return tuple()


class FakeUnitOfWork(UnitOfWork):
    """No real transaction — just yields None, since Spy stores ignore conn."""
    def transaction(self, *, tenant_id: str):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _cm():
            yield None

        return _cm()


TENANT = "test-tenant"


class SpyUnitOfWork(UnitOfWork):
    def __init__(self) -> None:
        self.transaction_called_with_tenant: str | None = None

    def transaction(self, *, tenant_id: str):
        self.transaction_called_with_tenant = tenant_id
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _cm():
            yield None

        return _cm()


# ========================================================
# Tests
# ========================================================

@pytest.fixture
def spies():
    return {
        "normalizer": SpyNormalizer(),
        "chunker": SpyChunker(),
        "vector_store": SpyVectorStore(),
        "sparse_store": SpySparseStore(),
        "embedding_provider": SpyEmbeddingProvider(),
        "client": FakeUnitOfWork(),
    }


@pytest.fixture
def orchestrator(spies) -> RetrievalOrchestrator:
    return RetrievalOrchestrator(**spies)


async def test_ingest_happy_path_and_data_flow(orchestrator, spies) -> None:
    result = await orchestrator.ingest("word1 word2 word3", document_id="doc-1", tenant_id=TENANT)

    assert result.success is True
    assert result.chunk_count == 3
    assert spies["vector_store"].upsert_called_with_embeddings is True
    assert "document" in spies["embedding_provider"].calls


async def test_ingest_empty_document_skips_embedding(spies) -> None:
    spies["chunker"] = EmptyChunker()
    orch = RetrievalOrchestrator(**spies)

    result = await orch.ingest(" ", document_id="doc-empty", tenant_id=TENANT)

    assert result.success is True
    assert result.chunk_count == 0
    assert len(spies["embedding_provider"].calls) == 0


async def test_retrieve_hybrid_fuses_and_overrides_scores(orchestrator, spies) -> None:
    await orchestrator.ingest("a b c d", document_id="doc-x", tenant_id=TENANT)
    spies["sparse_store"].seed(list(spies["vector_store"]._storage[k][0] for k in spies["vector_store"]._storage))

    request = RetrievalRequest(query="a", top_k=2, retrieval_mode=RetrievalMode.HYBRID)
    response = await orchestrator.retrieve(request, tenant_id=TENANT)

    assert len(response.sources) == 2
    for source in response.sources:
        assert source.score < 0.5, "Score must be the RRF result, not the raw store score"
        assert source.retrieval_mode == RetrievalMode.HYBRID


async def test_retrieve_dense_only_routing(spies) -> None:
    orch = RetrievalOrchestrator(**spies)
    await orch.ingest("a b c d", document_id="doc-y", tenant_id=TENANT)

    request = RetrievalRequest(query="a", top_k=1, retrieval_mode=RetrievalMode.DENSE)
    await orch.retrieve(request, tenant_id=TENANT)

    assert spies["sparse_store"].search_called is False
    assert "query" in spies["embedding_provider"].calls


async def test_retrieve_sparse_only_routing(spies) -> None:
    orch = RetrievalOrchestrator(**spies)
    await orch.ingest("a b c d", document_id="doc-z", tenant_id=TENANT)

    request = RetrievalRequest(query="a", top_k=1, retrieval_mode=RetrievalMode.SPARSE)
    await orch.retrieve(request, tenant_id=TENANT)

    assert spies["vector_store"].search_called is False
    assert spies["embedding_provider"].calls.count("document") == 1
    assert "query" not in spies["embedding_provider"].calls


async def test_ingest_raises_ingestion_error_on_storage_failure(spies) -> None:
    spies["vector_store"] = FailingVectorStore()
    orch = RetrievalOrchestrator(**spies)

    with pytest.raises(IngestionError):
        await orch.ingest("data", document_id="doc-fail", tenant_id=TENANT)


async def test_ingest_uses_transaction_with_correct_tenant(spies) -> None:
    spies["client"] = SpyUnitOfWork()
    orch = RetrievalOrchestrator(**spies)

    await orch.ingest("word1 word2", document_id="doc-tx", tenant_id=TENANT)

    assert spies["client"].transaction_called_with_tenant == TENANT


'''
async def test_reingest_same_document_does_not_duplicate_chunks(
    orchestrator,
    spies,
) -> None:
    await orchestrator.ingest(
        "one two three",
        document_id="doc-repeat",
        tenant_id=TENANT,
    )

    first_chunk_count = len(spies["vector_store"]._storage)

    await orchestrator.ingest(
        "one two three",
        document_id="doc-repeat",
        tenant_id=TENANT,
    )

    second_chunk_count = len(spies["vector_store"]._storage)

    assert second_chunk_count == first_chunk_count

    documents = await spies["vector_store"].list_documents(
        tenant_id=TENANT,
    )

    assert documents == ("doc-repeat",)
'''


async def test_delete_document_removes_all_chunks(
    orchestrator,
    spies,
) -> None:

    assert await spies["vector_store"].chunk_count() == 0

    chunks = await spies["chunker"].split(
        Document(
            id="x",
            content="one two three four",
        )
    )

    print(len(chunks))
    for c in chunks:
        print(repr(c.content))

    await orchestrator.ingest(
        "one two three four",
        document_id="doc-delete",
        tenant_id=TENANT,
    )

    assert await spies["vector_store"].chunk_count() == 4

    await orchestrator.delete_document(
        "doc-delete",
        tenant_id=TENANT,
    )

    assert await spies["vector_store"].chunk_count() == 0

    documents = await spies["vector_store"].list_documents(
        tenant_id=TENANT,
    )

    assert documents == ()


async def test_retrieve_propagates_tenant_to_all_stores(
    orchestrator,
    spies,
) -> None:
    await orchestrator.ingest(
        "apple banana orange",
        document_id="doc-tenant",
        tenant_id=TENANT,
    )

    spies["sparse_store"].seed(
        [
            chunk
            for chunk, _ in spies["vector_store"]._storage.values()
        ]
    )

    request = RetrievalRequest(
        query="apple",
        top_k=3,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    await orchestrator.retrieve(
        request,
        tenant_id=TENANT,
    )

    assert spies["vector_store"].last_search_tenant == TENANT
    assert spies["sparse_store"].last_search_tenant == TENANT


