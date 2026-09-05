from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lode.domain.models import (
    Document,
    DocumentChunk,
    IngestionResult,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
    Source,
)

# ----------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------

def test_document_is_immutable() -> None:
    document = Document(
        id="doc-1",
        content="hello",
    )

    with pytest.raises(FrozenInstanceError):
        document.content = "changed"  # type: ignore[misc]


def test_document_default_metadata_is_empty() -> None:
    document = Document(
        id="doc",
        content="text",
    )

    assert document.metadata == {}


# ----------------------------------------------------------------------
# DocumentChunk
# ----------------------------------------------------------------------

def test_document_chunk_is_immutable() -> None:
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="hello",
        chunk_index=0,
    )

    with pytest.raises(FrozenInstanceError):
        chunk.content = "changed"  # type: ignore[misc]


def test_document_chunk_preserves_chunk_index() -> None:
    chunk = DocumentChunk(
        id="chunk",
        document_id="doc",
        content="text",
        chunk_index=7,
    )

    assert chunk.chunk_index == 7


# ----------------------------------------------------------------------
# Source
# ----------------------------------------------------------------------

def test_source_preserves_retrieval_mode() -> None:
    source = Source(
        chunk_id="chunk",
        document_id="doc",
        content="hello",
        score=0.42,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    assert source.retrieval_mode is RetrievalMode.HYBRID


def test_source_is_immutable() -> None:
    source = Source(
        chunk_id="chunk",
        document_id="doc",
        content="hello",
        score=1.0,
        retrieval_mode=RetrievalMode.DENSE,
    )

    with pytest.raises(FrozenInstanceError):
        source.score = 0.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# RetrievalRequest
# ----------------------------------------------------------------------

def test_retrieval_request_defaults() -> None:
    request = RetrievalRequest(
        query="hello",
    )

    assert request.top_k == 5
    assert request.retrieval_mode is RetrievalMode.HYBRID
    assert request.metadata == {}


def test_retrieval_request_rejects_non_positive_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        RetrievalRequest(
            query="hello",
            top_k=0,
        )


# ----------------------------------------------------------------------
# RetrievalResponse
# ----------------------------------------------------------------------

def test_retrieval_response_defaults() -> None:
    response = RetrievalResponse(
        sources=(),
    )

    assert response.sources == ()
    assert response.metadata == {}


# ----------------------------------------------------------------------
# IngestionResult
# ----------------------------------------------------------------------

def test_successful_ingestion_result() -> None:
    result = IngestionResult(
        document_id="doc",
        chunk_count=3,
        success=True,
    )

    assert result.success is True
    assert result.error is None


def test_failed_ingestion_result_requires_error() -> None:
    with pytest.raises(
        ValueError,
        match="Failed ingestion",
    ):
        IngestionResult(
            document_id="doc",
            chunk_count=0,
            success=False,
        )


def test_successful_ingestion_cannot_have_error() -> None:
    with pytest.raises(
        ValueError,
        match="Successful ingestion",
    ):
        IngestionResult(
            document_id="doc",
            chunk_count=1,
            success=True,
            error="boom",
        )


# ----------------------------------------------------------------------
# Enum
# ----------------------------------------------------------------------

def test_retrieval_mode_values_are_stable() -> None:
    assert RetrievalMode.HYBRID.value == "hybrid"
    assert RetrievalMode.DENSE.value == "dense"
    assert RetrievalMode.SPARSE.value == "sparse"
