from __future__ import annotations

from lode.domain.exceptions import (
    EmbeddingError,
    IngestionError,
    LodeError,
    PartialIngestionError,
    ProviderAuthError,
    ProviderError,
    RetrievalError,
    SparseStoreError,
    StorageError,
    VectorStoreError,
)


def test_all_domain_errors_are_lode_errors() -> None:
    """Every public domain exception must inherit from LodeError."""

    exception_types = (
        IngestionError,
        PartialIngestionError,
        RetrievalError,
        StorageError,
        VectorStoreError,
        SparseStoreError,
        ProviderError,
        ProviderAuthError,
        EmbeddingError,
    )

    for exc_type in exception_types:
        assert issubclass(exc_type, LodeError)


def test_partial_ingestion_error_is_an_ingestion_error() -> None:
    """Partial ingestion failures must remain catchable as ingestion failures."""

    assert issubclass(
        PartialIngestionError,
        IngestionError,
    )


def test_vector_store_error_is_a_storage_error() -> None:
    """Vector store failures belong to the storage hierarchy."""

    assert issubclass(
        VectorStoreError,
        StorageError,
    )


def test_sparse_store_error_is_a_storage_error() -> None:
    """Sparse store failures belong to the storage hierarchy."""

    assert issubclass(
        SparseStoreError,
        StorageError,
    )


def test_provider_auth_error_is_a_provider_error() -> None:
    """Authentication failures are provider failures."""

    assert issubclass(
        ProviderAuthError,
        ProviderError,
    )


def test_embedding_error_is_a_provider_error() -> None:
    """Embedding failures belong to the provider hierarchy."""

    assert issubclass(
        EmbeddingError,
        ProviderError,
    )


def test_lode_error_catches_all_domain_errors() -> None:
    """Catching LodeError must catch every domain-specific exception."""

    exception_types = (
        IngestionError,
        PartialIngestionError,
        RetrievalError,
        StorageError,
        VectorStoreError,
        SparseStoreError,
        ProviderError,
        ProviderAuthError,
        EmbeddingError,
    )

    for exc_type in exception_types:
        caught = False

        try:
            raise exc_type("boom")
        except LodeError:
            caught = True

        assert caught


def test_exception_message_is_preserved() -> None:
    """Custom messages should survive exception construction."""

    message = "database unavailable"

    exc = StorageError(message)

    assert str(exc) == message


def test_different_error_families_are_distinct() -> None:
    """Storage and provider failures must remain separate hierarchies."""

    assert not issubclass(
        StorageError,
        ProviderError,
    )

    assert not issubclass(
        ProviderError,
        StorageError,
    )


def test_retrieval_error_is_not_storage_error() -> None:
    """Retrieval failures must not silently become storage failures."""

    assert not issubclass(
        RetrievalError,
        StorageError,
    )

