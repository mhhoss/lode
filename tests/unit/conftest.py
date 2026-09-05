from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lode.domain import (
    Chunker,
    EmbeddingProvider,
    Normalizer,
    SparseStore,
    VectorStore,
)
from lode.domain.interfaces import UnitOfWork
from lode.infra.postgres.client import PostgresClient

# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #

@pytest.fixture
def postgres_client() -> MagicMock:
    return MagicMock(spec=PostgresClient)


# --------------------------------------------------------------------------- #
# Domain interfaces
# --------------------------------------------------------------------------- #

@pytest.fixture
def normalizer() -> MagicMock:
    return MagicMock(spec=Normalizer)


@pytest.fixture
def chunker() -> MagicMock:
    return MagicMock(spec=Chunker)


@pytest.fixture
def embedding_provider() -> MagicMock:
    return MagicMock(spec=EmbeddingProvider)


@pytest.fixture
def vector_store() -> MagicMock:
    return MagicMock(spec=VectorStore)


@pytest.fixture
def sparse_store() -> MagicMock:
    return MagicMock(spec=SparseStore)


@pytest.fixture
def unit_of_work() -> MagicMock:
    uow = MagicMock(spec=UnitOfWork)

    transaction = AsyncMock()

    conn = MagicMock()

    transaction.__aenter__.return_value = conn
    transaction.__aexit__.return_value = False

    uow.transaction.return_value = transaction

    return uow


# --------------------------------------------------------------------------- #
# Common values
# --------------------------------------------------------------------------- #

@pytest.fixture
def tenant_id() -> str:
    return "tenant-123"



@pytest.fixture
def valid_model_dir(
    tmp_path: Path,
) -> Path:
    model_dir = tmp_path / "model"
    onnx_dir = model_dir / "onnx"

    onnx_dir.mkdir(parents=True)

    (onnx_dir / "model.onnx").touch()
    (onnx_dir / "tokenizer.json").write_text("{}")

    (onnx_dir / "config.json").write_text(
        json.dumps({})
    )

    (onnx_dir / "tokenizer_config.json").write_text(
        json.dumps({})
    )

    return model_dir
