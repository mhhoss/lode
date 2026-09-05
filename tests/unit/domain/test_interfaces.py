from __future__ import annotations

import inspect
from typing import Protocol

import pytest

from lode.domain.interfaces import (
    Chunker,
    EmbeddingProvider,
    Normalizer,
    SparseStore,
    UnitOfWork,
    VectorStore,
)

_PROTOCOLS = (
    Normalizer,
    Chunker,
    VectorStore,
    SparseStore,
    EmbeddingProvider,
    UnitOfWork,
)


def test_every_interface_is_a_protocol() -> None:
    for interface in _PROTOCOLS:
        assert issubclass(interface, Protocol)


def test_protocols_are_not_runtime_checkable() -> None:
    class Dummy:
        pass

    for interface in _PROTOCOLS:
        with pytest.raises(TypeError):
            isinstance(Dummy(), interface)


def test_protocols_cannot_be_instantiated() -> None:
    for interface in _PROTOCOLS:
        with pytest.raises(TypeError):
            interface()


def test_protocols_are_not_abstract_base_classes() -> None:
    for interface in _PROTOCOLS:
        assert not inspect.isabstract(interface)


