from __future__ import annotations

import pytest

from lode.adapters.normalizers.farsflow import (
    FarsflowNormalizer,
)


@pytest.mark.asyncio
async def test_normalize_index_returns_string() -> None:
    normalizer = FarsflowNormalizer()

    result = await normalizer.normalize_index(
        "سلام     دنیا"
    )

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_normalize_query_returns_string() -> None:
    normalizer = FarsflowNormalizer()

    result = await normalizer.normalize_query(
        "سلام     دنیا"
    )

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_normalize_index_handles_empty_text() -> None:
    normalizer = FarsflowNormalizer()

    result = await normalizer.normalize_index("")

    assert result == ""


@pytest.mark.asyncio
async def test_normalize_query_handles_empty_text() -> None:
    normalizer = FarsflowNormalizer()

    result = await normalizer.normalize_query("")

    assert result == ""


@pytest.mark.asyncio
async def test_normalize_index_is_deterministic() -> None:
    normalizer = FarsflowNormalizer()

    text = "سلام     دنیا"

    first = await normalizer.normalize_index(text)
    second = await normalizer.normalize_index(text)

    assert first == second


@pytest.mark.asyncio
async def test_normalize_query_is_deterministic() -> None:
    normalizer = FarsflowNormalizer()

    text = "سلام     دنیا"

    first = await normalizer.normalize_query(text)
    second = await normalizer.normalize_query(text)

    assert first == second


@pytest.mark.asyncio
async def test_normalize_index_preserves_non_empty_output() -> None:
    normalizer = FarsflowNormalizer()

    result = await normalizer.normalize_index(
        "مى روم   خانه"
    )

    assert result
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_normalize_query_preserves_non_empty_output() -> None:
    normalizer = FarsflowNormalizer()

    result = await normalizer.normalize_query(
        "مى روم   خانه"
    )

    assert result
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_index_and_query_normalization_are_independent() -> None:
    normalizer = FarsflowNormalizer()

    text = "مى روم خانه"

    index_result = await normalizer.normalize_index(text)
    query_result = await normalizer.normalize_query(text)

    assert isinstance(index_result, str)
    assert isinstance(query_result, str)


async def test_normalize_query_applies_joiner_fixer() -> None:
    normalizer = FarsflowNormalizer()

    result = await normalizer.normalize_query("می روم")

    assert result == "می\u200cروم"


async def test_normalize_index_and_query_produce_same_zwnj_pattern() -> None:
    """Both pipelines must agree on ZWNJ placement for identical input,
    so sparse (tsvector) matching stays symmetric."""
    normalizer = FarsflowNormalizer()

    index_result = await normalizer.normalize_index("من می روم")
    query_result = await normalizer.normalize_query("می روم")

    assert "می\u200cروم" in index_result
    assert query_result == "می\u200cروم"
