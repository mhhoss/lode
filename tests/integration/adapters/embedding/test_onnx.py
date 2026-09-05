from pathlib import Path

import numpy as np
import pytest

from lode.adapters.embedders import OnnxTextEmbeddingAdapter
from lode.domain.exceptions import EmbeddingError

MODEL_DIR = Path.home() / ".cache/lode/models/multilingual-e5-small"


@pytest.fixture(scope="session")
def adapter() -> OnnxTextEmbeddingAdapter:
    return OnnxTextEmbeddingAdapter(
        model_dir=MODEL_DIR,
    )


async def test_embed_document(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    embeddings = await adapter.embed(
        (
            "Artificial intelligence is transforming software.",
        ),
        mode="document",
    )

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


async def test_embed_query(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    embeddings = await adapter.embed(
        (
            "What is artificial intelligence?",
        ),
        mode="query",
    )

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


async def test_embed_multiple_texts(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    embeddings = await adapter.embed(
        (
            "hello",
            "this is a longer sentence",
            "this is a much much much longer paragraph than the previous ones",
        ),
        mode="document",
    )

    assert len(embeddings) == 3

    for embedding in embeddings:
        assert len(embedding) == 384


async def test_embed_empty_input(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    embeddings = await adapter.embed(
        (),
        mode="document",
    )

    assert embeddings == ()


async def test_invalid_mode(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    with pytest.raises(EmbeddingError):
        await adapter.embed(
            (
                "hello",
            ),
            mode="invalid",  # type: ignore[arg-type]
        )


async def test_long_document(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    text = "hello " * 200

    embeddings = await adapter.embed(
        (
            text,
        ),
        mode="document",
    )

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


async def test_embedding_is_l2_normalized(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:

    embedding = (
        await adapter.embed(
            (
                "Artificial intelligence",
            ),
            mode="document",
        )
    )[0]

    norm = np.linalg.norm(embedding)

    assert norm == pytest.approx(
        1.0,
        abs=1e-5,
    )


async def test_embedding_is_deterministic(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:

    first = await adapter.embed(
        (
            "Hello world",
        ),
        mode="document",
    )

    second = await adapter.embed(
        (
            "Hello world",
        ),
        mode="document",
    )

    np.testing.assert_allclose(
        first,
        second,
        atol=1e-6,
    )


async def test_query_and_document_embeddings_are_different(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:

    query = (
        await adapter.embed(
            (
                "machine learning",
            ),
            mode="query",
        )
    )[0]

    document = (
        await adapter.embed(
            (
                "machine learning",
            ),
            mode="document",
        )
    )[0]

    assert not np.allclose(
        query,
        document,
    )


async def test_persian_text_embedding(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:

    embedding = (
        await adapter.embed(
            (
                "هوش مصنوعی آینده نرم‌افزار را تغییر می‌دهد.",
            ),
            mode="document",
        )
    )[0]

    assert len(embedding) == 384


async def test_text_exceeding_max_sequence_length_is_truncated_not_crashed(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    # Repeat a distinct word ~1000 times — comfortably past any 512-token limit
    text = "کلمه " * 1000

    embeddings = await adapter.embed((text,), mode="document")

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


async def test_truncated_text_produces_consistent_embedding(
    adapter: OnnxTextEmbeddingAdapter,
) -> None:
    """
    Text beyond max_sequence_length should be truncated deterministically —
    two inputs that only differ *after* the truncation point must embed
    identically, proving truncation (not silent failure) is what's happening.
    """
    base = "کلمه " * 600  # comfortably past the 512-token limit
    text_a = base + "الف"
    text_b = base + "ب"

    embedding_a = (await adapter.embed((text_a,), mode="document"))[0]
    embedding_b = (await adapter.embed((text_b,), mode="document"))[0]

    np.testing.assert_allclose(embedding_a, embedding_b, atol=1e-6)
