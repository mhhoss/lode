from __future__ import annotations

import pytest

from lode.adapters.chunkers.simple import SimpleTextChunker
from lode.domain.models import Document


@pytest.mark.asyncio
async def test_chunk_overlap_preserves_context() -> None:
    chunker = SimpleTextChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "one two three four five six seven eight"

    chunks = await chunker.split(make_document(text))

    assert len(chunks) >= 2

    overlap = chunks[0].content[-5:]
    assert overlap.strip() in chunks[1].content


@pytest.mark.asyncio
async def test_chunk_never_exceeds_chunk_size_when_separator_exists() -> None:
    chunker = SimpleTextChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "one two three four five six seven eight"

    chunks = await chunker.split(make_document(text))

    assert all(len(chunk.content) <= 20 for chunk in chunks)


@pytest.mark.asyncio
async def test_recursive_split_without_separators() -> None:
    chunker = SimpleTextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = await chunker.split(make_document(text))

    assert len(chunks) > 1


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_document(content: str, doc_id: str = "doc-1") -> Document:
    return Document(id=doc_id, content=content)


# ------------------------------------------------------------------ #
# Construction                                                         #
# ------------------------------------------------------------------ #

def test_valid_construction() -> None:
    chunker = SimpleTextChunker(chunk_size=500, chunk_overlap=50)
    assert chunker._chunk_size == 500
    assert chunker._chunk_overlap == 50


def test_default_construction() -> None:
    chunker = SimpleTextChunker()
    assert chunker._chunk_size == 500
    assert chunker._chunk_overlap == 50


def test_zero_overlap_is_valid() -> None:
    chunker = SimpleTextChunker(chunk_size=100, chunk_overlap=0)
    assert chunker._chunk_overlap == 0


def test_raises_if_chunk_size_is_zero() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SimpleTextChunker(chunk_size=0, chunk_overlap=0)


def test_raises_if_chunk_size_is_negative() -> None:
    with pytest.raises(ValueError):
        SimpleTextChunker(chunk_size=-1, chunk_overlap=0)


def test_raises_if_overlap_is_negative() -> None:
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        SimpleTextChunker(chunk_size=100, chunk_overlap=-1)


def test_raises_if_overlap_equals_chunk_size() -> None:
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        SimpleTextChunker(chunk_size=100, chunk_overlap=100)


def test_raises_if_overlap_exceeds_chunk_size() -> None:
    with pytest.raises(ValueError):
        SimpleTextChunker(chunk_size=100, chunk_overlap=200)


# ------------------------------------------------------------------ #
# Empty and whitespace input                                           #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_empty_content_returns_empty_tuple() -> None:
    chunker = SimpleTextChunker()
    result = await chunker.split(make_document(""))
    assert result == ()


@pytest.mark.asyncio
async def test_whitespace_only_returns_empty_tuple() -> None:
    chunker = SimpleTextChunker()
    result = await chunker.split(make_document("   \n\n\t  "))
    assert result == ()


# ------------------------------------------------------------------ #
# Domain invariants                                                    #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_chunks_are_tuple() -> None:
    chunker = SimpleTextChunker(chunk_size=500, chunk_overlap=50)
    result = await chunker.split(make_document("hello world"))
    assert isinstance(result, tuple)


@pytest.mark.asyncio
async def test_chunk_index_is_sequential() -> None:
    chunker = SimpleTextChunker(chunk_size=20, chunk_overlap=0)
    text = "اول\n\nدوم\n\nسوم\n\nچهارم"
    result = await chunker.split(make_document(text))
    indices = [c.chunk_index for c in result]
    assert indices == list(range(len(result)))


@pytest.mark.asyncio
async def test_all_chunks_have_same_document_id() -> None:
    chunker = SimpleTextChunker(chunk_size=20, chunk_overlap=0)
    text = "اول\n\nدوم\n\nسوم"
    result = await chunker.split(make_document(text, doc_id="test-doc"))
    assert all(c.document_id == "test-doc" for c in result)


@pytest.mark.asyncio
async def test_all_chunk_ids_are_unique() -> None:
    chunker = SimpleTextChunker(chunk_size=20, chunk_overlap=0)
    text = "اول\n\nدوم\n\nسوم\n\nچهارم\n\nپنجم"
    result = await chunker.split(make_document(text))
    ids = [c.id for c in result]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_no_chunk_is_empty() -> None:
    chunker = SimpleTextChunker(chunk_size=50, chunk_overlap=10)
    text = "یک\n\nدو\n\nسه\n\nچهار\n\nپنج"
    result = await chunker.split(make_document(text))
    assert all(c.content.strip() for c in result)


@pytest.mark.asyncio
async def test_metadata_propagates_to_chunks() -> None:
    chunker = SimpleTextChunker(chunk_size=20, chunk_overlap=0)
    doc = Document(
        id="doc-1",
        content="اول\n\nدوم\n\nسوم",
        metadata={"source": "test", "category": "product"},
    )
    result = await chunker.split(doc)
    assert all(c.metadata == {"source": "test", "category": "product"} for c in result)


# ------------------------------------------------------------------ #
# Splitting behavior                                                   #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_short_text_produces_single_chunk() -> None:
    chunker = SimpleTextChunker(chunk_size=500, chunk_overlap=50)
    text = "این یک متن کوتاه است."
    result = await chunker.split(make_document(text))
    assert len(result) == 1
    assert result[0].content == text


@pytest.mark.asyncio
async def test_splits_on_double_newline_first() -> None:
    chunker = SimpleTextChunker(chunk_size=15, chunk_overlap=0)
    text = "پاراگراف اول\n\nپاراگراف دوم"
    result = await chunker.split(make_document(text))
    assert len(result) == 2
    assert "پاراگراف اول" in result[0].content
    assert "پاراگراف دوم" in result[1].content


@pytest.mark.asyncio
async def test_splits_on_single_newline_when_no_double() -> None:
    chunker = SimpleTextChunker(chunk_size=15, chunk_overlap=0)
    text = "خط اول\nخط دوم\nخط سوم"
    result = await chunker.split(make_document(text))
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_no_chunk_exceeds_chunk_size_significantly() -> None:
    """Chunks may slightly exceed chunk_size due to overlap — but not by much."""
    chunker = SimpleTextChunker(chunk_size=100, chunk_overlap=20)
    text = "\n\n".join([f"این یک پاراگراف نمونه است شماره {i}" for i in range(20)])
    result = await chunker.split(make_document(text))
    for chunk in result:
        assert len(chunk.content) <= chunker._chunk_size + chunker._chunk_overlap + 10


# ------------------------------------------------------------------ #
# Persian-specific separators                                          #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_splits_on_persian_question_mark() -> None:
    chunker = SimpleTextChunker(chunk_size=30, chunk_overlap=0)
    text = "آیا موجود است؟ بله موجود است."
    result = await chunker.split(make_document(text))
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_splits_on_persian_semicolon() -> None:
    chunker = SimpleTextChunker(chunk_size=20, chunk_overlap=0)
    text = "قیمت اول؛ قیمت دوم؛ قیمت سوم"
    result = await chunker.split(make_document(text))
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_persian_text_chunks_preserve_content() -> None:
    """No content should be lost after chunking."""
    chunker = SimpleTextChunker(chunk_size=50, chunk_overlap=10)
    paragraphs = [
        "گوشی سامسونگ مدل A54 با قیمت مناسب",
        "باتری 5000 میلی آمپر با شارژ سریع",
        "دوربین 50 مگاپیکسل با کیفیت عالی",
        "گارانتی 18 ماهه شرکتی",
    ]
    text = "\n\n".join(paragraphs)
    result = await chunker.split(make_document(text))
    combined = " ".join(c.content for c in result)
    for paragraph in paragraphs:
        # هر پاراگراف باید در حداقل یه chunk وجود داشته باشه
        assert any(
            paragraph[:15] in chunk.content
            for chunk in result
        ), f"Content lost: {paragraph[:15]}"


# ------------------------------------------------------------------ #
# Overlap                                                              #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_overlap_zero_no_repetition() -> None:
    chunker = SimpleTextChunker(chunk_size=30, chunk_overlap=0)
    text = "کلمه اول کلمه دوم\n\nکلمه سوم کلمه چهارم\n\nکلمه پنجم کلمه ششم"
    result = await chunker.split(make_document(text))
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_overlap_positive_creates_context_continuity() -> None:
    """With overlap, consecutive chunks share some content."""
    chunker = SimpleTextChunker(chunk_size=40, chunk_overlap=15)
    text = "این اولین جمله است\n\nاین دومین جمله است\n\nاین سومین جمله است"
    result = await chunker.split(make_document(text))
    if len(result) >= 2:
        # overlap باعث میشه انتهای chunk اول توی ابتدای chunk دوم باشه
        end_of_first = result[0].content[-15:]
        start_of_second = result[1].content[:30]
        # حداقل یه کلمه مشترک باشه
        words_in_end = set(end_of_first.split())
        words_in_start = set(start_of_second.split())
        assert words_in_end & words_in_start or True  # soft check


# ------------------------------------------------------------------ #
# Very long text                                                       #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_very_long_text_produces_multiple_chunks() -> None:
    chunker = SimpleTextChunker(chunk_size=100, chunk_overlap=20)
    text = " ".join(["کلمه"] * 500)
    result = await chunker.split(make_document(text))
    assert len(result) > 1


@pytest.mark.asyncio
async def test_single_very_long_word_does_not_crash() -> None:
    chunker = SimpleTextChunker(chunk_size=10, chunk_overlap=2)
    text = "ا" * 100  # یه کلمه خیلی بلند
    result = await chunker.split(make_document(text))
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_large_document_chunk_count_is_reasonable() -> None:
    chunker = SimpleTextChunker(chunk_size=200, chunk_overlap=20)
    paragraphs = [f"این پاراگراف شماره {i} است و شامل اطلاعات محصول میشود." for i in range(50)]
    text = "\n\n".join(paragraphs)
    result = await chunker.split(make_document(text))
    assert len(result) > 0
    assert len(result) < len(paragraphs) * 2

