"""
Pure Python recursive character text splitter.

Zero dependencies.
Works with normalized text from any preprocessing pipeline.
"""
from __future__ import annotations

from uuid import uuid4

from lode.domain.models import Document, DocumentChunk


class SimpleTextChunker:
    """
    Split text recursively using a hierarchy of separators while
    preserving contextual overlap between adjacent chunks.
    """

    _SEPARATORS = (
        "\n\n",
        "\n",
        ". ",
        "؟ ",
        "! ",
        "؛ ",
        "، ",
        " ",
        "",
    )

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be strictly less than chunk_size"
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap


    async def split(
        self,
        document: Document
    ) -> tuple[DocumentChunk, ...]:
        if not document.content.strip():
            return ()

        raw_chunks = self._recursive_split(
            document.content,
            self._SEPARATORS
        )

        return tuple(
            DocumentChunk(
                id=str(uuid4()),
                document_id=document.id,
                content=chunk.strip(),
                chunk_index=index,
                metadata=dict(document.metadata),
            )
            for index, chunk in enumerate(raw_chunks)
            if chunk.strip()
        )


    def _recursive_split(
        self,
        text: str,
        separators: tuple[str, ...],
    ) -> list[str]:
        """Split text recursively using progressively weaker separators."""
        separator = self._find_best_separator(
            text,
            separators,
        )

        parts = text.split(separator) if separator else [text]

        remaining = (
            separators[separators.index(separator) + 1 :]
            if separator in separators
            else ()
        )

        chunks: list[str] = []

        current_parts: list[str] = []
        current_length = 0
        separator_length = len(separator)

        for part in parts:
            part_length = len(part)

            if (
                current_length
                + part_length
                + separator_length
                <= self._chunk_size
            ):
                current_parts.append(part)
                current_length += part_length + separator_length
                continue

            if current_parts:
                chunk = self._flush_chunk(
                    chunks,
                    current_parts,
                    separator,
                )

                overlap = self._get_overlap(chunk)

                if overlap:
                    current_parts = [overlap, part]
                    current_length = (
                        len(overlap)
                        + part_length
                        + separator_length
                    )
                else:
                    current_parts = [part]
                    current_length = part_length

                continue

            if remaining:
                chunks.extend(
                    self._recursive_split_with_separators(
                        part,
                        remaining,
                    )
                )
            else:
                chunks.extend(
                    self._split_hard(part)
                )

        if current_parts:
            self._flush_chunk(
                chunks,
                current_parts,
                separator,
            )

        return chunks


    def _recursive_split_with_separators(
        self,
        text: str,
        separators: tuple[str, ...],
    ) -> list[str]:
        """Recursively split oversized text using progressively weaker separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining = separators[1:]

        result: list[str] = []

        for part in text.split(separator):
            if len(part) <= self._chunk_size:
                result.append(part)
            elif remaining:
                result.extend(
                    self._recursive_split_with_separators(
                        part,
                        remaining,
                    )
                )
            else:
                result.extend(
                    self._split_hard(part)
                )

        return result


    def _split_hard(
        self,
        text: str,
    ) -> list[str]:
        """
        Final fallback splitter.

        Splits by fixed-size windows while preserving configured overlap.
        """

        if len(text) <= self._chunk_size:
            return [text]

        step = self._chunk_size - self._chunk_overlap

        return [
            text[i : i + self._chunk_size]
            for i in range(
                0,
                len(text),
                step,
            )
        ]


    def _find_best_separator(
        self,
        text: str,
        separators: tuple[str, ...],
    ) -> str:
        """Return the strongest available separator that produces at least one valid chunk."""

        for separator in separators:
            if separator == "":
                return ""

            if separator not in text:
                continue

            parts = text.split(separator)

            if any(
                len(part) <= self._chunk_size
                for part in parts
            ):
                return separator

        return ""


    def _flush_chunk(
        self,
        chunks: list[str],
        parts: list[str],
        separator: str,
    ) -> str:
        """Append the accumulated chunk and return it."""

        chunk = separator.join(parts)
        chunks.append(chunk)
        return chunk


    def _get_overlap(
        self,
        text: str,
    ) -> str:
        """Extract overlap from the end of a chunk without breaking words."""

        if self._chunk_overlap <= 0:
            return ""

        overlap = text[-self._chunk_overlap :]

        for index, char in enumerate(overlap):
            if char in (" ", "\n", "."):
                return overlap[index + 1 :]

        return overlap

