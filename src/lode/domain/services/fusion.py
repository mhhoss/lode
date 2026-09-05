from __future__ import annotations

from collections.abc import Sequence

from lode.domain.models import ChunkId, ChunkScores


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ChunkId]],
    rank_constant: int = 60,
) -> ChunkScores:
    """
    Pure Reciprocal Rank Fusion (RRF).

    - Combines multiple ranked lists.
    - Duplicate chunk ids inside a single ranking are ignored.
    """

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    scores: ChunkScores = {}

    for ranking in rankings:
        seen: set[ChunkId] = set()

        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (rank_constant + rank)
            )

    return scores

