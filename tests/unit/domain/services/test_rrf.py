from __future__ import annotations

import pytest

from lode.domain.services import reciprocal_rank_fusion


def test_returns_empty_scores_for_empty_rankings() -> None:
    """An empty input must produce an empty score mapping."""

    assert reciprocal_rank_fusion([]) == {}


def test_scores_single_ranking_correctly() -> None:
    """Scores must follow the Reciprocal Rank Fusion formula."""

    scores = reciprocal_rank_fusion(
        [
            ("a", "b", "c"),
        ]
    )

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63)


def test_accumulates_scores_across_multiple_rankings() -> None:
    """The same chunk must accumulate contributions from every ranking."""

    scores = reciprocal_rank_fusion(
        [
            ("a", "b"),
            ("b", "a"),
        ]
    )

    expected = (1 / 61) + (1 / 62)

    assert scores["a"] == pytest.approx(expected)
    assert scores["b"] == pytest.approx(expected)


def test_higher_rank_receives_higher_score() -> None:
    """Higher-ranked chunks must receive larger scores."""

    scores = reciprocal_rank_fusion(
        [
            ("first", "second"),
        ]
    )

    assert scores["first"] > scores["second"]


def test_chunk_present_in_multiple_rankings_scores_higher() -> None:
    """Appearing in multiple rankings should increase a chunk's score."""

    scores = reciprocal_rank_fusion(
        [
            ("shared", "only_dense"),
            ("shared", "only_sparse"),
        ]
    )

    assert scores["shared"] > scores["only_dense"]
    assert scores["shared"] > scores["only_sparse"]


def test_invalid_rank_constant_raises_value_error() -> None:
    """rank_constant must be strictly positive."""

    with pytest.raises(ValueError, match="rank_constant must be positive"):
        reciprocal_rank_fusion(
            [],
            rank_constant=0,
        )


def test_rrf_is_deterministic() -> None:
    """The same input must always produce identical scores."""

    rankings = [
        ("a", "b", "c"),
        ("c", "a"),
    ]

    first = reciprocal_rank_fusion(rankings)
    second = reciprocal_rank_fusion(rankings)

    assert first == second


def test_custom_rank_constant_changes_scores() -> None:
    """Changing rank_constant should affect the final scores."""

    default = reciprocal_rank_fusion(
        [("a",)],
    )

    custom = reciprocal_rank_fusion(
        [("a",)],
        rank_constant=10,
    )

    assert default["a"] != custom["a"]
    assert custom["a"] == pytest.approx(1 / 11)


def test_order_of_rankings_does_not_change_result() -> None:
    """RRF must be invariant to the order of rankings."""

    rankings1 = [
        ("a", "b"),
        ("b", "c"),
    ]

    rankings2 = [
        ("b", "c"),
        ("a", "b"),
    ]

    assert reciprocal_rank_fusion(rankings1) == reciprocal_rank_fusion(rankings2)


def test_empty_ranking_is_ignored() -> None:
    """Empty rankings should not affect the result."""

    scores = reciprocal_rank_fusion(
        [
            (),
            ("a", "b"),
            (),
        ]
    )

    assert set(scores) == {"a", "b"}


def test_many_rankings_accumulate_scores() -> None:
    """A chunk appearing in many rankings should receive a larger score."""

    rankings = [
        ("shared", f"x{i}")
        for i in range(100)
    ]

    scores = reciprocal_rank_fusion(rankings)

    assert scores["shared"] > scores["x0"]


def test_duplicate_chunk_in_same_ranking_is_counted_once() -> None:
    """
    Duplicate chunk ids inside a single ranking should not receive
    multiple contributions from that ranking.
    """

    scores = reciprocal_rank_fusion(
        [
            ("a", "a", "b"),
        ]
    )

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 63)

