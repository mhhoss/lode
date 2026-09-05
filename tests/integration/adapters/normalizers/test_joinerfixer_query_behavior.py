"""
Decision test: is JoinerFixer safe to apply to short, fragment-like
search queries — not just full sentences?

Expected values are traced directly from JoinerFixerProcessor's actual
regex rules (_RE_MI, _RE_NEMI, _RE_HA, _RE_HE_AM/EI/AND) — not assumed.
"""

from __future__ import annotations

import pytest
from farsflow import JoinerFixer, Pipeline


@pytest.fixture(scope="module")
def fixer() -> JoinerFixer:
    return Pipeline([
        JoinerFixer(),
    ])


@pytest.mark.parametrize(
    "raw_query, expected",
    [
        # می/نمی prefix — matches: token-start or ^, then space, then Persian char
        ("می روم", "می\u200cروم"),
        ("می خواهم", "می\u200cخواهم"),
        ("می دهید", "می\u200cدهید"),
        ("نمی دانم", "نمی\u200cدانم"),

        # ها suffix — matches: Persian char immediately before, space, "ها", boundary/$
        ("کفش ها", "کفش\u200cها"),
        ("کتاب ها", "کتاب\u200cها"),
        ("قیمت کفش ها", "قیمت کفش\u200cها"),

        # ه‌ام / ه‌ای / ه‌اند — stem of 2+ Persian chars ending in ه, then suffix
        ("خسته ام", "خسته\u200cام"),
        ("آماده ای", "آماده\u200cای"),
        ("رفته اند", "رفته\u200cاند"),

        # NOT handled by this processor — تر (comparative) has no rule at all.
        # This is documented, intentional scope, not a bug.
        ("بزرگ تر", "بزرگ تر"),
        ("ارزان تر", "ارزان تر"),

        # Single word, no context — no space follows, prefix rule can't match
        ("می", "می"),
        ("ها", "ها"),

        # No matching pattern at all
        ("کفش مشکی", "کفش مشکی"),

        # Already-correct ZWNJ input — must not be touched or doubled
        ("می\u200cروم", "می\u200cروم"),
    ],
)
def test_joiner_fixer_on_realistic_short_queries(
    fixer: JoinerFixer,
    raw_query: str,
    expected: str,
) -> None:
    result = fixer(raw_query)

    assert result == expected, (
        f"JoinerFixer produced {result!r} for query {raw_query!r}, "
        f"expected {expected!r}."
    )


def test_joiner_fixer_is_idempotent_on_queries(fixer: JoinerFixer) -> None:
    once = fixer("می روم")
    twice = fixer(once)
    assert once == twice

