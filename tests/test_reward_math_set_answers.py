"""Regression: the thousands-comma stripper must not corrupt set/tuple/list answers.

MATH-500 answers are often sets, ordered pairs, or comma-lists (``{2, 100}``,
``(5, 120)``, ``[1, 2, 250]``). The digit-grouping-comma stripper deletes a comma
between a digit and a run of exactly three digits, which — once the set delimiters
are gone — merges two elements into one number (``{2, 100}`` -> ``2100``), grading a
wrong answer correct (a false positive). The stripper must skip collection answers
while still folding genuine thousands separators. Also folds the Unicode degree glyph
``°`` like ``^\\circ`` / ``\\degree`` (issue #296).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import pytest

from trinity.orchestration import reward as R


# --- the false positive: a set/tuple/list is NOT its digit-concatenation ---


@pytest.mark.parametrize(
    "candidate,reference",
    [
        (r"\boxed{\{2, 100\}}", "2100"),      # set {2, 100} != 2100
        (r"\boxed{2100}", r"\{2, 100\}"),     # symmetric
        (r"\boxed{(5, 120)}", "5120"),        # ordered pair (5, 120) != 5120
        (r"\boxed{[1, 2, 250]}", "12250"),    # list [1, 2, 250] != 12250
        (r"\boxed{\{2, 100\}}", "2, 100"),    # concatenation only differs by the merge
    ],
)
def test_set_answer_is_not_its_digit_concatenation(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 0.0


def test_correct_set_answer_still_matches():
    # A genuinely-correct set answer must still grade 1.0.
    assert R.score_text("math500", r"\boxed{\{2, 100\}}", r"\{2, 100\}") == 1.0
    assert R.score_text("math500", r"\boxed{(5, 120)}", r"(5, 120)") == 1.0


# --- preserved: real thousands separators still fold (no regression on #141) ---


@pytest.mark.parametrize(
    "candidate,reference",
    [
        (r"\boxed{2,000}", "2000"),
        (r"\boxed{1,000,000}", "1000000"),
        (r"\boxed{2{,}048}", "2048"),         # LaTeX-grouped thousands
    ],
)
def test_real_thousands_still_grade_correct(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 1.0


def test_small_comma_list_is_still_rejected():
    # 1-2 digit elements never triggered the strip; must stay a false-answer reject.
    assert R.score_text("math500", r"\boxed{\{2, 10\}}", "210") == 0.0


# --- the Unicode degree glyph, folded like ^\circ / \degree ---


def test_unicode_degree_glyph_matches_plain():
    assert R.score_text("math500", r"\boxed{90°}", "90") == 1.0
    assert R.score_text("math500", r"\boxed{90^\circ}", "90") == 1.0   # LaTeX form (already worked)
    assert R.score_text("math500", r"\boxed{45°}", "90") == 0.0        # still a real mismatch


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
