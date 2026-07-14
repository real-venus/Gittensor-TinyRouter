"""Per-task binary reward checkers for TRINITY trajectories.

The reward is the fitness signal sep-CMA-ES optimizes (SPEC §0.3.6, §5.2): a
single terminal Bernoulli ``R(tau) in {0, 1}`` per atomic evaluation. This
module is the single source of truth for "is this trajectory's final answer
correct?", dispatched on ``Trajectory.task.benchmark``.

Supported benchmarks
--------------------
* ``math500`` / ``aime``
    Extract a ``\\boxed{...}`` answer (else the last number) from the final
    answer, normalize, and compare to ``task.answer``. Symbolic equality via
    ``sympy`` when importable, otherwise a numeric/string fallback.
* ``mmlu`` / ``gpqa``
    Extract a single multiple-choice letter ``A-D`` (robust to phrasings such
    as ``"the answer is (B)"``, ``"B)"``, ``"B."``) and compare to
    ``task.answer``.
* ``livecodebench`` / ``bigcodebench``
    Execute candidate code against the task's tests in a subprocess sandbox
    with a timeout (``run_pass_at_1``). Never ``exec`` untrusted code in
    process.

Design contract
---------------
Every checker is a *pure* function of its inputs (no global state, no network)
so each can be unit-tested with one known-correct and one known-wrong case
(smoke test S5). The public entrypoint is :func:`score`.

This module has **no torch / GPU dependency** and imports only the stdlib plus
the shared :mod:`trinity.types`. ``sympy`` is imported lazily and guarded so the
module loads on a machine without it.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from typing import Sequence

from trinity.types import Role, Trajectory

__all__ = [
    "score",
    "score_text",
    "committed_answer",
    "has_answer",
    "extract_boxed",
    "extract_last_number",
    "normalize_math_answer",
    "math_equal",
    "extract_choice_letter",
    "normalize_reference_letter",
    "extract_code",
    "run_pass_at_1",
    "MATH_BENCHMARKS",
    "CHOICE_BENCHMARKS",
    "CODE_BENCHMARKS",
    "resolve_benchmark",
]

# Benchmark routing tables. Keys are matched case-insensitively against
# ``Task.benchmark`` (which the dataset loaders set, e.g. "math500").
MATH_BENCHMARKS: frozenset[str] = frozenset({"math500", "math", "aime", "aime2025"})
CHOICE_BENCHMARKS: frozenset[str] = frozenset(
    {"mmlu", "mmlu_pro", "mmlu-pro", "gpqa", "gpqa-diamond", "gpqa_diamond"}
)

# Multiple-choice option letters, in order. MMLU/GPQA use A-D; MMLU-Pro uses up
# to A-J (issue #12). The choice extractor and reference normaliser both range
# over this, so widening it here widens both consistently.
_CHOICE_LETTERS: str = "ABCDEFGHIJ"
CODE_BENCHMARKS: frozenset[str] = frozenset(
    {"livecodebench", "lcb", "bigcodebench", "bigcode"}
)

# Some frozen hidden-benchmark items carry a versioned adapter *identity* as
# their benchmark instead of a bare family key: the LiveCodeBench v6 adapter
# serialises ``"livecodebench_v6"`` so the frozen item records which release
# produced it (see ``adapters.livecodebench.LiveCodeBenchV6Adapter``). That
# identity is not itself a dispatch key, so it must be mapped to the family its
# checker is registered under; otherwise ``score_text``/``has_answer`` treat
# every frozen v6 item as an unknown benchmark and raise. Kept as an explicit map
# (never fuzzy suffix-stripping) so a real benchmark can never be mis-routed.
_BENCHMARK_ALIASES: dict[str, str] = {
    "livecodebench_v6": "livecodebench",
}


def resolve_benchmark(benchmark: str) -> str:
    """Normalize a benchmark identifier to its dispatch key.

    Lower-cases and trims ``benchmark``, then maps a known versioned/adapter
    *identity* (see ``_BENCHMARK_ALIASES``) onto the ``MATH``/``CHOICE``/``CODE``
    dispatch key its checker is registered under. Unknown or already-canonical
    keys are returned unchanged, so this is a safe no-op for the four shipped
    benchmarks and for a genuinely unrecognized name (which still raises in
    :func:`score_text`).

    Args:
        benchmark: Raw benchmark identifier, e.g. ``"livecodebench_v6"`` or
            ``"MATH500"``.

    Returns:
        The canonical dispatch key, lower-cased and trimmed.
    """
    key = (benchmark or "").strip().lower()
    return _BENCHMARK_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
def score(traj: Trajectory) -> float:
    """Return the binary reward ``R(tau) in {0.0, 1.0}`` for a trajectory.

    Dispatches on ``traj.task.benchmark``. The candidate answer is taken from
    ``traj.final_answer`` (the post-processed output of the terminating turn,
    ``O_tau``). For code benchmarks the candidate is the code extracted from the
    final answer and run against ``task.answer`` (the test spec).

    Args:
        traj: A completed :class:`~trinity.types.Trajectory` whose
            ``final_answer`` is populated and whose ``task`` carries the
            reference answer / test spec.

    Returns:
        ``1.0`` if the final answer is judged correct, else ``0.0``.

    Raises:
        ValueError: If the task's benchmark is not recognized.
    """
    benchmark = (traj.task.benchmark or "").strip().lower()
    ref = traj.task.answer
    candidate = _committed_answer(benchmark, traj)
    return score_text(benchmark, candidate, ref)


def committed_answer(benchmark: str, traj: Trajectory) -> str:
    """Public alias of :func:`_committed_answer`.

    Exposed so a benchmark adapter can score a full trajectory through its own
    ``score_output`` (picking the committed answer with the same multi-turn rule
    the evaluator uses) instead of re-implementing the selection. Keeps the
    routed (TRINITY / random) and single-model scoring paths consistent.
    """
    return _committed_answer(benchmark, traj)


def _committed_answer(benchmark: str, traj: Trajectory) -> str:
    """Pick the text to score from a multi-turn trajectory.

    ``_final_answer`` (last Worker output) is often a verbose derivation with no
    cleanly-extractable answer, while an answer DID appear in an earlier turn. To
    avoid throwing away answers the system actually produced, score the most
    recent turn whose output yields an extractable answer for this task type;
    fall back to ``final_answer``. This applies equally to TRINITY and the random
    baseline (the single-model baseline is one turn, so it is unaffected) — a fair
    fix, not a thumb on the scale. See JOURNAL 2026-06-23 (MMLU extraction).

    Verifier turns are never eligible. A Verifier-ACCEPT run terminates *on* a
    Verifier turn (see :func:`~trinity.orchestration.session.run_trajectory`), and
    post-processing is pass-through (:mod:`trinity.roles.postprocess`), so the
    Verifier's ``processed_output`` keeps its full critique — which routinely
    mentions a choice letter or number it is only *discussing* ("the worker points
    at B ... VERDICT: ACCEPT"). Scoring that text would credit or penalise the run
    for the *checker's* words rather than an answer the solver committed. The
    scored answer is the last non-verifier ``O_k``, matching
    :func:`~trinity.orchestration.session._final_answer` and the
    :func:`_terminating_role` contract. Prefer the most recent Worker turn, then
    any other non-verifier turn.
    """
    key = (benchmark or "").strip().lower()
    final = traj.final_answer or ""
    if has_answer(key, final):
        return final

    turns = getattr(traj, "turns", None) or []
    # Worker turns are the answer-producing role; prefer them, then fall back to
    # any other non-verifier turn (e.g. a Thinker that stated the result).
    worker = _last_answerful_output(key, turns, role=Role.WORKER)
    if worker is not None:
        return worker
    other = _last_answerful_output(key, turns, role=None)
    if other is not None:
        return other
    return final


def _answerful(benchmark: str, tr) -> bool:
    """Whether turn ``tr`` may source a committed answer for ``benchmark``.

    The single predicate behind both committed-answer selection
    (:func:`_last_answerful_output`) and the HERO self-consistency vote
    (:func:`answerful_non_verifier_outputs`), so "which turns can carry the
    answer" is defined in exactly one place. A :attr:`~trinity.types.Role.VERIFIER`
    turn is never eligible — the checker *discusses* answers (its pass-through
    ``processed_output`` keeps the critique) but never commits one — and the turn
    must carry an extractable answer (:func:`has_answer`).
    """
    if getattr(tr, "role", None) == Role.VERIFIER:
        return False
    txt = getattr(tr, "processed_output", "") or ""
    return has_answer(benchmark, txt)


def _last_answerful_output(
    benchmark: str, turns: Sequence, *, role: Role | None
) -> str | None:
    """Return the newest turn output that carries an extractable answer.

    Scans ``turns`` newest-first and returns the first ``processed_output`` that
    :func:`_answerful` accepts for ``benchmark``.
    :attr:`~trinity.types.Role.VERIFIER` turns are always skipped — the Verifier
    checks the solution, it never sources the committed answer. When ``role`` is
    given only turns of that role are considered; when ``role`` is ``None`` every
    non-verifier turn is eligible.

    Args:
        benchmark: Benchmark identifier (case-insensitive), already lower-cased.
        turns: The trajectory's turns, oldest-first.
        role: Restrict to this role, or ``None`` for any non-verifier role.

    Returns:
        The matching ``processed_output``, or ``None`` when no eligible turn
        carries an extractable answer.
    """
    for tr in reversed(turns):
        if not _answerful(benchmark, tr):
            continue
        if role is not None and getattr(tr, "role", None) != role:
            continue
        return getattr(tr, "processed_output", "") or ""
    return None


def answerful_non_verifier_outputs(benchmark: str, turns: Sequence | None) -> list[str]:
    """Every non-verifier turn output carrying an extractable answer, in turn order.

    The population the HERO self-consistency proxy votes over
    (:func:`trinity.optim.fitness.hero_quality`). Shares :func:`_answerful` with
    committed-answer selection, so a Verifier's critique can never enter the vote —
    the same discipline :func:`_committed_answer` applies to the *reference* answer.

    Args:
        benchmark: Benchmark identifier (case-insensitive).
        turns: The trajectory's turns, oldest-first (``None`` is treated as empty).

    Returns:
        The matching ``processed_output`` strings, oldest-first (possibly empty).
    """
    key = (benchmark or "").strip().lower()
    return [
        getattr(tr, "processed_output", "") or ""
        for tr in (turns or [])
        if _answerful(key, tr)
    ]


def has_answer(benchmark: str, text: str) -> bool:
    """Return ``True`` iff ``text`` contains an extractable answer for ``benchmark``.

    This is the format-validity predicate used both for picking the committed
    answer out of a multi-turn trajectory (:func:`_committed_answer`) and for the
    ``format_bonus`` term of the *training-only* shaped fitness (see
    :mod:`trinity.optim.fitness`). It re-uses the same ``extract_*`` helpers that
    :func:`score` relies on, so "has an answer" stays consistent with "can be
    scored". It does **not** judge correctness — only whether an answer is
    present in a parseable form.

    Args:
        benchmark: Benchmark identifier (case-insensitive), e.g. ``"math500"``.
        text: Candidate model output to inspect.

    Returns:
        ``True`` if an answer of the expected shape is present, else ``False``.
        Unknown benchmarks return ``False`` (no shape to look for).
    """
    if not text:
        return False
    key = resolve_benchmark(benchmark)
    if key in CHOICE_BENCHMARKS:
        return extract_choice_letter(text) is not None
    if key in MATH_BENCHMARKS:
        return extract_boxed(text) is not None or extract_last_number(text) is not None
    if key in CODE_BENCHMARKS:
        return "```" in text or "def " in text or "import " in text
    return False


def score_text(benchmark: str, candidate: str, reference: object) -> float:
    """Pure core of :func:`score`, decoupled from the Trajectory container.

    Useful for unit tests (S5): feed a benchmark name, a candidate string, and
    a reference answer directly.

    Args:
        benchmark: Benchmark identifier (case-insensitive), e.g. ``"math500"``.
        candidate: The model's final answer text (or code, for code tasks).
        reference: The reference answer. For math/choice this is the gold
            string; for code it is the test spec consumed by
            :func:`run_pass_at_1` (a list of tests, or a dict with ``tests`` and
            optional ``timeout_s``).

    Returns:
        ``1.0`` for correct, else ``0.0``.

    Raises:
        ValueError: If ``benchmark`` is not recognized.
    """
    key = resolve_benchmark(benchmark)
    if key in MATH_BENCHMARKS:
        return 1.0 if _check_math(candidate, reference) else 0.0
    if key in CHOICE_BENCHMARKS:
        return 1.0 if _check_choice(candidate, reference) else 0.0
    if key in CODE_BENCHMARKS:
        return 1.0 if _check_code(candidate, reference) else 0.0
    raise ValueError(
        f"Unknown benchmark {benchmark!r}. "
        f"Known: math={sorted(MATH_BENCHMARKS)}, "
        f"choice={sorted(CHOICE_BENCHMARKS)}, code={sorted(CODE_BENCHMARKS)}."
    )


# ---------------------------------------------------------------------------
# Math: MATH500 / AIME
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str | None:
    r"""Extract the contents of the last ``\boxed{...}`` in ``text``.

    Handles nested braces by balanced-brace scanning (so ``\boxed{\frac{1}{2}}``
    returns ``\frac{1}{2}``). Returns the **last** boxed expression, since the
    final answer is conventionally boxed last.

    Args:
        text: Arbitrary model output that may contain LaTeX.

    Returns:
        The inner content of the last ``\boxed{...}`` (stripped), or ``None`` if
        no balanced ``\boxed{...}`` is present.
    """
    if not text:
        return None
    results: list[str] = []
    marker = r"\boxed"
    idx = 0
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            break
        brace = pos + len(marker)
        # Skip whitespace between \boxed and the opening brace.
        while brace < len(text) and text[brace] in " \t":
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            idx = pos + len(marker)
            continue
        depth = 0
        start = brace + 1
        i = brace
        end = -1
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end == -1:
            # Unbalanced; stop scanning further occurrences.
            break
        results.append(text[start:end].strip())
        idx = end + 1
    return results[-1] if results else None


def extract_last_number(text: str) -> str | None:
    """Extract the last numeric literal from ``text``.

    Used as a fallback when no ``\\boxed{...}`` answer is present. Recognizes
    integers, decimals, signed numbers, and thousands separators (commas are
    stripped). Trailing punctuation (a sentence-ending period) is not consumed
    as a decimal point.

    Args:
        text: Arbitrary model output.

    Returns:
        The last number as a string (commas removed), or ``None`` if no number
        is found.
    """
    if not text:
        return None
    # LaTeX digit grouping: "1{,}000" renders as "1,000". Normalize it to a bare
    # comma so the thousands-separator branch below reads it as one number instead
    # of splitting it into "1" and "000".
    text = text.replace("{,}", ",")
    # Match a simple fraction a/b FIRST (so "1/2" is kept whole, not read as "2"),
    # then decimals/integers like -1,234.56 or 42 or .5 ; require a digit somewhere.
    pattern = re.compile(
        r"-?\d+\s*/\s*-?\d+"
        r"|-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        r"|-?\.\d+"
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").replace(" ", "")


_FONT_COMMANDS = ("text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "boldsymbol")


def _unwrap_font_commands(s: str) -> str:
    """Peel LaTeX font/style wrappers using balanced-brace scanning.

    Font commands change only presentation, not value. Unlike the old single-level
    ``[^{}]*`` regex, this handles braced payloads such as ``\\mathbf{\\frac{1}{2}}``.
    Unbalanced wrappers are left untouched.
    """
    changed = True
    while changed:
        changed = False
        for cmd in _FONT_COMMANDS:
            marker = f"\\{cmd}"
            idx = 0
            while True:
                pos = s.find(marker, idx)
                if pos == -1:
                    break
                brace = pos + len(marker)
                while brace < len(s) and s[brace] in " \t":
                    brace += 1
                if brace >= len(s) or s[brace] != "{":
                    idx = pos + len(marker)
                    continue
                depth = 0
                start = brace + 1
                i = brace
                end = -1
                while i < len(s):
                    ch = s[i]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                    i += 1
                if end == -1:
                    idx = pos + len(marker)
                    continue
                inner = s[start:end]
                s = s[:pos] + inner + s[end + 1 :]
                changed = True
                idx = pos + len(inner)
    return s


def normalize_math_answer(ans: str | None) -> str:
    r"""Normalize a math answer string for robust comparison.

    Strips LaTeX wrappers and cosmetic tokens that never change the value:
    ``$``/``\(``/``\)``, ``\left``/``\right``, ``\!``/``\,``/``\;``/``\:``,
    ``\text{...}``, ``\%`` and trailing ``%``, ``^\circ``/``\degree``, a leading
    ``=``, surrounding ``\{...\}``, and outer whitespace. Collapses internal
    whitespace and lowercases. Converts ``a/b`` integer fractions and
    ``\frac{a}{b}`` to a canonical ``Fraction`` string when possible, and folds
    ``\sqrt{x}``/``\sqrt x`` to a canonical ``sqrt(x)`` token and the constant
    pi (``\pi``, the Unicode ``π``, and a plain ``pi``) to one canonical ``pi``
    token.

    Args:
        ans: Raw answer text (or ``None``).

    Returns:
        A normalized string suitable for exact comparison (empty string for
        ``None``).
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    # Drop a leading "answer:" style prefix.
    s = re.sub(r"^(the\s+)?(final\s+)?answer(\s+is)?\s*[:=]?\s*", "", s, flags=re.I)
    # Remove math-mode delimiters. Strip the escaped dollar ``\$`` BEFORE the bare
    # ``$``; the reverse order leaves a stray backslash ("\$18.90" -> "\18.90")
    # and turns a correct dollar answer into a false negative.
    for tok in (r"\$", "$", r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\(", r"\)"):
        s = s.replace(tok, "")
    # Unwrap LaTeX font/style commands to their content. These change only how the
    # answer looks, not its value, so \mathbf{5} must normalize to 5 exactly as
    # \text{5}/\mathrm{5} already do (otherwise a bold-formatted answer is a false
    # negative against a plain reference).
    s = _unwrap_font_commands(s)
    s = s.replace(r"\%", "").replace("%", "")
    # Degree symbol in either brace form: ``^\circ`` and ``^{\circ}``. The braced
    # form is common LaTeX and was previously left intact, so ``90^{\circ}`` never
    # matched a plain ``90`` (a false negative). A caret is required, so a bare
    # ``\circ`` (function composition) is untouched.
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = s.replace(r"\degree", "")
    # The Unicode degree glyph ``°`` (U+00B0) is the plain-text twin of ``^\circ`` /
    # ``\degree`` above; fold it too so ``90°`` matches a plain ``90`` (else a correct
    # angle answer written with the glyph is a false negative while the LaTeX form works).
    s = s.replace("°", "")
    s = s.replace(r"\$", "")
    s = s.strip()
    if s.startswith("="):
        s = s[1:].strip()
    # A set / tuple / list answer (e.g. "{2, 100}", "(5, 120)", "[1, 2, 3]") uses commas
    # as ELEMENT separators, not digit grouping. Detect it HERE, before the delimiters are
    # stripped below, so the thousands-comma strip further down does not merge two elements
    # into one number (e.g. "{2, 100}" -> "2100", a false positive against the scalar 2100).
    _is_collection = bool(re.fullmatch(r"\\?[\{\[(].*,.*[\})\]]\\?", s, re.DOTALL))
    # Strip a single outer pair of \{ \} or { }. The capture is LAZY so the
    # trailing optional backslash can consume a "\}" escape; a greedy ".*" eats it
    # first, leaving a stray backslash ("\{1,2\}" -> "1,2\") that fails to match a
    # plainly-braced reference. Set-notation answers (\boxed{\{1,2,3\}}) hit this.
    s = re.sub(r"^\\?\{(.*?)\\?\}$", r"\1", s).strip()
    # \frac{a}{b} -> a/b. The [dt]? matches the whole textstyle/displaystyle
    # fraction family — \frac, \dfrac and \tfrac all render the same value, so they
    # must normalize identically (else \tfrac{3}{4} is a false negative vs \dfrac).
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r"\1/\2", s)
    # \sqrt{x} -> sqrt(x), and the unbraced single-token \sqrt2 -> sqrt(2). Like the
    # \frac handling just above, the braced and bare forms render the SAME value, so
    # they must normalize identically (else \sqrt{2} is a false negative against a
    # reference written \sqrt2). Folding to sympy's ``sqrt(...)`` spelling also lets
    # the symbolic fallback bridge value-equal forms like 2\sqrt{3} vs 2sqrt(3); the
    # bare backslash of ``\sqrt`` otherwise makes parse_expr raise. Runs after \frac
    # so a fraction nested inside the radicand is already unwrapped.
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*([0-9a-zA-Z])", r"sqrt(\1)", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    # Canonicalize the constant pi to a bare ``pi`` token. Models and datasets
    # write the same value three ways — the LaTeX ``\pi`` command, the Unicode
    # glyph ``π`` (U+03C0), and a plain ``pi`` — so ``2\pi``, ``2π`` and ``2 pi``
    # must all normalize identically (else a correct pi-valued answer is a false
    # negative against a differently-spelled reference). ``pi`` is also the name
    # sympy binds to the constant, so the symbolic fallback can then prove e.g.
    # ``pi/2`` equal to ``\frac{\pi}{2}``. The negative lookahead keeps ``\pi`` a
    # whole command (never a prefix of a longer ``\pi...`` macro); capital ``\Pi``
    # (the product symbol) is intentionally left untouched.
    s = re.sub(r"\\pi(?![a-zA-Z])", "pi", s)
    s = s.replace("π", "pi")
    # The fraction normalizer wraps arbitrary operands, so ``\frac{\pi}{2}``
    # becomes ``(pi)/(2)`` while ``\pi/2`` becomes ``pi/2``. Remove only
    # standalone atomic operands adjacent to division; this keeps function-call
    # parentheses such as ``sqrt(2)`` intact without requiring optional sympy.
    s = re.sub(r"(^|/)\((pi|\d+)\)(?=/|$)", r"\1\2", s)
    s = re.sub(r"\s+", "", s)
    # LaTeX digit grouping "1{,}000" -> "1,000" so the comma-strip below removes it
    # (a boxed answer like \boxed{2{,}048} otherwise never matches "2048").
    s = s.replace("{,}", ",")
    # Strip digit-grouping commas ("2,000" -> "2000", "1,000,000" -> "1000000") so
    # a thousands-separated answer is not a false negative. Only a comma between a
    # digit and a group of exactly three digits at a non-digit boundary is removed,
    # leaving list-like "1,2,3" untouched. Matches extract_last_number, which
    # already drops these commas. Skipped for set/tuple/list answers (``_is_collection``)
    # whose element-separator commas look identical to a thousands group once the
    # delimiters are gone (e.g. "{2, 100}" would otherwise collapse to "2100").
    if not _is_collection:
        s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    s = s.lower()
    # Canonicalize a pure integer ratio a/b. A leading sign may sit OUTSIDE the
    # parentheses: a negated LaTeX fraction (\-frac{3}{4}) normalizes to
    # "-(3)/(4)", so the minus precedes the numerator's paren and must still be
    # read as -3/4 (else -\frac{3}{4} never matches -3/4 or -0.75).
    m = re.fullmatch(r"(-?)\(?(-?\d+)\)?/\(?(-?\d+)\)?", s)
    if m:
        try:
            num = int(m.group(2))
            if m.group(1) == "-":
                num = -num
            return str(Fraction(num, int(m.group(3))))
        except (ZeroDivisionError, ValueError):
            pass
    return s


def _as_number(s: str) -> float | None:
    """Best-effort parse of a normalized string to a float, else ``None``."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?)\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", s)
    if m:
        try:
            denom = float(m.group(3))
            if denom != 0.0:
                num = float(m.group(2))
                if m.group(1) == "-":
                    num = -num
                return num / denom
        except ValueError:
            return None
    return None


def math_equal(a: str | None, b: str | None, *, abs_tol: float = 1e-6) -> bool:
    """Compare two math answers for equality.

    Resolution order:
      1. Exact match after :func:`normalize_math_answer`.
      2. Numeric match within ``abs_tol`` -- an ABSOLUTE tolerance that only
         bridges rounded float representations of the *same* value (e.g.
         ``0.333333`` vs ``1/3``), never merging genuinely different numbers.
      3. Symbolic equality via ``sympy`` if it is importable (guarded).

    The tolerance is deliberately absolute, not magnitude-scaled. The previous
    ``rel_tol * max(1, |a|, |b|)`` threshold grew with the answer's size, so it
    reached ``>= 1`` once ``|value| >= 1e6`` and graded off-by-one (or larger)
    large integers as correct (issue #141). MATH500 contains such large-integer
    answers; AIME (0-999) masked the bug. An absolute tolerance keeps the same
    behaviour for the small/near-unit answers the bridge was written for while
    comparing large answers exactly.

    Args:
        a: First answer (typically the candidate).
        b: Second answer (typically the reference).
        abs_tol: Absolute tolerance for the numeric comparison.

    Returns:
        ``True`` if the two answers are judged equal.
    """
    na = normalize_math_answer(a)
    nb = normalize_math_answer(b)
    if na == nb and na != "":
        return True

    fa = _as_number(na)
    fb = _as_number(nb)
    if fa is not None and fb is not None:
        if math.isclose(fa, fb, rel_tol=0.0, abs_tol=abs_tol):
            return True

    return _sympy_equal(na, nb)


def _sympy_equal(a: str, b: str) -> bool:
    """Symbolic-equality fallback. Returns ``False`` if sympy is unavailable."""
    if not a or not b:
        return False
    try:  # guarded import: local machine may lack sympy
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
    except Exception:
        return False
    transformations = standard_transformations + (
        implicit_multiplication_application,
    )
    try:
        ea = parse_expr(a, transformations=transformations, evaluate=True)
        eb = parse_expr(b, transformations=transformations, evaluate=True)
        diff = sympy.simplify(ea - eb)
        return diff == 0
    except Exception:
        return False


def _check_math(candidate: str, reference: object) -> bool:
    """True iff the candidate's extracted answer equals the reference."""
    extracted = extract_boxed(candidate)
    if extracted is None:
        extracted = extract_last_number(candidate)
    if extracted is None:
        # Last resort: compare the whole (normalized) candidate.
        extracted = candidate

    ref_str = reference if isinstance(reference, str) else _ref_to_str(reference)
    # The reference itself may be boxed (datasets vary).
    ref_boxed = extract_boxed(ref_str)
    if ref_boxed is not None:
        ref_str = ref_boxed
    return math_equal(extracted, ref_str)


def _ref_to_str(reference: object) -> str:
    """Coerce a non-string reference (int/float/Fraction) to text."""
    if reference is None:
        return ""
    return str(reference)


# ---------------------------------------------------------------------------
# Multiple choice: MMLU / GPQA
# ---------------------------------------------------------------------------
# Match in priority order. Earlier patterns are more explicit / trustworthy.
# LaTeX font/emphasis wrappers a model may put around the answer letter. Unwrapping
# them before matching lets a boxed, formatted choice (``\boxed{\text{B}}``,
# ``\boxed{\textbf{D}}``, ``\mathbf{C}``) be read exactly as a bare ``B``/``D``/``C``
# — the same commands ``normalize_math_answer`` already strips on the math path
# (issue #12 widened choices to A-J but the extractor never saw the wrapped letter).
_CHOICE_FONT_CMD_RE = re.compile(
    r"\\(?:text|textbf|textit|textrm|emph|mathrm|mathbf|mathit|mathsf|mathtt|boldsymbol)"
    r"\s*\{([^{}]*)\}"
)


def _strip_choice_font_wrappers(text: str) -> str:
    """Unwrap LaTeX font/emphasis commands to their content (idempotent per pass).

    Applied repeatedly so a nested wrapper (``\\textbf{\\text{B}}``) fully collapses.
    """
    prev = None
    while prev != text:
        prev = text
        text = _CHOICE_FONT_CMD_RE.sub(r"\1", text)
    return text


# Unambiguous commitment phrasings. A letter carried by any of these IS the answer
# the model is asserting, so the one that occurs LAST in the text is the committed
# answer (a model may reason toward one choice in prose and then commit another with
# a different phrasing — most commonly interim prose + a final ``\boxed{...}``).
_COMMITTED_ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Require the captured letter to be followed by a delimiter or end-of-word,
    # so "the answer Beats..." does NOT match "B" (P2 review fix).
    re.compile(r"answer\s*(?:is|:)?\s*\(?\s*([A-J])\s*(?:[\).:]|\b)(?![A-Za-z])", re.I),
    re.compile(r"\\boxed\s*\{\s*\(?\s*([A-J])\s*\)?\s*\}", re.I),
    re.compile(r"\bfinal\s+answer\s*[:=]?\s*\(?\s*([A-J])(?![A-Za-z])", re.I),
)
# Weaker cues, kept as strictly lower tiers so they never override a committed answer:
# ``option B`` often *discusses* a choice ("option B is wrong"), and a bare ``B)`` line
# is usually part of an option list the model echoes back. Each is taken as its own
# last match, in priority order, only when no committed-answer pattern matched.
_FALLBACK_CHOICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\boption\s*\(?\s*([A-J])(?![A-Za-z])", re.I),
    re.compile(r"^\s*\(?\s*([A-J])\s*[\).:]", re.M),
)
# Kept for anything that referenced the combined set (order = priority).
_CHOICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    _COMMITTED_ANSWER_PATTERNS + _FALLBACK_CHOICE_PATTERNS
)


def extract_choice_letter(text: str) -> str | None:
    """Extract a single multiple-choice letter ``A``-``D`` from ``text``.

    Robust to common phrasings: ``"the answer is (B)"``, ``"Answer: C"``,
    ``"B)"``, ``"B."``, ``"\\boxed{D}"``, ``"Option A"``. Tries explicit
    answer-bearing patterns first; if none match, falls back to the **last**
    standalone capital ``A``-``D`` token in the text (final answers usually come
    last). Letters embedded in words (e.g. the ``A`` in ``"And"``) are excluded
    by requiring word boundaries / delimiters.

    Args:
        text: Arbitrary model output.

    Returns:
        The uppercase letter, or ``None`` if no choice can be identified.
    """
    if not text:
        return None
    # Unwrap LaTeX font/emphasis commands first, so a boxed but formatted letter
    # (``\boxed{\text{B}}``, ``\textbf{C}``) is matched exactly like a bare one.
    text = _strip_choice_font_wrappers(text)
    # Among the unambiguous commitment phrasings, the committed answer is the one
    # that occurs LAST in the text — a model may reason toward one choice and then
    # commit another with a different phrasing (most often interim prose + a final
    # ``\boxed{...}``). "Last committed wins" is honoured ACROSS these patterns, not
    # only within one, so pattern priority no longer lets an earlier "answer is B"
    # beat a later ``\boxed{D}``. Priority is only a tie-break for matches ending at
    # the same position. This mirrors the "final answers usually come last" contract
    # also honoured by extract_boxed and trinity.roles.verifier.parse_verdict.
    best: tuple[tuple[int, int], str] | None = None
    for priority, pat in enumerate(_COMMITTED_ANSWER_PATTERNS):
        for cm in pat.finditer(text):
            key = (cm.end(), -priority)
            if best is None or key > best[0]:
                best = (key, cm.group(1).upper())
    if best is not None:
        return best[1]
    # No committed-answer phrasing: fall back to the weaker cues, each as its own
    # last match in priority order. These never override a committed answer above.
    for pat in _FALLBACK_CHOICE_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            return matches[-1].group(1).upper()
    # Fallback (P2 review fix): only trust the LAST non-empty line, and only when
    # it is essentially just the letter (e.g. "B", "(C)", "D."). This avoids the
    # English article "A" in prose like "A nice approach" being read as a choice.
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        m = re.fullmatch(r"\(?\s*([A-J])\s*\)?[.:]?", line, re.I)
        if m:
            return m.group(1).upper()
        break  # only inspect the final non-empty line
    return None


def _check_choice(candidate: str, reference: object) -> bool:
    """True iff the extracted letter matches the reference letter."""
    got = extract_choice_letter(candidate)
    if got is None:
        return False
    ref = normalize_reference_letter(reference)
    if ref is None:
        return False
    return got == ref


def normalize_reference_letter(reference: object) -> str | None:
    """Coerce a reference answer to a single ``A``-``D`` letter.

    Accepts a letter string (``"B"``, ``"(B)"``) or a 0-based / 1-based integer
    index (``1`` -> ``"B"`` under 0-based; datasets vary, so a bare letter is
    preferred). Returns ``None`` if it cannot be resolved.
    """
    if reference is None:
        return None
    if isinstance(reference, str):
        letter = extract_choice_letter(reference)
        if letter is not None:
            return letter
        s = reference.strip().upper()
        return s if s in set(_CHOICE_LETTERS) else None
    if isinstance(reference, bool):
        return None
    if isinstance(reference, int):
        if 0 <= reference < len(_CHOICE_LETTERS):
            return _CHOICE_LETTERS[reference]
        return None
    return None


# ---------------------------------------------------------------------------
# Code: LiveCodeBench / BigCodeBench
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(
    r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def extract_code(text: str) -> str:
    """Extract a Python code block from model output.

    If the text contains one or more fenced code blocks (```` ```python ... ```
    ````), the **last** such block is returned (the final solution usually comes
    last). If no fence is present, the text is returned verbatim (stripped),
    assuming the whole output is code.

    Args:
        text: Model output that may wrap code in Markdown fences.

    Returns:
        The extracted source code (without the fence markers).
    """
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip("\n")
    return text.strip()


def _coerce_test_spec(reference: object) -> tuple[list, int, str | None]:
    """Normalize a code reference into ``(tests, timeout_s, fn_name)``.

    The ``Task.answer`` for code benchmarks may be:
      * a ``list`` of tests, or
      * a ``dict`` with key ``"tests"``, optional ``"timeout_s"``, and optional
        ``"fn_name"`` (the entry-point for LiveCodeBench *functional* tests), or
      * a JSON string encoding either of the above.

    Each test is one of:
      * a ``str`` of assert-based Python (executed after the candidate code), or
      * a ``dict`` ``{"stdin": str, "expected_stdout": str}`` for I/O tests, or a
        ``{"input": str, "output": str, "testtype": "functional"}`` call test, or
      * a 2-tuple/list ``(stdin, expected_stdout)``.
    """
    timeout_s = 10
    fn_name: str | None = None
    spec: object = reference
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except (json.JSONDecodeError, ValueError):
            spec = [spec]
    if isinstance(spec, dict):
        timeout_s = int(spec.get("timeout_s", timeout_s))
        raw_fn = spec.get("fn_name") or spec.get("func_name")
        fn_name = str(raw_fn) if raw_fn else None
        tests = spec.get("tests", [])
    else:
        tests = spec
    if tests is None:
        tests = []
    if not isinstance(tests, list):
        tests = [tests]
    return tests, timeout_s, fn_name


def _check_code(candidate: str, reference: object) -> bool:
    """True iff extracted code passes all tests in the reference spec."""
    code = extract_code(candidate)
    if not code.strip():
        return False
    tests, timeout_s, fn_name = _coerce_test_spec(reference)
    return run_pass_at_1(code, tests, timeout_s=timeout_s, fn_name=fn_name)


def run_pass_at_1(
    code: str, tests: Sequence, timeout_s: int = 10, *, fn_name: str | None = None
) -> bool:
    """Execute candidate ``code`` against ``tests`` in a subprocess sandbox.

    The candidate code is **never** executed in-process. Each invocation writes
    a temporary script and runs it with the current Python interpreter in a
    fresh subprocess with a wall-clock timeout. The candidate is judged to pass
    only if **every** test passes.

    Three test flavors are supported (they may be mixed in one list):

    * **assert-based** (``str``): arbitrary Python appended after the candidate
      code; a test passes if the script exits ``0`` with no exception. Use this
      for function-call style benchmarks (BigCodeBench).
    * **stdin/stdout** (``dict`` with ``"stdin"`` / ``"expected_stdout"`` or a
      ``(stdin, expected_stdout)`` pair): the candidate is run as a program, fed
      ``stdin`` on standard input, and its stdout is compared (whitespace-
      trimmed per line) to ``expected_stdout``. Use this for competitive-
      programming style benchmarks (LiveCodeBench stdin problems).
    * **functional** (``dict`` with ``"testtype": "functional"``): ``input`` holds
      the call arguments (one JSON/literal value per line) and ``output`` the
      expected return; the candidate is imported and ``fn_name`` (a top-level
      function or a ``Solution`` method) is called with the parsed arguments, and
      its return value is compared to the expected. Use this for LeetCode-style
      LiveCodeBench problems, which otherwise score 0 when run as stdin/stdout.

    Args:
        code: Candidate Python source (already fence-stripped).
        tests: Sequence of tests as described above.
        timeout_s: Per-test wall-clock timeout in seconds.
        fn_name: Entry-point name for functional tests (ignored otherwise).

    Returns:
        ``True`` iff the candidate passes all tests (and there is at least one
        test). An empty test list returns ``False`` (nothing was verified).
    """
    if not code.strip():
        return False
    if not tests:
        return False
    for test in tests:
        if not _run_one_test(code, test, timeout_s, fn_name=fn_name):
            return False
    return True


def _run_one_test(
    code: str, test: object, timeout_s: int, *, fn_name: str | None = None
) -> bool:
    """Run a single test in an isolated subprocess. Returns pass/fail."""
    stdin_data: str | None = None
    expected_stdout: str | None = None
    assert_block: str | None = None

    if isinstance(test, dict):
        testtype = str(test.get("testtype", "")).strip().lower()
        if testtype == "functional" and fn_name:
            # LeetCode-style call test: parse args, invoke fn_name, compare return.
            return _run_functional_test(
                code,
                str(test.get("input", test.get("stdin", ""))),
                str(test.get("output", test.get("expected_stdout", ""))),
                fn_name,
                timeout_s,
            )
        if (
            "stdin" in test
            or "input" in test
            or "expected_stdout" in test
            or "output" in test
        ):
            # dataset.py emits LiveCodeBench tests as {"input": ..., "output": ...};
            # accept both key conventions so stdin is never silently empty.
            stdin_data = str(test.get("stdin", test.get("input", "")))
            expected_stdout = str(
                test.get("expected_stdout", test.get("output", ""))
            )
        elif "assert" in test:
            assert_block = str(test["assert"])
        else:
            # Unknown dict shape — treat any "test"/"code" field as assert code.
            assert_block = str(test.get("test", test.get("code", "")))
    elif isinstance(test, (tuple, list)) and len(test) == 2:
        stdin_data = str(test[0])
        expected_stdout = str(test[1])
    elif isinstance(test, str):
        assert_block = test
    else:
        return False

    if assert_block is not None:
        script = code + "\n\n" + assert_block + "\n"
        return _exec_script(script, stdin_data="", timeout_s=timeout_s)

    # stdin/stdout test.
    ok, stdout = _exec_script_capture(
        code, stdin_data=stdin_data or "", timeout_s=timeout_s
    )
    if not ok:
        return False
    return _stdout_matches(stdout, expected_stdout or "")


def _stdout_matches(got: str, expected: str) -> bool:
    """Compare program output to expected, ignoring trailing whitespace."""
    got_lines = [ln.rstrip() for ln in got.replace("\r\n", "\n").rstrip().split("\n")]
    exp_lines = [
        ln.rstrip() for ln in expected.replace("\r\n", "\n").rstrip().split("\n")
    ]
    return got_lines == exp_lines


def _parse_functional_value(raw: str) -> object:
    """Parse a single LiveCodeBench functional value (a JSON or Python literal).

    Functional test inputs/outputs are stored as text. Prefer JSON (the format
    the ``code_generation_lite`` release uses), fall back to a Python literal, and
    finally to the raw stripped string so an unparseable value still round-trips.
    """
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    try:
        import ast

        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _parse_functional_args(raw: str) -> list:
    """Parse LiveCodeBench functional call arguments (one value per line)."""
    return [
        _parse_functional_value(line)
        for line in str(raw).split("\n")
        if line.strip() != ""
    ]


def _run_functional_test(
    code: str, raw_input: str, raw_expected: str, fn_name: str, timeout_s: int
) -> bool:
    """Score one LeetCode-style functional test by calling ``fn_name``.

    Arguments and the expected return are parsed on the parent side and embedded
    into the child script as Python literals (so the child never re-parses raw
    text). The child resolves ``fn_name`` as a top-level function or a
    ``Solution`` method, calls it with the parsed arguments, and asserts the
    return equals the expected value; the process exits ``0`` iff it matches.
    """
    args = _parse_functional_args(raw_input)
    expected = _parse_functional_value(raw_expected)
    harness = (
        f"\n\n_args = {args!r}\n"
        f"_expected = {expected!r}\n"
        f"_name = {str(fn_name)!r}\n"
        "if _name in globals() and callable(globals()[_name]):\n"
        "    _fn = globals()[_name]\n"
        "elif 'Solution' in globals() and hasattr(Solution, _name):\n"
        "    _fn = getattr(Solution(), _name)\n"
        "else:\n"
        "    raise SystemExit('functional entry point not found: ' + _name)\n"
        "_got = _fn(*_args)\n"
        "assert _got == _expected, 'got %r, expected %r' % (_got, _expected)\n"
    )
    return _exec_script(code + harness, stdin_data="", timeout_s=timeout_s)


def _sandbox_env() -> dict[str, str]:
    """Minimal environment for the child interpreter."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    # Preserve a HOME so libraries that need a writable dir do not crash.
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


def _exec_script(script: str, *, stdin_data: str, timeout_s: int) -> bool:
    """Run a script; pass iff it exits 0 within the timeout. No output check."""
    ok, _ = _exec_script_capture(script, stdin_data=stdin_data, timeout_s=timeout_s)
    return ok


def _exec_script_capture(
    script: str, *, stdin_data: str, timeout_s: int
) -> tuple[bool, str]:
    """Run a script in a subprocess and capture stdout.

    Args:
        script: The full Python source to execute.
        stdin_data: Data piped to the child's standard input.
        timeout_s: Wall-clock timeout in seconds.

    Returns:
        ``(ok, stdout)`` where ``ok`` is ``True`` iff the process exited with
        return code ``0`` (no exception/timeout) and ``stdout`` is the captured
        standard output (empty on failure).
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            tmp_path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=_sandbox_env(),
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired:
            return False, ""
        except (OSError, ValueError):
            return False, ""
        return (proc.returncode == 0), (proc.stdout or "")
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Convenience: keep Role import meaningful for downstream type checks.
# ---------------------------------------------------------------------------
def _terminating_role(traj: Trajectory) -> Role | None:
    """Return the role of the terminating turn, or ``None`` if no turns.

    Exposed for orchestration/debugging: a Verifier-ACCEPT terminated run ends
    on a :class:`~trinity.types.Role.VERIFIER` turn, but the scored answer is
    the last non-verifier ``O_k`` carried in ``final_answer``.
    """
    if not traj.turns:
        return None
    return traj.turns[-1].role
