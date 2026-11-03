"""Scalar scoring only: final-answer accuracy is NOT a reasoning/code-quality score."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"


def canonical(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return str(Fraction(value.replace(",", "").strip().rstrip(".,;:")))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def extract_answer(text: str) -> str | None:
    # Do not search for the expected answer: this parser must also reject wrong answers.
    # The original regex took the FIRST operand after "Final answer:". Restrict to
    # the LAST answer clause, then use its final scalar (including equations/units).
    clean = re.sub(r"\\(?:boxed|text|mathrm)\{([^{}]*)\}", r"\1", text)
    clean = clean.replace("**", "").replace("$", "")
    anchors = list(re.finditer(r"\b(?:final\s+answer(?:\s+is)?\s*[:=]?|(?:the\s+)?answer\s+is\s*:?|answer\s*:)\s*", clean, re.I))
    clause = clean[anchors[-1].end():] if anchors else clean
    # Ignore later explanatory sentences, but do not split decimal points.
    if anchors:
        clause = re.split(r"(?<=[.!?])\s+(?=[A-Z])", clause.strip(), maxsplit=1)[0]
    nums = re.findall(NUMBER, clause)
    return canonical(nums[-1]) if nums else None


def correct(text: str, expected: str) -> bool:
    return extract_answer(text) == canonical(expected)


def quality_gate(base_correct: list[bool], test_correct: list[bool], base_cost: float, test_cost: float) -> dict:
    if len(base_correct) != len(test_correct) or not base_correct:
        raise ValueError("Need nonempty paired results")
    regressions = sum(b and not p for b, p in zip(base_correct, test_correct))
    return {"observed_correctness_regressions": regressions,
            "total_cost_ratio": test_cost / base_cost if base_cost > 0 else None,
            "eligible_on_observed_sample": regressions == 0 and sum(test_correct) >= sum(base_correct) and test_cost < base_cost,
            "claim": "Observed-sample gate only; no population noninferiority or code completeness established."}
