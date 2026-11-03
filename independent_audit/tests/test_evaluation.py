import json
from pathlib import Path

from independent_audit.evaluation import correct, extract_answer, quality_gate
from independent_audit.saved_audit import original_function, ROOT


def test_saved_false_negatives():
    rows = json.loads((ROOT / "results/activation_patching/quality_assessment_v8.json").read_text())["detailed_results"]
    old = original_function("extract_answer")
    for index, expected, wrong in [(13, "9", "81"), (22, "28", "14")]:
        text = rows[index]["baseline"]["text"]
        assert old(text) == wrong
        assert extract_answer(text) == expected


def test_parser_not_expected_answer_search():
    assert not correct("Final answer: 81 divided by 9 equals 8.", "9")
    assert correct("Final answer: -3.50", "-3.5")
    assert correct("Final answer: 1,200 dollars.", "1200")
    assert correct(r"The answer is \boxed{3/4}.", ".75")
    assert extract_answer("I cannot determine it.") is None


def test_original_verdict_accepts_28_percent_regression():
    rows = [{"baseline": {"correct": True}, "intervened": {"correct": i < 18},
             "changes": {"token_reduction": 50, "marker_reduction": 4, "maintained_correctness": i < 18}}
            for i in range(25)]
    assert original_function("summarize_results")(rows)["verdict"] == "EFFECTIVE"
    assert not quality_gate([True]*25, [i < 18 for i in range(25)], 100, 50)["eligible_on_observed_sample"]


def test_shorter_but_slower_rejected():
    assert not quality_gate([True], [True], 1., 2.)["eligible_on_observed_sample"]


def test_all_main_saved_final_answers_regression():
    rows=json.loads((ROOT / "results/activation_patching/quality_assessment_v8.json").read_text())["detailed_results"]
    for r in rows:
        for side in ["baseline","intervened"]:
            assert correct(r[side]["text"],r["expected_answer"])
    # Explicitly retain a false intermediate statement in the quality audit.
    assert "Since 90 is less than 81" in rows[13]["baseline"]["text"]


def test_main_gold_answers_against_independent_arithmetic():
    from fractions import Fraction
    data=json.loads((ROOT/"results/activation_patching/quality_assessment_v8.json").read_text())["detailed_results"]
    computed=[7+5,20-8,6*4,5**2,Fraction(14,2),Fraction(100,4),180-60-60,3+4,7,13-5,
              9+8,15-6,7*8,Fraction(81,9),Fraction(20-5,3),Fraction(10+6,4),8*3,
              Fraction(120,3),50-18,8,25-6,7*5,Fraction(14,100)*200,Fraction(3,4)*32,180-90-45]
    assert [Fraction(r["expected_answer"]) for r in data]==computed
