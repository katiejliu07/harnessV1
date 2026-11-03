import json
from independent_audit.saved_audit import ROOT
from independent_audit.scoring_v2 import correct,extract_answer


def test_verification_reference_not_mistaken_for_final_answer():
    text='To ensure our answer is correct, square 7:\n7 * 7 = 49.\n\nTherefore, the square root of 49 is 7.'
    assert extract_answer(text)=="7"
    assert not correct(text,"49")


def test_saved_main_final_answers_and_wrong_counterfactuals():
    rows=json.loads((ROOT/"results/activation_patching/quality_assessment_v8.json").read_text())["detailed_results"]
    assert all(correct(r[s]["text"],r["expected_answer"]) for r in rows for s in ["baseline","intervened"])
    assert not correct('Final answer: 81 divided by 9 equals 8.',"9")
    assert correct(r'Final answer: \boxed{3/4}',".75")
    assert correct('Final answer: 6.88e2',"688")
