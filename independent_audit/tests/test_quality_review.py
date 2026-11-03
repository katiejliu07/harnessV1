from independent_audit.quality_review import final_format,logarithmic_access_test
from independent_audit.tasks import CODING
from independent_audit.code_quality import evaluate_code,solution_source


def test_format_is_separate_from_correct_final_number():
    assert final_format("Some explanation.\nFinal answer: 12.")
    assert not final_format("The answer is 12.")
    assert not final_format("Final answer: 14 percent of 200 is 28.")


def test_linear_scan_rejected_even_when_all_small_function_tests_pass():
    linear='def lower_bound(a,x):\n    for i,v in enumerate(a):\n        if v>=x: return i\n    return len(a)'
    verdict=logarithmic_access_test(linear,CODING[2])
    assert verdict["basic"]["passed"]
    assert not verdict["passed"]
    binary='def lower_bound(a,x):\n    lo,hi=0,len(a)\n    while lo<hi:\n        mid=(lo+hi)//2\n        if a[mid]<x: lo=mid+1\n        else: hi=mid\n    return lo'
    assert logarithmic_access_test(binary,CODING[2])["passed"]


def test_permitted_import_and_examples_do_not_cause_false_failure():
    text='```python\nfrom itertools import groupby\ndef runs(values):\n    return [(k,len(list(g))) for k,g in groupby(values)]\nprint(runs([1,1,2]))\n```'
    assert evaluate_code(text,CODING[1])["passed"]
    assert not evaluate_code('import os\ndef dedup(items):\n    return list(dict.fromkeys(items))',CODING[0])["passed"]


def test_linear_scan_fails_supplemental_grader_and_no_import_constraint():
    text='def lower_bound(a,x):\n    for i,v in enumerate(a):\n        if v>=x: return i\n    return len(a)'
    result=evaluate_code(text,CODING[2])
    assert all(result["functional"])
    assert not result["passed"]
    assert not evaluate_code('from bisect import bisect_left\ndef lower_bound(a,x):\n    return bisect_left(a,x)',CODING[2])["passed"]


def test_return_contracts_not_relaxed_by_python_boolean_equality():
    boolean_as_integer='def balanced(s):\n    stack=[]\n    pairs={")":"(","]":"[","}":"{"}\n    for c in s:\n        if c in "([{": stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop()!=pairs[c]: return 0\n    return int(not stack)'
    assert not evaluate_code(boolean_as_integer,CODING[3])["passed"]
    mutable_pairs='from itertools import groupby\ndef runs(values):\n    return [[k,len(list(g))] for k,g in groupby(values)]'
    assert not evaluate_code(mutable_pairs,CODING[1])["passed"]


def test_unclosed_markdown_fence_does_not_erase_complete_function():
    text='A function:\n```python\ndef dedup(items):\n    return list(dict.fromkeys(items))\nprint(dedup([1,1])) # Example output:'
    assert evaluate_code(text,CODING[0])["passed"]
    # An incomplete Python expression remains a real executable failure. The
    # parser may accept absent Markdown punctuation, never repair program text.
    assert not evaluate_code('```python\ndef dedup(items):\n    return list(',CODING[0])["passed"]
