from independent_audit.tasks import CODING,score_code


def test_correct_code_and_empty_inputs():
    implementations=[
      'def dedup(items):\n    return list(dict.fromkeys(items))',
      'def runs(values):\n    out=[]\n    for v in values:\n        if out and out[-1][0]==v:\n            out[-1]=(v,out[-1][1]+1)\n        else:\n            out.append((v,1))\n    return out',
      'def lower_bound(a,x):\n    lo,hi=0,len(a)\n    while lo<hi:\n        mid=(lo+hi)//2\n        if a[mid]<x: lo=mid+1\n        else: hi=mid\n    return lo',
      'def balanced(s):\n    stack=[]\n    pairs={")":"(","]":"[","}":"{"}\n    for c in s:\n        if c in "([{": stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop()!=pairs[c]: return False\n    return not stack',
    ]
    for task,code in zip(CODING,implementations):
        assert score_code('```python\n'+code+'\n```',task)["passed"]


def test_incomplete_wrong_and_unsafe_code_fail():
    assert not score_code('def dedup(items):\n    pass',CODING[0])["passed"]
    assert not score_code('import os\ndef dedup(items):\n    return []',CODING[0])["passed"]
    assert not score_code('def dedup(items):\n    items.clear()\n    return []',CODING[0])["passed"]
