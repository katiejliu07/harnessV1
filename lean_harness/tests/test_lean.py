import hashlib
import json
from pathlib import Path

import pytest

from independent_audit.code_quality import evaluate_code
from lean_harness.benchmark import compare
from lean_harness.runtime import PROFILES,make_prompt,recommended_profile
from lean_harness.tasks import DEV,VALIDATION,DEV_CODE,VALIDATION_CODE


# Hand-written reference implementations, independent of lean_harness.differential.REFERENCES.
FRESH_REFS={
    'interleave':'def interleave(a,b):\n    out=[]\n    for i in range(max(len(a),len(b))):\n        if i<len(a): out.append(a[i])\n        if i<len(b): out.append(b[i])\n    return out',
    'clip_values':'def clip_values(values,lo,hi):\n    return [max(lo,min(hi,x)) for x in values]',
    'adjacent_differences':'def adjacent_differences(values):\n    return [b-a for a,b in zip(values,values[1:])]',
    'even_first':'def even_first(values):\n    return [x for x in values if x%2==0]+[x for x in values if x%2!=0]',
    'rotate_right':'def rotate_right(values,k):\n    if not values: return []\n    k%=len(values)\n    return values[-k:]+values[:-k]',
    'merge_sorted':'def merge_sorted(a,b):\n    return sorted(a+b)',
    'prefix_sums':'def prefix_sums(values):\n    out=[]\n    total=0\n    for x in values:\n        total+=x\n        out.append(total)\n    return out',
    'moving_sums':'def moving_sums(values,k):\n    if k<=0 or k>len(values): return []\n    return [sum(values[i:i+k]) for i in range(len(values)-k+1)]',
    'expand_runs':'def expand_runs(pairs):\n    return [v for v,n in pairs for i in range(n)]',
    'nonzero_indices':'def nonzero_indices(values):\n    return [i for i,x in enumerate(values) if x!=0]',
    'flatten_once':'def flatten_once(rows):\n    return [x for row in rows for x in row]',
    'keep_last':'def keep_last(items):\n    return [x for i,x in enumerate(items) if x not in items[i+1:]]',
}

CONFIRM_REFS={
    'chunked':'def chunked(values,size):\n    return [values[i:i+size] for i in range(0,len(values),size)]',
    'running_max':'def running_max(values):\n    out=[]\n    for x in values:\n        out.append(x if not out or x>out[-1] else out[-1])\n    return out',
    'pairwise_swap':'def pairwise_swap(values):\n    out=list(values)\n    for i in range(0,len(out)-1,2):\n        out[i],out[i+1]=out[i+1],out[i]\n    return out',
    'top_two':'def top_two(values):\n    return sorted(set(values),reverse=True)[:2]',
    'intersperse':'def intersperse(values,sep):\n    out=[]\n    for i,x in enumerate(values):\n        if i: out.append(sep)\n        out.append(x)\n    return out',
    'drop_every':'def drop_every(values,n):\n    if n<=0: return list(values)\n    return [x for i,x in enumerate(values,1) if i%n]',
    'indices_of_max':'def indices_of_max(values):\n    return [i for i,x in enumerate(values) if x==max(values)]',
    'word_lengths':'def word_lengths(text):\n    return [len(w) for w in text.split()]',
}


HARD_REFS={
    'merge_intervals':'def merge_intervals(intervals):\n    pts=sorted([list(x) for x in intervals])\n    out=[]\n    for a,b in pts:\n        if out and a<=out[-1][1]:\n            out[-1][1]=max(out[-1][1],b)\n        else:\n            out.append([a,b])\n    return out',
    'spiral_order':'def spiral_order(matrix):\n    out=[]\n    if not matrix or not matrix[0]: return out\n    top,bottom,left,right=0,len(matrix)-1,0,len(matrix[0])-1\n    while top<=bottom and left<=right:\n        for j in range(left,right+1): out.append(matrix[top][j])\n        for i in range(top+1,bottom+1): out.append(matrix[i][right])\n        if top<bottom:\n            for j in range(right-1,left-1,-1): out.append(matrix[bottom][j])\n        if left<right:\n            for i in range(bottom-1,top,-1): out.append(matrix[i][left])\n        top+=1; bottom-=1; left+=1; right-=1\n    return out',
    'top_k_frequent':'def top_k_frequent(words,k):\n    if k<=0: return []\n    counts={}\n    for w in words: counts[w]=counts.get(w,0)+1\n    return sorted(counts,key=lambda w:(-counts[w],w))[:k]',
    'next_greater':'def next_greater(values):\n    out=[None]*len(values)\n    stack=[]\n    for i,x in enumerate(values):\n        while stack and values[stack[-1]]<x:\n            out[stack.pop()]=x\n        stack.append(i)\n    return out',
    'primes_upto':'def primes_upto(n):\n    if n<2: return []\n    sieve=[True]*(n+1)\n    sieve[0]=sieve[1]=False\n    for i in range(2,int(n**0.5)+1):\n        if sieve[i]:\n            for j in range(i*i,n+1,i): sieve[j]=False\n    return [i for i,ok in enumerate(sieve) if ok]',
    'rle_encode':'def rle_encode(text):\n    out=[]\n    i=0\n    while i<len(text):\n        j=i\n        while j<len(text) and text[j]==text[i]: j+=1\n        out.append([text[i],j-i])\n        i=j\n    return out',
    'sliding_max':'def sliding_max(values,k):\n    if k<=0 or k>len(values): return []\n    out=[]\n    for i in range(len(values)-k+1):\n        best=values[i]\n        for x in values[i:i+k]:\n            if x>best: best=x\n        out.append(best)\n    return out',
    'tokenize_expr':'def tokenize_expr(text):\n    out=[]\n    i=0\n    while i<len(text):\n        c=text[i]\n        if c.isdigit():\n            j=i\n            while j<len(text) and text[j].isdigit(): j+=1\n            out.append(int(text[i:j]))\n            i=j\n            continue\n        if c in "+-*/()": out.append(c)\n        i+=1\n    return out',
}


def test_full_task_and_constraints_survive_every_wrapper():
    task='Write f(x). Do not mutate x. Keep required comments. Explain the boundary cases.'
    for profile in PROFILES:
        for kind in ['scalar','code']:
            assert task in make_prompt(task,kind,profile)
        assert make_prompt(task,'general',profile)==task
    with pytest.raises(ValueError):
        make_prompt('', 'code','lean')


def test_no_item_regressions_hidden_by_shorter_output_or_shared_truncation():
    def row(i,tokens,seconds,passed,eos):
        return dict(task={'id':str(i)},total_tokens=tokens,total_seconds=seconds,
                    task_checks_passed=passed,checks_and_eos=passed and eos)
    b=[row(0,100,2,True,False),row(1,100,2,True,True)]
    p=[row(0,10,.2,False,False),row(1,10,.2,True,True)]
    result=compare(b,p)
    assert not result['eligible'] and result['task_check_regressions']==['0']
    assert result['checks_and_eos_regressions']==[]
    with pytest.raises(ValueError,match='unique'):
        compare([b[0],b[0]],p)


def test_policy_requires_matching_backend_and_inference_source(tmp_path,monkeypatch):
    from lean_harness import runtime
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    source=tmp_path/'runtime.py'
    source.write_text('measured source')
    env={'device':'mps','dtype':'torch.float16','model_revision':runtime.REVISION}
    policy={'environment':env,'kinds':{'code':{'profile':'terse','promoted':True}},
            'inference_source_sha256':{'runtime.py':hashlib.sha256(source.read_bytes()).hexdigest()}}
    path=tmp_path/'policy.json'
    path.write_text(json.dumps(policy))
    assert recommended_profile('code',path,environment=env)=='terse'
    assert recommended_profile('code',path,environment={**env,'device':'cuda'})=='legacy'
    source.write_text('unmeasured source')
    assert recommended_profile('code',path,environment=env)=='legacy'


def test_fresh_sets_disjoint_and_original_questions_are_not_validation():
    dev={t['prompt'] for t in DEV}
    val={t['prompt'] for t in VALIDATION}
    assert len(dev)==16 and len(val)==32 and not dev.intersection(val)
    root=Path(__file__).resolve().parents[2]
    old=[json.loads(s)['task']['prompt'] for s in (root/'independent_audit/results/mps_torch271_full/benchmark.jsonl').read_text().splitlines()]
    assert not (dev|val).intersection(old)


def test_fresh_code_gold_cases_and_safe_grader_support():
    from lean_harness.quality_stress import EXTRA
    # Independent reference implementations check hand-authored gold cases before
    # model generation; do not change these cases in response to model failures.
    refs=FRESH_REFS
    for task in DEV_CODE+VALIDATION_CODE:
        verdict=evaluate_code(refs[task['function']],{**task,'cases':task['cases']+EXTRA.get(task['function'],[])})
        assert verdict['passed'],(task['id'],verdict)
    task=next(t for t in VALIDATION_CODE if t['function']=='keep_last')
    assert not evaluate_code('def keep_last(items):\n    return list(dict.fromkeys(items))',task)['passed']


def test_system_free_profiles_drop_only_the_default_system_turn():
    from transformers import AutoTokenizer
    from lean_harness.runtime import SYSTEM_FREE,render
    tok=AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[2]/'.audit-model',local_files_only=True)
    task='Write f(x). Keep required comments.'
    legacy=render(tok,task,'legacy')[0].tolist()
    system=tok('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n',add_special_tokens=False)['input_ids']
    assert legacy[:len(system)]==system
    for profile in SYSTEM_FREE:
        ids=render(tok,task,profile)[0].tolist()
        # identical user/assistant framing and text, minus exactly the system turn
        assert ids==legacy[len(system):]
        assert task in tok.decode(ids)


def test_confirmation_set_is_fresh_and_its_gold_cases_are_right():
    from lean_harness.confirm_tasks import CONFIRMATION,CONFIRM_CODE
    conf={t['prompt'] for t in CONFIRMATION}
    root=Path(__file__).resolve().parents[2]
    old={json.loads(s)['task']['prompt'] for s in (root/'independent_audit/results/mps_torch271_full/benchmark.jsonl').read_text().splitlines()}
    assert len(conf)==len(CONFIRMATION)==56
    assert not conf & ({t['prompt'] for t in DEV+VALIDATION}|old)
    # Hand-written references check the gold cases before any model output exists.
    refs=CONFIRM_REFS
    for task in CONFIRM_CODE:
        verdict=evaluate_code(refs[task['function']],task)
        assert verdict['passed'],(task['id'],verdict)
    # and a plausible wrong answer is caught
    task=next(t for t in CONFIRM_CODE if t['function']=='top_two')
    assert not evaluate_code('def top_two(values):\n    return sorted(values,reverse=True)[:2]',task)['passed']


def test_differential_references_agree_with_independent_implementations():
    from lean_harness.confirm_tasks import CONFIRM_CODE
    from lean_harness.differential import REFERENCES,cases,differential
    from lean_harness.hard_tasks import HARD_CODE
    tasks=DEV_CODE+VALIDATION_CODE+CONFIRM_CODE+HARD_CODE
    assert set(REFERENCES)=={t['function'] for t in tasks}
    for task in tasks:
        name=task['function']
        # the generated cases are deterministic, so a reference bug can't hide behind luck
        assert cases(name)==cases(name)
        # two independent implementations must agree on every hand and random case
        src={**FRESH_REFS,**CONFIRM_REFS,**HARD_REFS}[name]
        row={'task':task,'text':src,'profile':'reference','task_checks_passed':True}
        verdict=differential(row)
        assert verdict['differential_passed'],(name,verdict)
    # a plausible wrong program (negative k rotates the wrong way) is caught
    task=next(t for t in VALIDATION_CODE if t['function']=='rotate_right')
    wrong='def rotate_right(values,k):\n    if not values: return []\n    k=abs(k)%len(values)\n    return values[-k:]+values[:-k] if k else list(values)'
    assert not differential({'task':task,'text':wrong,'profile':'x'})['differential_passed']


def test_error_bars():
    from lean_harness.stats import mcnemar_exact,paired,undetectable_loss_rate,wilson
    lo,hi=wilson(24,24)
    assert hi==1 and 0.85<lo<0.87
    assert mcnemar_exact(0,0)==1 and abs(mcnemar_exact(0,6)-0.03125)<1e-12
    assert abs(undetectable_loss_rate(24)-0.1173)<1e-3
    row=lambda i,ok:{'task':{'id':i},'checks_and_eos':ok}
    p=paired([row('a',True),row('b',False)],[row('a',True),row('b',True)])
    assert (p['candidate_wins'],p['candidate_losses'])==(1,0)


def test_hard_set_is_fresh_and_well_formed():
    from lean_harness.confirm_tasks import CONFIRMATION
    from lean_harness.hard_tasks import HARD
    root=Path(__file__).resolve().parents[2]
    old={json.loads(s)['task']['prompt'] for s in (root/'independent_audit/results/mps_torch271_full/benchmark.jsonl').read_text().splitlines()}
    hard={t['prompt'] for t in HARD}
    assert len(hard)==len(HARD)==32
    assert not hard & ({t['prompt'] for t in DEV+VALIDATION+CONFIRMATION}|old)
    for t in HARD:
        if t['kind']=='scalar':
            from fractions import Fraction
            Fraction(t['answer'])   # exact, parseable gold
