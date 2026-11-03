"""EvalPlus-style differential tests for the fresh code tasks (evaluator-only).

EvalPlus (arXiv 2305.01210) found hand-written suites miss many wrong programs:
80x more tests cut HumanEval pass@k by up to 19.3-28.9%. Here each task gets
seeded random inputs within its stated preconditions; the expected outputs come
from an independent reference implementation, and the generated function runs
through the same audited grader. Never shown to the model, never used to choose
a prompt: it can only block a promotion, so it cannot be overfit.
"""
import copy
import random

from independent_audit.code_quality import evaluate_code

# Reference implementations, checked against the hand-written gold cases in
# tests/test_lean.py before any model output is graded with them.
REFERENCES={
    "interleave":lambda a,b:[x for i in range(max(len(a),len(b))) for x in ([a[i]] if i<len(a) else [])+([b[i]] if i<len(b) else [])],
    "clip_values":lambda values,lo,hi:[max(lo,min(hi,x)) for x in values],
    "adjacent_differences":lambda values:[b-a for a,b in zip(values,values[1:])],
    "even_first":lambda values:[x for x in values if x%2==0]+[x for x in values if x%2!=0],
    "rotate_right":lambda values,k:values[-(k%len(values)):]+values[:-(k%len(values))] if values and k%len(values) else list(values),
    "merge_sorted":lambda a,b:sorted(a+b),
    "prefix_sums":lambda values:[sum(values[:i+1]) for i in range(len(values))],
    "moving_sums":lambda values,k:[] if k<=0 or k>len(values) else [sum(values[i:i+k]) for i in range(len(values)-k+1)],
    "expand_runs":lambda pairs:[v for v,n in pairs for _ in range(n)],
    "nonzero_indices":lambda values:[i for i,x in enumerate(values) if x!=0],
    "flatten_once":lambda rows:[x for row in rows for x in row],
    "keep_last":lambda items:[x for i,x in enumerate(items) if x not in items[i+1:]],
    "chunked":lambda values,size:[values[i:i+size] for i in range(0,len(values),size)],
    "running_max":lambda values:[max(values[:i+1]) for i in range(len(values))],
    "pairwise_swap":lambda values:[x for i in range(0,len(values),2) for x in values[i:i+2][::-1]],
    "top_two":lambda values:sorted(set(values),reverse=True)[:2],
    "intersperse":lambda values,sep:[y for i,x in enumerate(values) for y in ([sep,x] if i else [x])],
    "drop_every":lambda values,n:list(values) if n<=0 else [x for i,x in enumerate(values,1) if i%n],
    "indices_of_max":lambda values:[i for i,x in enumerate(values) if x==max(values)],
    "word_lengths":lambda text:[len(w) for w in text.split()],
    # hard held-out set
    "merge_intervals":lambda intervals:_merge(intervals),
    "spiral_order":lambda matrix:_spiral([list(r) for r in matrix]),
    "top_k_frequent":lambda words,k:[] if k<=0 else sorted(set(words),key=lambda w:(-words.count(w),w))[:k],
    "next_greater":lambda values:[next((y for y in values[i+1:] if y>x),None) for i,x in enumerate(values)],
    "primes_upto":lambda n:[p for p in range(2,n+1) if all(p%d for d in range(2,int(p**.5)+1))],
    "rle_encode":lambda text:_rle(text),
    "sliding_max":lambda values,k:[] if k<=0 or k>len(values) else [max(values[i:i+k]) for i in range(len(values)-k+1)],
    "tokenize_expr":lambda text:_tokens(text),
}


def _merge(intervals):
    out=[]
    for s,e in sorted(intervals):
        if out and s<=out[-1][1]:
            out[-1][1]=max(out[-1][1],e)
        else:
            out.append([s,e])
    return out


def _spiral(m):
    out=[]
    while m and m[0]:
        out+=m.pop(0)
        m=[list(r) for r in zip(*m)][::-1] if m and m[0] else []
    return out


def _rle(text):
    out=[]
    for ch in text:
        if out and out[-1][0]==ch:
            out[-1][1]+=1
        else:
            out.append([ch,1])
    return out


def _tokens(text):
    out,num=[],""
    for ch in text+" ":
        if ch.isdigit():
            num+=ch
            continue
        if num:
            out.append(int(num))
            num=""
        if ch in "+-*/()":
            out.append(ch)
    return out


def _ints(r,lo=-20,hi=20,n=None):
    return [r.randint(lo,hi) for _ in range(r.randint(0,8) if n is None else n)]


# Inputs respect each prompt's stated preconditions (e.g. lo <= hi, sorted
# inputs, positive size, nonnegative counts) and stay JSON-serializable.
GENERATORS={
    "interleave":lambda r:[_ints(r),_ints(r)],
    "clip_values":lambda r:(lambda lo:[_ints(r),lo,lo+r.randint(0,10)])(r.randint(-10,10)),
    "adjacent_differences":lambda r:[_ints(r)],
    "even_first":lambda r:[_ints(r)],
    "rotate_right":lambda r:[_ints(r),r.randint(-15,15)],
    "merge_sorted":lambda r:[sorted(_ints(r)),sorted(_ints(r))],
    "prefix_sums":lambda r:[_ints(r)],
    "moving_sums":lambda r:[_ints(r),r.randint(-2,9)],
    "expand_runs":lambda r:[[[r.choice(["a","b",1,None,False]),r.randint(0,3)] for _ in range(r.randint(0,5))]],
    "nonzero_indices":lambda r:[_ints(r,-3,3)],
    "flatten_once":lambda r:[[[r.choice([1,None,[2],"x"]) for _ in range(r.randint(0,3))] for _ in range(r.randint(0,4))]],
    # No bools here: True==1 and False==0 for "occurrence" but the grader's
    # JSON comparison tells them apart, so mixing them would test an ambiguity.
    "keep_last":lambda r:[[r.choice([0,1,2,"a",None]) for _ in range(r.randint(0,8))]],
    "chunked":lambda r:[_ints(r),r.randint(1,5)],
    "running_max":lambda r:[_ints(r)],
    "pairwise_swap":lambda r:[[r.choice([0,1,None,False,"a"]) for _ in range(r.randint(0,7))]],
    "top_two":lambda r:[_ints(r,-5,5)],
    "intersperse":lambda r:[_ints(r),r.choice([0,None,"-"])],
    "drop_every":lambda r:[_ints(r),r.randint(-1,5)],
    "indices_of_max":lambda r:[_ints(r,-3,3)],
    "word_lengths":lambda r:["".join(r.choice(["a","bb"," ","\t","\n","  "]) for _ in range(r.randint(0,10)))],
    "merge_intervals":lambda r:[[(lambda s:[s,s+r.randint(0,6)])(r.randint(-10,15)) for _ in range(r.randint(0,6))]],
    "spiral_order":lambda r:(lambda h,w:[[[r.randint(-9,9) for _ in range(w)] for _ in range(h)]])(r.randint(0,5),r.randint(1,5)),
    "top_k_frequent":lambda r:[[r.choice(["a","b","c","d","e"]) for _ in range(r.randint(0,12))],r.randint(-1,6)],
    "next_greater":lambda r:[_ints(r,-5,5)],
    "primes_upto":lambda r:[r.randint(-3,120)],
    "rle_encode":lambda r:["".join(r.choice("aab c") for _ in range(r.randint(0,12)))],
    "sliding_max":lambda r:[_ints(r),r.randint(-1,9)],
    "tokenize_expr":lambda r:["".join(r.choice(["1","23","405","+","-","*","/","(",")"," "]) for _ in range(r.randint(0,10)))],
}


def cases(function,n=60,seed=202610295):
    """n seeded random cases with reference outputs; deterministic per function."""
    r=random.Random(f"{seed}:{function}")
    out=[]
    for _ in range(n):
        args=GENERATORS[function](r)
        out.append((args,REFERENCES[function](*copy.deepcopy(args))))
    return out


def differential(row,n=60):
    """Grade one generated row on its hand cases plus n differential cases.
    Tasks without a reference (the preserved regression suite) return None."""
    task=row["task"]
    if task.get("kind")!="code" or task.get("function") not in REFERENCES:
        return None
    combined={**task,"cases":task["cases"]+cases(task["function"],n)}
    verdict=evaluate_code(row["text"],combined)
    return {"task_id":task["id"],"profile":row["profile"],"stage":row.get("stage"),"base_passed":row.get("task_checks_passed"),
            "differential_passed":verdict["passed"],"cases":len(combined["cases"]),
            "failed_cases":[i for i,ok in enumerate(verdict.get("functional",[])) if not ok][:5]}
