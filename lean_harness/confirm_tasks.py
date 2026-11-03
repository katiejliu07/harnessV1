"""Fresh confirmation set, unseen by every earlier stage; evaluator-only answers.

Why it exists: v3 validation has 24 scalar and 8 code tasks. A profile that is
truly 1-in-20 worse still shows zero losses on 24 items ~29% of the time, so a
promotion must also survive this larger, independent set before it becomes a
default. Never tune prompts on these outputs; a failure keeps legacy.
"""
import json

from independent_audit.model import ROOT
from .tasks import DEV, VALIDATION, arithmetic, code

CONFIRM_CODE=[
    code("chunked","Write chunked(values, size), returning a new list of consecutive sublists of length size, the last one possibly shorter. size is a positive integer. Handle empty input. Do not modify the input.",
         [([[1,2,3,4,5],2],[[1,2],[3,4],[5]]),([[],3],[]),([[1,2],5],[[1,2]]),([[None,0,False],1],[[None],[0],[False]])]),
    code("running_max","Write running_max(values), returning a new list where each position holds the largest value seen so far. Handle empty input and negative numbers. Do not modify the input.",
         [([[3,1,4,1,5]],[3,3,4,4,5]),([[]],[]),([[-5,-7,-2]],[-5,-5,-2]),([[2,2]],[2,2])]),
    code("pairwise_swap","Write pairwise_swap(values), returning a new list with each adjacent pair swapped: items 0 and 1, then 2 and 3, and so on. A final unpaired item stays last. Handle empty input. Do not modify the input.",
         [([[1,2,3,4]],[2,1,4,3]),([[1,2,3]],[2,1,3]),([[]],[]),([[None]],[None]),([[False,None]],[None,False])]),
    code("top_two","Write top_two(values), returning a list of the two largest distinct numbers in descending order; return fewer if there are fewer distinct numbers. Handle empty input. Do not modify the input.",
         [([[4,1,4,3]],[4,3]),([[]],[]),([[7,7]],[7]),([[-1,-5]],[-1,-5]),([[2,9,9,9,1]],[9,2])]),
    code("intersperse","Write intersperse(values, sep), returning a new list with sep placed between every two adjacent items. Handle empty and single-item input. Do not modify the input.",
         [([[1,2,3],0],[1,0,2,0,3]),([[],0],[]),([['a'],'-'],['a']),([[None,None],None],[None,None,None])]),
    code("drop_every","Write drop_every(values, n), returning a new list without every n-th item (the n-th, 2n-th, and so on, counting from 1). For n <= 0 return a copy of values. Do not modify the input.",
         [([[1,2,3,4,5,6],2],[1,3,5]),([[1,2,3],1],[]),([[1,2,3],0],[1,2,3]),([[1,2],5],[1,2]),([[],3],[])]),
    code("indices_of_max","Write indices_of_max(values), returning a list of every index where the largest value occurs, in increasing order, or [] for an empty list. Do not modify the input.",
         [([[3,9,2,9]],[1,3]),([[]],[]),([[-4,-2,-2]],[1,2]),([[5]],[0])]),
    # The audited grader requires list returns for these tasks.
    code("word_lengths","Write word_lengths(text), returning a list of the lengths of the words in text, where words are separated by any run of whitespace. Ignore leading and trailing whitespace.",
         [(["  ab   c\tdef \n"],[2,1,3]),([""],[]),(["   "],[]),(["x"],[1]),(["a\n\nbb"],[1,2])]),
]

def fresh_arithmetic(n=48,seed=202610294):
    """n scalar tasks whose prompts no earlier stage has shown the model. Some
    templates draw from small ranges, so drawn repeats are skipped, not reused."""
    seen={t["prompt"] for t in DEV+VALIDATION}
    seen|={json.loads(s)["task"]["prompt"] for s in (ROOT/"independent_audit/results/mps_torch271_full/benchmark.jsonl").read_text().splitlines()}
    out=[]
    for task in arithmetic(seed,(n//12)*3):
        if task["prompt"] not in seen:
            seen.add(task["prompt"])
            out.append(dict(task,id="confirm_"+task["id"]))
        if len(out)==n:
            return out
    raise ValueError("Not enough fresh arithmetic tasks")


CONFIRMATION=fresh_arithmetic()+CONFIRM_CODE
