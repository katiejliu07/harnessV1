"""Predeclared extra edge cases for v2; no additional model calls or repairs.

These tests guard against generic-list solutions that quietly drop None, lose
duplicates, flatten too far, or mutate inputs. They are evaluator-only and never
sent to the model. A failed candidate is blocked, not replaced using these tests.
"""
from independent_audit.code_quality import evaluate_code

EXTRA={
    "interleave":[([[None,1],[2,None,3]],[None,2,1,None,3]),([[1],[4,5,6]],[1,4,5,6])],
    "clip_values":[([[-7,-2,0,4],-3,-1],[-3,-2,-1,-1])],
    "adjacent_differences":[([[4,4,4]],[0,0])],
    "even_first":[([[-5,-4,-3,-2,0]],[-4,-2,0,-5,-3])],
    "rotate_right":[([[None,False,2],-4],[False,2,None]),([[1],100],[1])],
    "merge_sorted":[([[1,1],[1,1]],[1,1,1,1]),([[2],[]],[2])],
    "prefix_sums":[([[-1,1,-2,2]],[-1,0,-2,0])],
    "moving_sums":[([[1,-1,2],3],[2]),([[0,0,0],2],[0,0])],
    "expand_runs":[([[[None,2],[False,1]]],[None,None,False])],
    "nonzero_indices":[([[False,2,0,True]],[1,3])],
    "flatten_once":[([[[None],[[],[1]]]],[None,[],[1]])],
    "keep_last":[([[False,None,False,0,None,1,True]],[0,None,True])],
    "dedup":[([[None,False,0,True,1]],[None,False,True])],
    "runs":[([[None,None,False,False,0,0,1]],[[None,2],[False,4],[1,1]])],
    "lower_bound":[([[-3,-3,-3],-3],0),([[-3,-3,-3],0],3),([[-3,-3,-3],-4],0)],
    "balanced":[(["a([{}])z"],True),(["([{}])]"],False)],
}


def stress(row):
    task=row["task"]
    combined={**task,"cases":task["cases"]+EXTRA.get(task["function"],[])}
    verdict=evaluate_code(row["text"],combined)
    return {"stage":row["stage"],"task_id":task["id"],"profile":row["profile"],
            "base_checks_passed":row["task_checks_passed"],"combined_case_count":len(combined["cases"]),
            "passed":verdict["passed"],"grade":verdict}
