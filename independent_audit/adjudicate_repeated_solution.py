"""Reproduce one explicit review of repeated, identical dedup definitions.

Never choose the best of conflicting solutions. This case passes only if BOTH
offered function blocks normalize identically and independently pass the fixed
tests. Automatic/raw scores stay preserved; report.py applies a hash-bound,
separately visible adjudication rather than changing the general frozen grader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .code_quality import evaluate_code,solution_source
from .model import ROOT


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("directory",type=Path,nargs="?",default=ROOT/"independent_audit/results/budget_1024_followup")
    args=ap.parse_args()
    folder=args.directory.resolve()
    raw=(folder/"benchmark.jsonl").read_bytes()
    rows=[json.loads(s) for s in raw.decode().splitlines()]
    chosen=[r for r in rows if (r["suite"],r["task_index"],r["condition"],r["repeat"],r["donor_variant"])==("coding",0,"B_reasoning",0,"same")]
    if len(chosen)!=1:
        raise ValueError("Expected the single reviewed B/dedup case")
    row=chosen[0]
    blocks=re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```",row["text"],re.S)
    offered=[(i,b) for i,b in enumerate(blocks) if re.search(r"\bdef\s+dedup\s*\(",b)]
    sources=[solution_source(block,"dedup") for i,block in offered]
    tests=[evaluate_code(block,row["task"]) for i,block in offered]
    if len(offered)!=2 or len(set(sources))!=1 or not all(t["passed"] for t in tests):
        raise ValueError("This response does not satisfy the narrow reviewed identical-solution case")
    record={**{k:row[k] for k in ["suite","task_index","condition","repeat","donor_variant"]},
        "text_sha256":hashlib.sha256(row["text"].encode()).hexdigest(),"task_checks_passed":True,
        "reason":"Both offered dedup definitions are AST-identical after comments/whitespace removal; both pass all fixed functional tests. Repetition is not a functional failure. This list-membership implementation is quadratic in the worst case, but this task specifies no time bound.",
        "candidate_block_indices":[i for i,b in offered],"normalized_source":sources[0],"independent_block_tests":tests}
    result={"input_sha256":hashlib.sha256(raw).hexdigest(),"records":[record],
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grader_sha256":hashlib.sha256((ROOT/"independent_audit/code_quality.py").read_bytes()).hexdigest(),
        "scope":"Explicit reviewed exception; no favorable selection among differing solutions, no changes to the automatic diagnostics or raw data."}
    path=folder/"manual_adjudications.json"
    if path.exists() and json.loads(path.read_text())!=result:
        raise ValueError("Existing different review preserved; choose another result directory")
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({"adjudicated_rows":1,"blocks_checked":[i for i,b in offered],"all_identical":True,"all_passed":True},indent=2))


if __name__=="__main__":
    main()
