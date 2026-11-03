"""Back-to-back on the hard held-out set: no harness vs harness profiles.

"raw" is no harness: the task text exactly as given, the model's default chat
template, the same output budget as the harness gets for that kind. Every
condition is graded by the same scorer; code also by differential tests. The
hard set can only block a promotion; it never chooses a prompt.
"""
import argparse
import json
import random
import re

import torch

from independent_audit.model import ROOT,load
from independent_audit.evaluation import canonical
from .benchmark import SEED,grade,write
from .differential import differential
from .hard_tasks import HARD
from .runtime import infer
from .stats import paired,wilson


def run_condition(model,tok,task,condition,budgets):
    budget=budgets[task["kind"]]
    if condition=="raw":
        result=infer(model,tok,task["prompt"],"general","legacy",budget)
    else:
        result=infer(model,tok,task["prompt"],task["kind"],condition,budget)
    return result


def scorer_review_flag(row):
    # A wrong scalar answer whose text still contains the gold value may be a
    # scorer miss (e.g. trailing numbers after the answer); list it for review.
    if row["task"]["kind"]!="scalar" or row["task_checks_passed"]:
        return False
    return any(canonical(v)==canonical(row["task"]["answer"]) for v in re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?",row["text"].replace(",","")))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--conditions",nargs="+",default=["raw","legacy"])
    ap.add_argument("--out",required=True)
    ap.add_argument("--scalar-budget",type=int,default=320)
    ap.add_argument("--code-budget",type=int,default=1024)
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    args=ap.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    budgets={"scalar":args.scalar_budget,"code":args.code_budget}
    plan={"conditions":args.conditions,"budgets":budgets,"tasks":HARD}
    if (out/"plan.json").exists() and json.loads((out/"plan.json").read_text())!=plan:
        raise ValueError("Changed plan; choose a fresh output directory")
    write(out/"plan.json",plan)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model,tok,env=load(args.device,args.dtype)
    write(out/"environment.json",env)
    path=out/"generations.jsonl"
    rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={(r["task"]["id"],r["profile"]) for r in rows}
    for index,task in enumerate(HARD):
        order=list(args.conditions)
        random.Random(SEED+20_000+index).shuffle(order)
        for condition in order:
            if (task["id"],condition) in done:
                continue
            result=run_condition(model,tok,task,condition,budgets)
            row={"stage":"hard","task":task,**result,"profile":condition,**grade(result,task)}
            row["differential"]=differential(row)
            row["scorer_review"]=scorer_review_flag(row)
            with path.open("a") as f:
                f.write(json.dumps(row,allow_nan=False)+"\n")
            rows.append(row)
            d=row["differential"]
            print(f"hard {index+1}/{len(HARD)} {task['id']} {condition} in={row['input_tokens']} out={row['generated_tokens_including_eos']} checks={row['task_checks_passed']} eos={row['stopped_on_eos']} trunc={row['truncated']}"+(f" diff={d['differential_passed']}" if d else ""),flush=True)
    summary={}
    for kind in ["scalar","code"]:
        for c in args.conditions:
            rs=[r for r in rows if r["task"]["kind"]==kind and r["profile"]==c]
            k=sum(r["checks_and_eos"] for r in rs)
            entry={"n":len(rs),"correct":k,"wilson95":wilson(k,len(rs)),"truncated":sum(r["truncated"] for r in rs),
                   "mean_input":sum(r["input_tokens"] for r in rs)/len(rs),"mean_output":sum(r["generated_tokens_including_eos"] for r in rs)/len(rs),
                   "total_tokens":sum(r["total_tokens"] for r in rs),"scorer_review":[r["task"]["id"] for r in rs if r["scorer_review"]]}
            if kind=="code":
                entry["differential_passed"]=sum(bool(r["differential"] and r["differential"]["differential_passed"]) for r in rs)
            for base in ["raw","legacy"]:
                if c!=base and base in args.conditions:
                    entry[f"vs_{base}"]=paired([r for r in rows if r["task"]["kind"]==kind and r["profile"]==base],rs)
            summary[f"{kind}:{c}"]=entry
    write(out/"summary.json",summary)
    for key,e in summary.items():
        lo,hi=e["wilson95"]
        line=f"{key:16} {e['correct']:2}/{e['n']} [{lo:.2f},{hi:.2f}] trunc={e['truncated']} in={e['mean_input']:.0f} out={e['mean_output']:.0f} total={e['total_tokens']}"
        if "differential_passed" in e:
            line+=f" diff={e['differential_passed']}/{e['n']}"
        for base in ["raw","legacy"]:
            if f"vs_{base}" in e:
                v=e[f"vs_{base}"]
                line+=f" | vs {base} +{v['candidate_wins']}/-{v['candidate_losses']} p={v['mcnemar_exact_p']:.3f}"
        if e["scorer_review"]:
            line+=f" | review {e['scorer_review']}"
        print(line,flush=True)


if __name__=="__main__":
    main()
