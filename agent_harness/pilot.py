"""Pilot: do lean mode and output hygiene save tokens without losing accuracy?

Real Claude Code / Codex runs (uses subscription limits). Decisions follow the
same rule as the rest of the harness: a mode is enabled only with zero item
losses against the default and lower tokens. Metric: tokens per correct answer
(Cost-of-Pass, arXiv 2504.13359), with input counted as every token read
(fresh + cache writes + cache reads).

A. Lean mode, tool-free tasks (Claude): default vs no-tools vs no-tools plus a
   short replacement system prompt, on hard word problems and coding functions,
   graded by exact checks and EvalPlus-style differential tests.
B. Output hygiene, tool tasks with noisy output (Claude and Codex): with and
   without the instruction, repeated, graded by each task's check command.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]      # repo root; avoids importing torch via independent_audit.model
from lean_harness.differential import differential
from lean_harness.hard_tasks import HARD
from . import agents
from .run import load_config,run_task
from .verify import has_check

LEAN_PROMPT="You are a precise assistant. Answer the question directly."
CLAUDE_TOOL_ARGS=["--allowedTools","Read","Edit","Write","Bash(python3 *)","Bash(sh *)","Bash(./run.sh*)","Bash(cat *)","Bash(head *)","Bash(tail *)","Bash(grep *)","Bash(wc *)","Bash(ls*)","Bash(awk *)","Bash(sort *)"]


def lean_tasks():
    scalar=[t for t in HARD if t["kind"]=="scalar" and t["id"] in {"hard_00_01","hard_00_02","hard_00_05","hard_00_06","hard_00_08","hard_00_10"}]
    code=[t for t in HARD if t["kind"]=="code" and t["function"] in {"merge_intervals","spiral_order","top_k_frequent","next_greater","rle_encode","tokenize_expr"}]
    # The same output-format line for every condition, so differences come from
    # the mode under test, not from how the scorer reads the answer.
    # Hard-set gold answers live in "answer"; the harness checker reads "expected".
    tasks=([{**t,"expected":t["answer"],"prompt":t["prompt"]+" End with: Final answer: <number or fraction>."} for t in scalar]+
           [{**t,"prompt":t["prompt"]+" Reply with the complete Python function in a code block."} for t in code])
    if not all(has_check(t) for t in tasks):
        raise ValueError("Every pilot task needs an objective check; refusing to run ungraded")
    return tasks


def hygiene_task(name,root):
    d=Path(root)/name
    d.mkdir(parents=True)
    rng=random.Random(name)
    if name=="log":
        lines=[f"2026-10-29 10:{i//60:02d}:{i%60:02d},{i%1000:03d} INFO heartbeat ok worker={i%8} queue={rng.randint(0,9)}" for i in range(4000)]
        lines.insert(2731,"2026-10-29 10:45:31,731 ERROR component=payment-gateway code=E1042 upstream timeout")
        (d/"events.log").write_text("\n".join(lines)+"\n")
        (d/"run.sh").write_text("#!/bin/sh\ncat \"$(dirname \"$0\")/events.log\"\n")
        (d/"run.sh").chmod(0o755)
        return {"prompt":"Run ./run.sh in this directory. Find the component that logged an ERROR and write only that component's name to answer.txt.",
                "check_cmd":"grep -qx 'payment-gateway' answer.txt","workdir":str(d)}
    if name=="tests":
        (d/"calc.py").write_text("def running_total(values):\n    out=[]\n    total=0\n    for v in values[1:]:\n        total+=v\n        out.append(total)\n    return out\n")
        (d/"test.py").write_text("import sys\nfrom calc import running_total\nfailed=0\nfor n in range(2000):\n    vals=list(range(n%7))\n    exp=[sum(vals[:i+1]) for i in range(len(vals))]\n    got=running_total(vals)\n    if got==exp:\n        print(f'check {n}: ok')\n    else:\n        failed+=1\n        print(f'check {n}: FAIL expected {exp} got {got}')\nprint(f'{failed} failed')\nsys.exit(1 if failed else 0)\n")
        return {"prompt":"Run python3 test.py in this directory. Fix calc.py so every check passes. Do not edit test.py.",
                "check_cmd":"python3 test.py > /dev/null && grep -q 'for n in range(2000)' test.py","workdir":str(d)}
    rows=["id,status,amount"]+[f"{i},{rng.choice(['paid','refunded','pending'])},{rng.randint(1,500)}" for i in range(5000)]
    (d/"data.csv").write_text("\n".join(rows)+"\n")
    paid=sum(int(r.split(',')[2]) for r in rows[1:] if r.split(',')[1]=="paid")
    return {"prompt":"In data.csv in this directory, add up the amount column for rows whose status is paid. Write only the integer total to answer.txt.",
            "check_cmd":f"test \"$(tr -d '[:space:]' < answer.txt)\" = '{paid}'","workdir":str(d)}


def single(config,agent,model,effort,lean=False,system_prompt=None,hygiene=False,extra=None,toolset=None):
    """One-rung config so conditions differ only in the setting under test.
    First-rung insertion is off, so exactly this one step runs."""
    config=json.loads(json.dumps(config))
    step={"agent":agent,"model":model,"effort":effort}
    if lean:
        step.update({"lean":True,"system_prompt":system_prompt})
    if toolset is not None:
        step["toolset"]=toolset
    if hygiene:
        config["hygiene_agents"]=[agent]
    for name in ("technical","general","important"):
        config["profiles"][name].update({"ladder":[step],"lean_first_rung":False,"toolset_first_rung":False,"hygiene":hygiene})
    if extra is not None:
        config["agents"][agent]["extra_args"]=extra
    return config


def record(f,part,condition,task,report):
    a=report["attempts"][-1] if report["attempts"] else {}
    row={"part":part,"condition":condition,"task_id":task.get("id",task.get("name")),"verified":report["verified"],
         "usage":report["usage_total"],"seconds":sum(x["seconds"] for x in report["attempts"]),"error":a.get("error"),
         "text":report["text"][-2000:]}
    if task.get("function"):
        d=differential({"task":task,"text":report["text"],"profile":condition})
        row["differential"]=d and d["differential_passed"]
    f.write(json.dumps(row)+"\n")
    f.flush()
    print(f"{part} {condition:22} {row['task_id']:24} verified={row['verified']} diff={row.get('differential')} input={row['usage'].get('input_total')} out={row['usage'].get('output_tokens')}",flush=True)
    return row


def summarize(rows):
    out={}
    for r in rows:
        key=(r["part"],r["condition"])
        e=out.setdefault(key,{"n":0,"correct":0,"input_total":0,"output_tokens":0})
        e["n"]+=1
        ok=bool(r["verified"]) and r.get("differential",True) is not False
        e["correct"]+=ok
        e["input_total"]+=r["usage"].get("input_total",0)
        e["output_tokens"]+=r["usage"].get("output_tokens",0)
    for e in out.values():
        e["tokens_per_correct"]=(e["input_total"]+e["output_tokens"])/e["correct"] if e["correct"] else None
    return {f"{p}:{c}":v for (p,c),v in out.items()}


def losses(rows,base,cand):
    ok=lambda r:bool(r["verified"]) and r.get("differential",True) is not False
    b={(r["task_id"],i):ok(r) for i,r in enumerate([x for x in rows if x["condition"]==base])}
    c={(r["task_id"],i):ok(r) for i,r in enumerate([x for x in rows if x["condition"]==cand])}
    return sorted(k[0] for k in b if b[k] and not c.get(k,False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="agent_harness/results/pilot_v1")
    ap.add_argument("--parts",default="AB")
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    base=load_config()
    rows=[]
    with (out/"rows.jsonl").open("a") as f:
        if "A" in args.parts:
            conditions={"default":dict(),"lean_no_tools":dict(lean=True),"lean_no_tools_short_sys":dict(lean=True,system_prompt=LEAN_PROMPT)}
            for task in lean_tasks():
                order=list(conditions)
                random.Random(task["id"]).shuffle(order)
                for name in order:
                    with tempfile.TemporaryDirectory() as empty:   # a default run has tools: give it nothing to touch
                        cfg=single(base,"claude","sonnet","low",**conditions[name])
                        report=run_task({**task,"kind":"technical","tools":False,"workdir":empty},cfg)
                    rows.append(record(f,"A",name,task,report))
        if "B" in args.parts:
            root=Path(tempfile.mkdtemp(prefix="hygiene_"))
            try:
                for rep in range(args.repeats):
                    for name in ["log","tests","data"]:
                        for agent,extra in [("claude",CLAUDE_TOOL_ARGS),("codex",["--skip-git-repo-check","--ephemeral","-s","workspace-write"])]:
                            for hygiene in random.Random(f"{rep}{name}{agent}").sample([False,True],2):
                                task=hygiene_task(f"{name}_{agent}_{int(hygiene)}_{rep}",root)
                                model,effort=("sonnet","low") if agent=="claude" else (None,"low")
                                cfg=single(base,agent,model,effort,hygiene=hygiene,extra=extra)
                                report=run_task({**task,"name":name},cfg)
                                rows.append(record(f,"B",f"{agent}_{'hygiene' if hygiene else 'plain'}",{"name":name},report))
            finally:
                shutil.rmtree(root,ignore_errors=True)
        if "C" in args.parts:
            # Only the toolset a coding task needs vs every built-in tool (Claude).
            root=Path(tempfile.mkdtemp(prefix="toolset_"))
            minimal=base["toolsets"]["code"]
            try:
                for rep in range(args.repeats):
                    for name in ["log","tests","data"]:
                        for condition in random.Random(f"C{rep}{name}").sample(["full_tools","code_toolset"],2):
                            task=hygiene_task(f"{name}_{condition}_{rep}",root)
                            cfg=single(base,"claude","sonnet","low",extra=CLAUDE_TOOL_ARGS,toolset=minimal if condition=="code_toolset" else None)
                            report=run_task({**task,"name":name},cfg)
                            rows.append(record(f,"C",condition,{"name":name},report))
            finally:
                shutil.rmtree(root,ignore_errors=True)
    summary=summarize(rows)
    decisions={}
    if "A" in args.parts:
        for cand in ["lean_no_tools","lean_no_tools_short_sys"]:
            lost=losses(rows,"default",cand)
            s,b=summary.get(f"A:{cand}"),summary["A:default"]
            decisions[cand]={"losses":lost,"tokens_per_correct":s["tokens_per_correct"],"default_tokens_per_correct":b["tokens_per_correct"],
                             "enable":not lost and s["correct"]>0 and (s["input_total"]+s["output_tokens"])<(b["input_total"]+b["output_tokens"])}
    if "B" in args.parts:
        for agent in ["claude","codex"]:
            lost=losses(rows,f"{agent}_plain",f"{agent}_hygiene")
            s,b=summary.get(f"B:{agent}_hygiene"),summary.get(f"B:{agent}_plain")
            decisions[f"hygiene_{agent}"]={"losses":lost,"tokens_per_correct":s["tokens_per_correct"],"plain_tokens_per_correct":b["tokens_per_correct"],
                                           "enable":not lost and s["correct"]>0 and (s["input_total"]+s["output_tokens"])<(b["input_total"]+b["output_tokens"])}
    if "C" in args.parts:
        lost=losses(rows,"full_tools","code_toolset")
        s,b=summary.get("C:code_toolset"),summary.get("C:full_tools")
        decisions["toolset_first_rung"]={"losses":lost,"tokens_per_correct":s["tokens_per_correct"],"full_tokens_per_correct":b["tokens_per_correct"],
                                         "enable":not lost and s["correct"]>0 and (s["input_total"]+s["output_tokens"])<(b["input_total"]+b["output_tokens"])}
    (out/"summary.json").write_text(json.dumps({"summary":summary,"decisions":decisions},indent=2))
    print(json.dumps({"summary":summary,"decisions":decisions},indent=2),flush=True)


if __name__=="__main__":
    main()
