"""Live A/B: Codex coding tasks, plain call vs the harness's toolset first rung.

    python -m agent_harness.ab_codex [--repeats 3] [--jobs 6]

baseline: one Codex call with its full default setup (what a caller gets
          without the harness).
harness:  the harness's real path: a first rung without the tools a code
          task never uses (config agents.codex.toolset_args); if its check
          fails, the unchanged full setup runs, and its tokens count too.

Six coding tasks: the three multi-file recheck projects (graded by hidden
tests kept outside the workdir and never mentioned) and the three pilot tool
tasks. Each run gets a fresh directory. Within a (task, repeat) pair the arm
order alternates; pairs for different tasks run in parallel. Rows are appended
as they finish, and a rerun skips rows already saved (resume, never replace).
"""
import argparse
import json
import random
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .pilot import hygiene_task
from .recheck import PROJECTS,hidden_check,materialize,self_check
from .run import load_config,run_task

ROOT=Path(__file__).resolve().parents[1]
TASKS=list(PROJECTS)+["log","tests","csv"]
ARMS=("baseline","harness")


def arm_config(base,arm):
    cfg=json.loads(json.dumps(base))
    tech=cfg["profiles"]["technical"]
    tech["ladder"]=[{"agent":"codex"}]                       # the installed default model and effort
    tech["toolset_first_rung"]=arm=="harness"
    tech["hygiene"]=False
    return cfg


def make_task(name,root,hidden):
    if name in PROJECTS:
        d=materialize(name,root)
        return {"prompt":PROJECTS[name]["prompt"],"check_cmd":hidden_check(name,hidden),"workdir":str(d)}
    return hygiene_task(name,root)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="agent_harness/results/ab_codex_toolset_v1")
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--jobs",type=int,default=6)
    args=ap.parse_args()
    self_check()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    rows_path=out/"rows.jsonl"
    done={(r["task"],r["repeat"],r["arm"]) for r in map(json.loads,rows_path.read_text().splitlines())} if rows_path.exists() else set()
    base=load_config()
    lock=threading.Lock()
    scratch=Path(tempfile.mkdtemp(prefix="ab_codex_"))

    def pair(job):
        name,rep=job
        order=list(ARMS) if (TASKS.index(name)+rep)%2==0 else list(reversed(ARMS))
        for arm in order:
            if (name,rep,arm) in done:
                continue
            root=Path(tempfile.mkdtemp(dir=scratch))
            hidden=Path(tempfile.mkdtemp(dir=scratch))
            report=run_task(make_task(name,root,hidden),arm_config(base,arm))
            u=report["usage_total"]
            row={"task":name,"repeat":rep,"arm":arm,"order":order.index(arm),"verified":report["verified"],
                 "attempts":len(report["attempts"]),"tokens_spent":report["tokens_spent"],
                 "new_input":u.get("new_input"),"cached_input":u.get("cached_input"),"output_tokens":u.get("output_tokens",0),
                 "weighted_tokens":u.get("weighted_tokens"),"seconds":round(sum(a["seconds"] for a in report["attempts"]),3),
                 "errors":[a.get("exception") or a.get("error") for a in report["attempts"]]}
            with lock:
                with rows_path.open("a") as f:
                    f.write(json.dumps(row)+"\n")
                print(f"{name:10} rep={rep} {arm:8} verified={row['verified']} attempts={row['attempts']} "
                      f"tokens={row['tokens_spent']} weighted={row['weighted_tokens']} s={row['seconds']}",flush=True)
            shutil.rmtree(root,ignore_errors=True)

    jobs=[(n,r) for r in range(args.repeats) for n in TASKS]
    random.Random(0).shuffle(jobs)
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(pair,jobs))
    finally:
        shutil.rmtree(scratch,ignore_errors=True)
    summarize(rows_path,out/"summary.json")


def summarize(rows_path,summary_path):
    rows=[json.loads(l) for l in rows_path.read_text().splitlines()]
    by={}
    for r in rows:
        by.setdefault((r["task"],r["repeat"]),{})[r["arm"]]=r
    pairs=[v for v in by.values() if all(a in v for a in ARMS)]
    keys=("tokens_spent","new_input","cached_input","output_tokens","weighted_tokens","seconds")
    total={a:{k:round(sum(p[a][k] or 0 for p in pairs),1) for k in keys} for a in ARMS}
    for a in ARMS:
        total[a]["passed"]=sum(bool(p[a]["verified"]) for p in pairs)
        total[a]["runs"]=len(pairs)
    change={k:round(100*(total["harness"][k]-total["baseline"][k])/total["baseline"][k],2) for k in keys if total["baseline"][k]}
    per_task={}
    for t in TASKS:
        ps=[p for p in pairs if p["baseline"]["task"]==t]
        if ps:
            b=sum(p["baseline"]["weighted_tokens"] for p in ps); h=sum(p["harness"]["weighted_tokens"] for p in ps)
            per_task[t]={"pairs":len(ps),"weighted_change_percent":round(100*(h-b)/b,2),
                         "passed":{a:sum(bool(p[a]["verified"]) for p in ps) for a in ARMS},
                         "harness_fell_back":sum(p["harness"]["attempts"]>1 for p in ps)}
    losses=sorted(f"{p['baseline']['task']}#{p['baseline']['repeat']}" for p in pairs
                  if p["baseline"]["verified"] and not p["harness"]["verified"])
    summary={"pairs":len(pairs),"unpaired":len(by)-len(pairs),"total":total,"change_percent":change,"per_task":per_task,
             "harness_losses":losses,"weights":"weighted = new input + 0.1 x cached + 8 x output (illustrative, not prices)"}
    summary_path.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
