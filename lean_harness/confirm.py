"""Second, larger accuracy gate for profiles a benchmark promoted.

A promotion stays only if, on the fresh confirmation set, the profile loses no
item that legacy passes (checks and EOS) and spends fewer total input+output
tokens and less request time. Kinds the policy did not promote are not run.
Never tune on these outputs: a failure keeps legacy, it does not pick another
profile. The confirmed policy goes to a new path; the input policy is untouched.
"""
import argparse
import hashlib
import json
import random

import torch

from independent_audit.model import ROOT,load
from .benchmark import SEED,compare,grade,summarize,write
from .confirm_tasks import CONFIRMATION
from .runtime import infer


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--policy",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--policy-out",required=True)
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    args=ap.parse_args()
    policy=json.loads((ROOT/args.policy).read_text())
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    if (ROOT/args.policy_out).exists():
        raise ValueError("Existing policy preserved; choose a new --policy-out")
    promoted={k:v["profile"] for k,v in policy["kinds"].items() if v.get("promoted")}
    tasks=[t for t in CONFIRMATION if t["kind"] in promoted]
    plan={"policy":args.policy,"policy_sha256":hashlib.sha256((ROOT/args.policy).read_bytes()).hexdigest(),
        "promoted":promoted,"tasks":tasks,
        "gate":"Zero checks/EOS regressions against legacy, lower total tokens and request time, per kind."}
    plan_path=out/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError("Changed plan; choose a fresh output directory")
    write(plan_path,plan)
    rows=[]
    if tasks:
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        model,tok,env=load(args.device,args.dtype)
        if any(env[k]!=policy["environment"][k] for k in ["device","dtype","model_revision"]):
            raise ValueError("Confirmation must run on the policy's measured backend")
        data_path=out/"generations.jsonl"
        rows=[json.loads(s) for s in data_path.read_text().splitlines()] if data_path.exists() else []
        done={(r["task"]["id"],r["profile"]) for r in rows}
        for index,task in enumerate(tasks):
            order=["legacy",promoted[task["kind"]]]
            random.Random(SEED+10_000+index).shuffle(order)
            for profile in order:
                if (task["id"],profile) in done:
                    continue
                result=infer(model,tok,task["prompt"],task["kind"],profile)
                row={"stage":"confirmation","task":task,**result,**grade(result,task)}
                with data_path.open("a") as f:
                    f.write(json.dumps(row,allow_nan=False)+"\n")
                rows.append(row)
                print(f"confirmation {index+1}/{len(tasks)} {task['id']} {profile} in={row['input_tokens']} out={row['generated_tokens_including_eos']} checks={row['task_checks_passed']} eos={row['stopped_on_eos']} {row['total_seconds']:.2f}s",flush=True)
    kinds={}
    for kind,entry in policy["kinds"].items():
        if kind not in promoted:
            kinds[kind]=entry
            continue
        subset=[r for r in rows if r["task"]["kind"]==kind]
        verdict=compare([r for r in subset if r["profile"]=="legacy"],[r for r in subset if r["profile"]==promoted[kind]])
        kinds[kind]={**entry,"promoted":verdict["eligible"],"confirmation":verdict}
    write(out/"summary.json",{"summaries":summarize(rows),"kinds":kinds})
    confirmed={**policy,"kinds":kinds,"confirmation_evidence":str(out.relative_to(ROOT))}
    write(ROOT/args.policy_out,confirmed)
    print(json.dumps({k:{"profile":v["profile"],"promoted":v["promoted"]} for k,v in kinds.items()},indent=2),flush=True)


if __name__=="__main__":
    main()
