"""Conditional output-budget diagnostic for actual A/B/C/E truncations.

This is not a new blind test set and does not overwrite the 320-token matrix.
For every target where ANY application condition truncated, rerun ALL four at
the same 1024-token cap. Never repair only the preferred condition or interpret
a capped baseline as intrinsically worse without this check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from .benchmark import SEED,run_condition,dump
from .model import ROOT,load

CONDITIONS=["A_normal","B_reasoning","C_concise","E_v8_prefix"]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="independent_audit/results/budget_1024_followup")
    ap.add_argument("--main",default="independent_audit/results/mps_torch271_full")
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    args=ap.parse_args()
    main_dir=ROOT/args.main
    if not (main_dir/"completed.json").exists():
        raise ValueError("Wait for main benchmark and serialize GPU followups")
    raw=main_dir/"benchmark.jsonl"
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    candidates=[r for r in rows if r["status"]=="ok" and r["donor_variant"]=="same" and r["condition"] in CONDITIONS and r["truncated"]]
    triggers={}
    for row in candidates:
        key=(row["suite"],row["task_index"])
        triggers.setdefault(key,{"task":row["task"],"truncated_conditions":[]})["truncated_conditions"].append(row["condition"])
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    plan={"arguments":vars(args),"selection_rule":"Any main A/B/C/E matched-donor response truncated at 320; symmetric rerun of all four conditions",
          "targets":[{"suite":k[0],"task_index":k[1],**v} for k,v in sorted(triggers.items())],
          "conditions":CONDITIONS,"max_new_tokens":1024,"seed":SEED,
          "main_data_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),
          "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "scope":"Conditional diagnostic, not independent replications or a replacement for the original 320-token benchmark"}
    if (out/"plan.json").exists() and json.loads((out/"plan.json").read_text())!=plan:
        raise ValueError("Changed followup plan requires a fresh --out directory")
    dump(out/"plan.json",plan)
    if not triggers:
        dump(out/"completed.json",{"rows":0,"reason":"No A/B/C/E truncations"})
        print("No application-condition truncations; no extra GPU runs",flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model,tok,meta=load(args.device,args.dtype)
    meta["generation"]["max_new_tokens"]=1024
    dump(out/"environment.json",meta)
    path=out/"benchmark.jsonl"
    done={(r["suite"],r["task_index"],r["condition"]) for s in path.read_text().splitlines() if (r:=json.loads(s))} if path.exists() else set()
    for (suite,index),info in sorted(triggers.items()):
        order=CONDITIONS.copy()
        random.Random(SEED+index).shuffle(order)
        for condition in order:
            if (suite,index,condition) in done:
                continue
            row={"suite":suite,"task_index":index,"task":info["task"],"condition":condition,"repeat":0,
                 "donor_variant":"same","donor_task":info["task"],"triggered_by":info["truncated_conditions"]}
            row.update(run_condition(model,tok,info["task"],condition,max_tokens=1024))
            with path.open("a") as f:
                f.write(json.dumps(row)+"\n")
            print(f"budget 1024 {suite} {index} {condition} {row['generated_tokens_including_eos']}t truncated={row['truncated']}",flush=True)
    dump(out/"completed.json",{"rows":len(path.read_text().splitlines()),"selected_targets":len(triggers)})


if __name__=="__main__":
    main()
