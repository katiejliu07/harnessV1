"""Exact donor-prefix optimization; no change to the frozen main benchmark.

Donor layers after max(patched_layers) cannot causally affect earlier residuals.
Stop that donor forward early, and verify bitwise tensor equality before claiming
the optimization. This is separate from the KV-caching numerical-parity question.
"""
import argparse
import json
import statistics
import time

import torch

from .engine import capture,synchronize
from .model import ROOT,load
from .prompts import tokenize
from .tasks import HELDOUT,CODING


def capture_early(model,ids,layers):
    if not layers:
        return {}
    if min(layers)<0 or max(layers)>=model.cfg.n_layers:
        raise ValueError("Layer outside model")
    states={}
    def save(x,hook):
        states[int(hook.name.split(".")[1])]=x.detach().clone()
        return x
    with torch.inference_mode():
        model.run_with_hooks(ids,return_type=None,stop_at_layer=max(layers)+1,
                             fwd_hooks=[(f"blocks.{i}.hook_resid_post",save) for i in layers])
    return states


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--out",default="independent_audit/results/donor_early_stop.json")
    args=ap.parse_args()
    torch.set_num_threads(4)
    model,tok,meta=load(args.device,args.dtype)
    from .benchmark import original_module
    tasks=original_module().TEST_PROBLEMS+HELDOUT+CODING
    records=[]
    for index,task in enumerate(tasks):
        ids=tokenize(tok,task["prompt"],task["prompt"])["ids"].to(model.cfg.device)
        timing={"full":[],"early":[]}
        equal=True
        max_abs=0.
        for rep in range(args.repeats):
            states={}
            for mode in (["full","early"] if rep%2==0 else ["early","full"]):
                synchronize(model.cfg.device)
                start=time.perf_counter()
                states[mode]=(capture if mode=="full" else capture_early)(model,ids,[6,2,3])
                synchronize(model.cfg.device)
                timing[mode].append(time.perf_counter()-start)
            equal &= all(torch.equal(states["full"][i],states["early"][i]) for i in [6,2,3])
            max_abs=max(max_abs,max(float((states["full"][i]-states["early"][i]).abs().max()) for i in [6,2,3]))
        records.append({"task_index":index,"prompt":task["prompt"],"bitwise_equal":equal,"max_absolute_error":max_abs,
                        "timings":timing,"full_median_seconds":statistics.median(timing["full"]),
                        "early_median_seconds":statistics.median(timing["early"])})
        print(f"donor {index+1}/{len(tasks)} bitwise_equal={equal}",flush=True)
    result={"environment":meta,"records":records,"all_bitwise_equal":all(r["bitwise_equal"] for r in records),
            "aggregate_median_time_ratio":sum(r["early_median_seconds"] for r in records)/sum(r["full_median_seconds"] for r in records),
            "scope":"Donor-forward-only cost; does not remove donor preparation or generation costs. Applied only to this separate optimization experiment."}
    (ROOT/args.out).write_text(json.dumps(result,indent=2,default=str))
    if not result["all_bitwise_equal"]:
        raise RuntimeError("Early-stop donor parity failed")


if __name__=="__main__":
    main()
