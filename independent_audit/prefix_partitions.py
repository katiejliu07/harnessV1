"""Partition the ORIGINAL intervention without changing its donor prompt.

The main H_instruction control maps a concise donor's instruction span to target
instructions. That is useful but not a partition of E. Here keep E's bare-question
donor and absolute-position map fixed and split only target positions. Empty
partitions are analytically no-ops; never invent measured timings for them.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from .benchmark import SEED,dump,original_module
from .engine import Patch,capture,generate,synchronize
from .scoring_v2 import correct
from .model import ROOT,load
from .prompts import REASONING,CODE_REASONING,tokenize,position_map
from .tasks import HELDOUT,CODING
from .code_quality import evaluate_code


def partition_map(target,donor):
    ti,di=position_map(target,donor,"prefix")
    instruction,question=set(target["instruction"]),set(target["question"])
    output={"instruction":([],[]),"question":([],[]),"chat_or_boundary":([],[])}
    for t,d in zip(ti,di):
        key="instruction" if t in instruction else "question" if t in question else "chat_or_boundary"
        output[key][0].append(t)
        output[key][1].append(d)
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    ap.add_argument("--main",default="independent_audit/results/mps_torch271_full")
    ap.add_argument("--out",default="independent_audit/results/prefix_partition_followup")
    args=ap.parse_args()
    main_dir=ROOT/args.main
    if not (main_dir/"completed.json").exists():
        raise ValueError("Finish main GPU benchmark before this serial followup")
    source_rows=[json.loads(s) for s in (main_dir/"benchmark.jsonl").read_text().splitlines()]
    baseline={(r["suite"],r["task_index"]):r for r in source_rows if r["condition"]=="B_reasoning" and r["donor_variant"]=="same" and r["repeat"]==0}
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    plan={"arguments":vars(args),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "conditions":["instruction","question","chat_or_boundary"],"max_new_tokens":320,
                          "empty_partition_policy":"Analytical identity; reuse baseline IDs with provenance, no new latency claim",
                          "timing_scope":"Donor forward, generation and decoding only; tokenization/maps excluded. This is a causal partition experiment, not the end-to-end cost benchmark."}
    plan["supporting_source_sha256"]={name:hashlib.sha256((ROOT/"independent_audit"/name).read_bytes()).hexdigest()
        for name in ["engine.py","model.py","prompts.py","tasks.py","scoring_v2.py","code_quality.py"]}
    if (out/"plan.json").exists() and json.loads((out/"plan.json").read_text())!=plan:
        raise ValueError("Choose a fresh output directory for a changed plan")
    dump(out/"plan.json",plan)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model,tok,meta=load(args.device,args.dtype)
    dump(out/"environment.json",meta)
    sets=[("original",original_module().TEST_PROBLEMS),("heldout_arithmetic",HELDOUT),("coding",CODING)]
    path=out/"partitions.jsonl"
    done={(r["suite"],r["task_index"],r["partition"]) for s in path.read_text().splitlines() if (r:=json.loads(s))} if path.exists() else set()
    for suite,tasks in sets:
        for index,task in enumerate(tasks):
            q=task["prompt"]
            target=tokenize(tok,(CODE_REASONING if "function" in task else REASONING).format(prompt=q),q)
            donor=tokenize(tok,q,q)
            maps=partition_map(target,donor)
            order=list(maps)
            random.Random(SEED+index).shuffle(order)
            for part in order:
                if (suite,index,part) in done:
                    continue
                ti,di=maps[part]
                record={"suite":suite,"task_index":index,"task":task,"partition":part,
                        "target_positions":ti,"donor_positions":di,"donor_prompt":donor["rendered"],"prompt":target["rendered"],
                        "target_input_ids":target["ids"][0].tolist(),"donor_input_ids":donor["ids"][0].tolist()}
                if not ti:
                    ref=baseline[(suite,index)]
                    record.update(analytical_noop=True,token_ids=ref["token_ids"],text=ref["text"],truncated=ref["truncated"],
                                  measured_donor_generation_decode_seconds=None,
                                  baseline_provenance=str(main_dir/"benchmark.jsonl"))
                else:
                    synchronize(model.cfg.device)
                    start=time.perf_counter()
                    states=capture(model,donor["ids"].to(model.cfg.device),[6,2,3])
                    result=generate(model,target["ids"].to(model.cfg.device),320,tok.eos_token_id,patch=Patch(states,ti,di))
                    result.pop("last_logits")
                    synchronize(model.cfg.device)
                    record.update(result,analytical_noop=False,text=tok.decode(result["token_ids"],skip_special_tokens=True).strip(),
                                  measured_donor_generation_decode_seconds=time.perf_counter()-start)
                verdict=evaluate_code(record["text"],task)["passed"] if "function" in task else correct(record["text"],task["answer"])
                record["correct"]=bool(verdict and not record.get("truncated",baseline[(suite,index)]["truncated"]))
                record["same_tokens_as_baseline"]=record["token_ids"]==baseline[(suite,index)]["token_ids"]
                with path.open("a") as f:
                    f.write(json.dumps(record)+"\n")
                print(f"partition {suite} {index+1}/{len(tasks)} {part} n={len(ti)} identity={record['same_tokens_as_baseline']}",flush=True)
    dump(out/"completed.json",{"rows":len(path.read_text().splitlines())})


if __name__=="__main__":
    main()
