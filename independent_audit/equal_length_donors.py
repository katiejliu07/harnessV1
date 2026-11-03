"""Numeral-only donor changes: same tokenizer length, same absolute patch map.

The full matrix intentionally tests original v8 behavior with different donor
lengths. Those changes alter coverage as well as content. These six fixed pairs
address that confound on main E and its three single-layer ablations. No donor
is selected by observing its generated answer. Six targets are not 24 independent
replications. Run only after other GPU measurements have ended.
"""
import hashlib
import json
import argparse
from pathlib import Path
import random

import torch
from transformers import AutoTokenizer

from .benchmark import SEED
from .engine import Patch,capture,generate
from .model import ROOT,MODEL_DIR,load
from .prompts import tokenize
from .scoring_v2 import extract_answer,correct

PAIRS=[
    ("original",0,"What is 8 + 6?","14"),
    ("original",1,"If Sarah has 24 dollars and spends 9, how much does she have left?","15"),
    ("original",2,"What is 7 times 9?","63"),
    ("heldout_arithmetic",0,"What is 26 plus 53?","79"),
    ("heldout_arithmetic",1,"What is 24 times 16?","384"),
    ("heldout_arithmetic",2,"What is 16 percent of 450?","72"),
]
CONDITIONS=["E_v8_prefix","F_layer2","F_layer3","F_layer6"]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--main",default="independent_audit/results/mps_torch271_full")
    ap.add_argument("--out",default="independent_audit/results/equal_length_donor_followup")
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    args=ap.parse_args()
    folder=ROOT/args.main
    if not (folder/"completed.json").exists():
        raise ValueError("Wait for main benchmark and serialize GPU followups")
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    plan={"arguments":vars(args),"pairs":PAIRS,"conditions":CONDITIONS,"max_new_tokens":320,"seed":SEED,
          "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "timing_scope":"Generation only; fixed-span content control, not a new end-to-end cost comparison"}
    if (out/"plan.json").exists():
        # JSON normalizes tuple literals to lists; compare serialized plans.
        if json.loads((out/"plan.json").read_text())!=json.loads(json.dumps(plan)):
            raise ValueError("Changed plan requires new output directory")
    (out/"plan.json").write_text(json.dumps(plan,indent=2))
    rows=[json.loads(s) for s in (folder/"benchmark.jsonl").read_text().splitlines()]
    lookup={(r["suite"],r["task_index"],r["condition"]):r for r in rows if r["donor_variant"]=="same" and r["repeat"]==0}
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    for suite,index,donor_prompt,gold in PAIRS:
        ids=tokenize(tokenizer,donor_prompt,donor_prompt)["ids"]
        reference=lookup[(suite,index,"E_v8_prefix")]
        if ids.shape[1]!=reference["donor_input_tokens"]:
            raise ValueError(f"Donor length mismatch for {suite}:{index}; do not silently pad or truncate")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model,tok,meta=load(args.device,args.dtype)
    (out/"environment.json").write_text(json.dumps(meta,indent=2,default=str))
    path=out/"comparisons.jsonl"
    done={(r["suite"],r["task_index"],r["condition"]) for s in path.read_text().splitlines() if (r:=json.loads(s))} if path.exists() else set()
    for pair_index,(suite,index,donor_prompt,gold) in enumerate(PAIRS):
        donor=tokenize(tok,donor_prompt,donor_prompt)["ids"].to(model.cfg.device)
        order=CONDITIONS.copy()
        random.Random(SEED+pair_index).shuffle(order)
        for condition in order:
            if (suite,index,condition) in done:
                continue
            baseline=lookup[(suite,index,condition)]
            layers=baseline["patched_layers"]
            target=torch.tensor([baseline["input_ids"]],dtype=torch.long,device=model.cfg.device)
            source=capture(model,donor,layers)
            patch=Patch(source,baseline["target_positions"],baseline["donor_positions"])
            result=generate(model,target,320,tok.eos_token_id,patch=patch)
            result.pop("last_logits")
            text=tok.decode(result["token_ids"],skip_special_tokens=True).strip()
            answer=extract_answer(text)
            record={"suite":suite,"task_index":index,"condition":condition,"target_task":baseline["task"],
                    "new_donor_prompt":donor_prompt,"new_donor_answer":gold,"new_donor_ids":donor[0].tolist(),
                    "donor_length_matches":donor.shape[1]==baseline["donor_input_tokens"],
                    "target_input_ids":baseline["input_ids"],"target_positions":baseline["target_positions"],
                    "donor_positions":baseline["donor_positions"],"patch_map_unchanged":True,
                    "text":text,"answer":answer,"baseline_answer":extract_answer(baseline["text"]),
                    "answer_changed":answer!=extract_answer(baseline["text"]),"attracted_to_donor_answer":answer==gold,
                    "correct":correct(text,baseline["task"]["answer"]) and not result["truncated"],**result}
            with path.open("a") as f:
                f.write(json.dumps(record)+"\n")
            print(f"equal-length donor {suite} {index} {condition} answer={answer} target={baseline['task']['answer']}",flush=True)
    result_rows=[json.loads(s) for s in path.read_text().splitlines()]
    summary={"unique_targets":6,"comparisons":len(result_rows),
             "by_condition":{c:{"n":sum(r["condition"]==c for r in result_rows),
                                  "answer_changes":sum(r["condition"]==c and r["answer_changed"] for r in result_rows),
                                  "donor_answer_attraction":sum(r["condition"]==c and r["attracted_to_donor_answer"] for r in result_rows),
                                  "correct":sum(r["condition"]==c and r["correct"] for r in result_rows)} for c in CONDITIONS}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
