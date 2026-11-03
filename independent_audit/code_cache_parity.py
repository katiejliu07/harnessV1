"""Check full-sequence versus KV-cached v8 outputs on the actual coding tasks.

Use saved input IDs so this isolates generation instead of changing prompts.
The uncached path is the original manual generator. Donor tensors are shared
and preparation excluded from both generation timers; do not label these times
end-to-end request costs. Different outputs must be reviewed, never hidden.
"""
import json
import time
import argparse

import torch

from .benchmark import original_module,SEED
from .code_quality import evaluate_code
from .engine import Patch,generate,synchronize
from .model import ROOT,load


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-stem",default="independent_audit/results/code_cache_parity")
    ap.add_argument("--main",default="independent_audit/results/mps_torch271_full")
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    args=ap.parse_args()
    stem=ROOT/args.out_stem
    path=stem.with_name(stem.name+"_cases.jsonl")
    if path.exists():
        raise ValueError("Existing parity cases preserved; use a new --out-stem")
    folder=ROOT/args.main
    if not (folder/"completed.json").exists():
        raise ValueError("Wait for the main GPU benchmark before this serial check")
    rows=[json.loads(s) for s in (folder/"benchmark.jsonl").read_text().splitlines()]
    baseline_lookup={r["task_index"]:r for r in rows if r["suite"]=="coding" and r["condition"]=="B_reasoning" and r["donor_variant"]=="same" and r["repeat"]==0}
    selected=[r for r in rows if r["suite"]=="coding" and r["condition"]=="E_v8_prefix" and r["donor_variant"]=="same" and r["repeat"]==0]
    if len(selected)!=4:
        raise ValueError("Need all four actual coding cases")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model,tok,meta=load(args.device,args.dtype)
    original=original_module()
    cases=[]
    # This small check runs once after measurements; existing runs stay intact.
    for row in sorted(selected,key=lambda r:r["task_index"]):
        ids=torch.tensor([row["input_ids"]],dtype=torch.long,device=model.cfg.device)
        baseline=baseline_lookup[row["task_index"]]
        if baseline["input_ids"]!=row["input_ids"]:
            raise ValueError("Baseline and patched condition use different input IDs")
        donor_ids=torch.tensor([row["donor_ids"]],dtype=torch.long,device=model.cfg.device)
        source=original.get_layer_outputs(model,donor_ids,[6,2,3],"resid_post")
        ti,di=row["target_positions"],row["donor_positions"]
        hooks=[(f"blocks.{i}.hook_resid_post",original.patch_layer_hook(i,source,1.,0,ids.shape[1])) for i in [6,2,3]]
        synchronize(model.cfg.device)
        start=time.perf_counter()
        full=original.manual_generate(model,ids,320,False,1.,1.,tok.eos_token_id,hooks,None)
        synchronize(model.cfg.device)
        full_seconds=time.perf_counter()-start
        full_ids=full[0,ids.shape[1]:].tolist()
        cached=generate(model,ids,320,tok.eos_token_id,patch=Patch(source,ti,di))
        case={"task_index":row["task_index"],"task":row["task"],
              "full_token_ids":full_ids,"cached_token_ids":cached["token_ids"],
              "full_text":tok.decode(full_ids,skip_special_tokens=True).strip(),
              "cached_text":tok.decode(cached["token_ids"],skip_special_tokens=True).strip(),
              "full_generation_seconds":full_seconds,"cached_generation_seconds":cached["generation_seconds"],
              "full_and_cached_token_identical":full_ids==cached["token_ids"],
              "cached_matches_main_benchmark":cached["token_ids"]==row["token_ids"]}
        case["full_code_tests"]=evaluate_code(case["full_text"],row["task"])
        case["cached_code_tests"]=evaluate_code(case["cached_text"],row["task"])
        # Broaden ordinary-native versus custom-hook-path validation to complete
        # programming generations, rather than trusting a short arithmetic check.
        synchronize(model.cfg.device)
        native_start=time.perf_counter()
        native=model.generate(ids,max_new_tokens=320,do_sample=False,temperature=1.,top_p=1.,
                              eos_token_id=tok.eos_token_id,return_type="tokens",verbose=False)
        synchronize(model.cfg.device)
        case["native_baseline_generation_seconds"]=time.perf_counter()-native_start
        native_ids=native[0,ids.shape[1]:].tolist()
        case["native_baseline_token_ids"]=native_ids
        case["native_baseline_text"]=tok.decode(native_ids,skip_special_tokens=True).strip()
        case["native_baseline_matches_matrix"]=native_ids==baseline["token_ids"]
        case["native_baseline_code_tests"]=evaluate_code(case["native_baseline_text"],row["task"])
        with path.open("a") as f:
            f.write(json.dumps(case)+"\n")
        cases.append(case)
        print(f"code cache parity {row['task_index']+1}/4 exact={case['full_and_cached_token_identical']} full={full_seconds:.2f}s cached={cached['generation_seconds']:.2f}s",flush=True)
    result={"arguments":vars(args),"environment":meta,"cases":cases,
            "all_token_identical":all(r["full_and_cached_token_identical"] for r in cases),
            "all_cached_match_main":all(r["cached_matches_main_benchmark"] for r in cases),
            "all_native_baselines_match_matrix":all(r["native_baseline_matches_matrix"] for r in cases),
            "scope":"Four coding tasks; generation-only timing, same precomputed donor, original full-sequence versus cached prompt-only patch. Tests are finite and do not certify general code quality."}
    stem.with_suffix(".json").write_text(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    main()
