"""Preregistered controls; resumable JSONL preserves every attempted generation.

Do not tune configurations on this benchmark. New hypotheses require new frozen
tasks. All conditions use the same model, dtype, greedy argmax and 320-token cap.
Repeat observations are NOT independent accuracy trials. No CUDA claims on MPS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import resource
import sys
import time
from pathlib import Path

import torch

from .engine import Patch,capture,generate,synchronize
from .evaluation import canonical,correct,extract_answer
from .model import ROOT,load
from .prompts import REASONING,CONCISE,CODE_REASONING,CODE_CONCISE,tokenize,position_map,equal_length_concise
from .tasks import HELDOUT,CODING,score_code

SEED=20261029
CONDITIONS=["A_normal","B_reasoning","C_concise","C_length_matched","D_noop",
            "E_v8_prefix","F_layer2","F_layer3","F_layer6","G_random_layer",
            "G_shuffled","G_unrelated","G_alpha0","H_instruction","H_question","H_suffix"]
PATCHED=CONDITIONS[5:]


def choose_donor(items,index,condition,variant):
    unrelated=condition=="G_unrelated"
    if variant=="same" and not unrelated:
        return None
    candidates=[items[(index+j)%len(items)] for j in range(1,len(items))
                if items[(index+j)%len(items)].get("answer") != items[index].get("answer") or "function" in items[index]]
    if not candidates:
        raise ValueError("Need a different-answer donor; expand the task set")
    rank={"same":0,"different_1":1 if unrelated else 0,"different_2":2 if unrelated else 1}[variant]
    if rank>=len(candidates):
        raise ValueError("Insufficient distinct donor questions for leakage probes")
    return candidates[rank]


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,default=str,allow_nan=False))


def original_module():
    sys.path.insert(0,str(ROOT / "code"))
    import quality_assessment_v8 as module
    return module


def run_condition(model,tok,task,condition,donor_task=None,max_tokens=320):
    synchronize(model.cfg.device)
    start=time.perf_counter()
    cpu_start=time.process_time()
    usage_start=resource.getrusage(resource.RUSAGE_SELF)
    coding="function" in task
    question=task["prompt"]
    if condition=="A_normal":
        prompt=question
    elif condition=="C_concise":
        prompt=(CODE_CONCISE if coding else CONCISE).format(prompt=question)
    else:
        prompt=(CODE_REASONING if coding else REASONING).format(prompt=question)
    target=tokenize(tok,prompt,question)
    if condition=="C_length_matched":
        target=equal_length_concise(tok,question,target["ids"].shape[1],coding)
        if target is None:
            return {"status":"skipped","reason":"No exact-length rendered concise control found"}
    patch=None
    layers=[6,2,3]
    donor=None
    donor_seconds=0.
    ti,di=[],[]
    if condition in PATCHED:
        if condition.startswith("F_layer"):
            layers=[int(condition[-1])]
        if condition=="G_random_layer":
            layers=[random.Random(SEED).choice([i for i in range(model.cfg.n_layers) if i not in [2,3,6]])]
        dq=(donor_task or task)["prompt"]
        donor_prompt=(CODE_CONCISE if coding else CONCISE).format(prompt=dq) if condition=="H_instruction" else dq
        donor=tokenize(tok,donor_prompt,dq)
        mode={"H_instruction":"instruction","H_question":"question","H_suffix":"suffix"}.get(condition,"prefix")
        ti,di=position_map(target,donor,mode)
        if condition=="G_shuffled":
            random.Random(SEED).shuffle(di)
        synchronize(model.cfg.device)
        donor_start=time.perf_counter()
        states=capture(model,donor["ids"].to(model.cfg.device),layers)
        synchronize(model.cfg.device)
        donor_seconds=time.perf_counter()-donor_start
        patch=Patch(states,ti,di,0. if condition=="G_alpha0" else 1.)
    result=generate(model,target["ids"].to(model.cfg.device),max_tokens,tok.eos_token_id,
                    patch=patch,noop_layers=[6,2,3] if condition=="D_noop" else ())
    result.pop("last_logits")
    text=tok.decode(result["token_ids"],skip_special_tokens=True).strip()
    synchronize(model.cfg.device)
    total=time.perf_counter()-start
    cpu_seconds=time.process_time()-cpu_start
    usage_end=resource.getrusage(resource.RUSAGE_SELF)
    verdict=score_code(text,task) if coding else {"passed":correct(text,task["answer"])}
    return {"status":"ok","text":text,"extracted_answer":None if coding else extract_answer(text),
            "final_answer_or_code_tests_passed":verdict["passed"],
            "correct":verdict["passed"] and not result["truncated"],"code_test":verdict if coding else None,
            "total_seconds":total,"donor_forward_seconds":donor_seconds,
            "process_cpu_seconds":cpu_seconds,
            "minor_page_faults":usage_end.ru_minflt-usage_start.ru_minflt,
            "major_page_faults":usage_end.ru_majflt-usage_start.ru_majflt,
            "model_seconds_including_donor":donor_seconds+result["generation_seconds"],
            "input_tokens":target["ids"].shape[1],"donor_input_tokens":donor["ids"].shape[1] if donor else 0,
            "input_ids":target["ids"][0].tolist(),"prompt_text":target["rendered"],
            "donor_ids":donor["ids"][0].tolist() if donor else [],"donor_prompt":donor["rendered"] if donor else None,
            "patched_layers":layers if patch else [],"target_positions":ti,"donor_positions":di,
            "patched_instruction_token_count":len(set(ti)&set(target["instruction"])),
            "patched_question_token_count":len(set(ti)&set(target["question"])),
            "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mps_current_allocated_bytes_not_peak":torch.mps.current_allocated_memory() if str(model.cfg.device).startswith("mps") else None,
            "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            **result}


def parity(model,tok,out,max_tokens=12):
    q="What is 7 + 5?"
    sanity=generate(model,tokenize(tok,q,q)["ids"].to(model.cfg.device),32,tok.eos_token_id)
    sanity_text=tok.decode(sanity["token_ids"],skip_special_tokens=True).strip()
    if not correct(sanity_text,"12"):
        dump(out/"invalid_backend.json",{"reason":"Unpatched semantic sanity failed","text":sanity_text})
        raise RuntimeError("Unpatched model failed semantic sanity; do not benchmark interventions")
    item=tokenize(tok,REASONING.format(prompt=q),q)
    ids=item["ids"].to(model.cfg.device)
    reference=generate(model,ids,max_tokens,tok.eos_token_id)
    with torch.inference_mode():
        logits=model(ids)[:,-1,:]
        with model.hooks(fwd_hooks=[(f"blocks.{i}.hook_resid_post",lambda x,hook:x) for i in [6,2,3]]):
            hook_logits=model(ids)[:,-1,:]
        native=model.generate(ids,max_new_tokens=max_tokens,do_sample=False,temperature=1.,top_p=1.,
                              return_type="tokens",verbose=False,eos_token_id=tok.eos_token_id)
    noop=generate(model,ids,max_tokens,tok.eos_token_id,noop_layers=[6,2,3])
    donor=capture(model,tokenize(tok,q,q)["ids"].to(model.cfg.device),[6,2,3])
    ti,di=position_map(item,tokenize(tok,q,q),"prefix")
    patch=Patch(donor,ti,di)
    cached=generate(model,ids,max_tokens,tok.eos_token_id,patch=patch)
    uncached=generate(model,ids,max_tokens,tok.eos_token_id,patch=patch,cached=False)
    alpha0=generate(model,ids,max_tokens,tok.eos_token_id,patch=Patch(donor,ti,di,0.))
    original=original_module()
    hooks=[(f"blocks.{i}.hook_resid_post",original.patch_layer_hook(i,donor,1.,0,ids.shape[1])) for i in [6,2,3]]
    old_tokens=original.manual_generate(model,ids,max_tokens,False,1.,1.,tok.eos_token_id,hooks,None)
    checks={"semantic_sanity_text":sanity_text,"noop_prompt_logits_max_abs":float((logits-hook_logits).abs().max()),
            "noop_token_parity":reference["token_ids"]==noop["token_ids"],
            "alpha_zero_token_parity":reference["token_ids"]==alpha0["token_ids"],
            "native_token_parity":reference["token_ids"]==native[0,ids.shape[1]:].tolist(),
            "cached_vs_uncached_patch_tokens":cached["token_ids"]==uncached["token_ids"],
            "cached_vs_original_patch_tokens":cached["token_ids"]==old_tokens[0,ids.shape[1]:].tolist(),
            "patch_last_logits_max_abs":float((cached["last_logits"]-uncached["last_logits"]).abs().max()),
            "cached_seconds":cached["generation_seconds"],"uncached_seconds":uncached["generation_seconds"],
            "cached_query_tokens":cached["forward_query_tokens"],"uncached_query_tokens":uncached["forward_query_tokens"],
            "tested_new_tokens":max_tokens,"scope":"One warmup prompt; full-run no-op/alpha-zero parity is also checked in report"}
    dump(out/"parity.json",checks)
    if checks["noop_prompt_logits_max_abs"] != 0 or not all(checks[k] for k in checks if k.endswith("parity") or k.endswith("_tokens") and k.startswith("cached_vs")):
        raise RuntimeError(f"Parity prerequisite failed: {checks}")
    return checks


def reproduce_v8(model,tok,tasks,out,max_tokens=320):
    original=original_module()
    path=out/"v8_original_path_reproduction.jsonl"
    done={json.loads(s)["task_index"] for s in path.read_text().splitlines()} if path.exists() else set()
    for i,task in enumerate(tasks):
        if i in done:
            continue
        reason=tokenize(tok,REASONING.format(prompt=task["prompt"]),task["prompt"])["ids"].to(model.cfg.device)
        direct=tokenize(tok,task["prompt"],task["prompt"])["ids"].to(model.cfg.device)
        synchronize(model.cfg.device)
        start=time.perf_counter()
        donor=original.get_layer_outputs(model,direct,[6,2,3],"resid_post")
        synchronize(model.cfg.device)
        donor_seconds=time.perf_counter()-start
        row={"task_index":i,"task":task,"donor_seconds":donor_seconds,"results":{}}
        for side in ["baseline","intervened"]:
            hooks=[] if side=="baseline" else [(f"blocks.{j}.hook_resid_post",original.patch_layer_hook(j,donor,1.,0,reason.shape[1])) for j in [6,2,3]]
            synchronize(model.cfg.device)
            st=time.perf_counter()
            output=original.generate_tokens(model,reason,max_tokens,False,1.,1.,tok.eos_token_id,fwd_hooks=hooks)
            synchronize(model.cfg.device)
            wall=time.perf_counter()-st
            text,n=original.decode_response(tok,output,reason.shape[1])
            row["results"][side]={"text":text,"tokens":n,"token_ids":output[0,reason.shape[1]:].tolist(),
                "generation_seconds":wall,"total_seconds":wall+(donor_seconds if side=="intervened" else 0),
                "original_extracted_answer":original.extract_answer(text),"independent_extracted_answer":extract_answer(text),
                "original_correct":original.extract_answer(text)==task["answer"],"independent_correct":correct(text,task["answer"])}
        with path.open("a") as f:
            f.write(json.dumps(row)+"\n")
        print(f"reproduction {i+1}/{len(tasks)} base={row['results']['baseline']['tokens']} patched={row['results']['intervened']['tokens']}",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="auto")
    ap.add_argument("--dtype",default="float16")
    ap.add_argument("--out",default="independent_audit/results/mps_full")
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--max-new-tokens",type=int,default=320)
    ap.add_argument("--repeats",type=int,default=1)
    ap.add_argument("--conditions",nargs="+",default=CONDITIONS)
    ap.add_argument("--donor-probes",type=int,default=3)
    ap.add_argument("--reproduce",action="store_true")
    ap.add_argument("--extensions",action="store_true")
    args=ap.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    spec=vars(args)
    spec.update(seed=SEED,conditions=args.conditions,reasoning_template=REASONING,
                selection_policy="Frozen controls; report all, no early EFFECTIVE stopping")
    spec["source_sha256"]={name:hashlib.sha256((ROOT/"independent_audit"/name).read_bytes()).hexdigest()
                           for name in ["benchmark.py","engine.py","model.py","prompts.py","tasks.py","evaluation.py"]}
    plan=out/"plan.json"
    if plan.exists() and json.loads(plan.read_text()) != spec:
        raise ValueError("Existing output directory belongs to a different plan; choose a new --out")
    dump(plan,spec)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model,tok,meta=load(args.device,args.dtype)
    meta["generation"]["max_new_tokens"]=args.max_new_tokens
    meta["chat_template_sha256"]=hashlib.sha256(tok.chat_template.encode()).hexdigest()
    dump(out/"environment.json",meta)
    print("Loaded; checking parity",flush=True)
    parity(model,tok,out)
    print("Parity checks passed",flush=True)
    original=original_module()
    tasks=original.TEST_PROBLEMS[:args.limit or None]
    if args.reproduce:
        reproduce_v8(model,tok,tasks,out,args.max_new_tokens)
    sets=[("original",tasks)]
    if args.extensions:
        sets += [("heldout_arithmetic",HELDOUT),("coding",CODING)]
    path=out/"benchmark.jsonl"
    existing=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={tuple(r[k] for k in ["suite","task_index","condition","repeat","donor_variant"]) for r in existing}
    for suite,items in sets:
        for i,task in enumerate(items):
            for repeat in range(args.repeats):
                order=args.conditions.copy()
                random.Random(SEED+i+repeat*1000).shuffle(order)
                for condition in order:
                    variants=["same"]
                    if i < args.donor_probes and condition in PATCHED:
                        variants += ["different_1","different_2"]
                    for variant in variants:
                        key=(suite,i,condition,repeat,variant)
                        if key in done:
                            continue
                        donor_task=choose_donor(items,i,condition,variant)
                        row={"suite":suite,"task_index":i,"task":task,"condition":condition,"repeat":repeat,
                             "donor_variant":variant,"donor_task":donor_task or task}
                        try:
                            row.update(run_condition(model,tok,task,condition,donor_task,args.max_new_tokens))
                        except Exception as exc:
                            row.update(status="error",error=repr(exc))
                            with path.open("a") as f:
                                f.write(json.dumps(row)+"\n")
                            raise
                        with path.open("a") as f:
                            f.write(json.dumps(row)+"\n")
                        print(f"{suite} {i+1}/{len(items)} {condition} {variant} {row.get('generated_tokens_including_eos')}t correct={row.get('correct')} total={row.get('total_seconds',0):.2f}s",flush=True)
    dump(out/"completed.json",{"status":"completed","rows":len(path.read_text().splitlines()),"ended_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})


if __name__=="__main__":
    main()
