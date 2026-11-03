"""Select on development only; validate once; retain all failed candidates.

Optimize input+generated tokens, guarded by item-level correctness, EOS and
measured whole-request latency. Validation never chooses another candidate after
failure. Old tasks are explicitly regressions, not new independent evidence.
Do not pass answer/cases to infer(); quality scoring happens after the timer.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch

from independent_audit.code_quality import evaluate_code
from independent_audit.model import ROOT,load
from independent_audit.scoring_v2 import correct,extract_answer
from independent_audit.saved_audit import paired_stats
from .runtime import PROFILES,POLICY_PATH,infer
from .tasks import DEV,VALIDATION

SEED=202610293
INFERENCE_SOURCES=["lean_harness/runtime.py","independent_audit/engine.py","independent_audit/model.py","independent_audit/prompts.py"]
SOURCES=INFERENCE_SOURCES+["lean_harness/benchmark.py","lean_harness/tasks.py","independent_audit/code_quality.py","independent_audit/scoring_v2.py","independent_audit/evaluation.py"]


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=str))


def grade(result,task):
    detail=evaluate_code(result["text"],task) if task["kind"]=="code" else {
        "passed":correct(result["text"],task["answer"]),"answer":extract_answer(result["text"])}
    return {"task_checks_passed":bool(detail["passed"]),"checks_and_eos":bool(detail["passed"] and result["stopped_on_eos"]),"grade":detail}


def compare(baseline,candidate):
    by_id={r["task"]["id"]:r for r in baseline}
    if len(by_id)!=len(baseline) or set(by_id)!={r["task"]["id"] for r in candidate} or len(baseline)!=len(candidate):
        raise ValueError("Need unique, matched tasks")
    b=[by_id[r["task"]["id"]] for r in candidate]
    check_losses=[r["task"]["id"] for x,r in zip(b,candidate) if x["task_checks_passed"] and not r["task_checks_passed"]]
    eos_losses=[r["task"]["id"] for x,r in zip(b,candidate) if x["checks_and_eos"] and not r["checks_and_eos"]]
    bt,ct=sum(r["total_tokens"] for r in b),sum(r["total_tokens"] for r in candidate)
    bs,cs=sum(r["total_seconds"] for r in b),sum(r["total_seconds"] for r in candidate)
    return {"n":len(b),"task_check_regressions":check_losses,"checks_and_eos_regressions":eos_losses,
        "total_token_ratio":ct/bt,"total_seconds_ratio":cs/bs,
        "eligible":not check_losses and not eos_losses and any(r["checks_and_eos"] for r in candidate) and ct<bt and cs<bs,
        "paired_total_tokens":paired_stats([r["total_tokens"] for r in b],[r["total_tokens"] for r in candidate]),
        "scope":"Observed matched sample only; timing is shared-host wall time; no population accuracy guarantee."}


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row["stage"],row["task"]["kind"],row["profile"])].append(row)
    summaries=[]
    for (stage,kind,profile),rs in sorted(groups.items()):
        summaries.append({"stage":stage,"kind":kind,"profile":profile,"n":len(rs),
            "task_checks_passed":sum(r["task_checks_passed"] for r in rs),
            "checks_and_eos":sum(r["checks_and_eos"] for r in rs),
            "budget_reached":sum(r["truncated"] for r in rs),
            "mean_input_tokens":statistics.mean(r["input_tokens"] for r in rs),
            "mean_output_tokens":statistics.mean(r["generated_tokens_including_eos"] for r in rs),
            "mean_total_tokens":statistics.mean(r["total_tokens"] for r in rs),
            "mean_total_seconds":statistics.mean(r["total_seconds"] for r in rs)})
    return summaries


def regressions():
    path=ROOT/"independent_audit/results/mps_torch271_full/benchmark.jsonl"
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    tasks={}
    for row in rows:
        if row["donor_variant"]!="same" or row["condition"]!="C_concise":
            continue
        tasks[(row["suite"],row["task_index"])]=dict(row["task"],
            id=f"old_{row['suite']}_{row['task_index']:02d}",kind="code" if "function" in row["task"] else "scalar")
    if len(tasks)!=37:
        raise ValueError("Expected the preserved37-task regression suite")
    return [tasks[k] for k in sorted(tasks)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lean_harness/results/token_light_v1")
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    ap.add_argument("--policy-out",default="lean_harness/policy.json")
    args=ap.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    policy_path=ROOT/args.policy_out
    # Reject collisions before spending GPU time, rather than after a long run.
    if policy_path.exists():
        raise ValueError("Existing policy preserved; choose a new --policy-out for a new study")
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    old=regressions()
    plan={"arguments":vars(args),"seed":SEED,"profiles":list(PROFILES),"source_sha256":source_hashes,
        "development":DEV,"validation":VALIDATION,"regressions":old,"budgets":{"scalar":320,"code":1024},
        "selection":"Per kind: zero item-level regressions against legacy on dev, lower total input+output tokens and whole-request time; choose minimum summed total tokens, deterministic profile-name tie break.",
        "promotion":"No validation/regression quality losses, lower validation total tokens/time. Prior37-task regression gate additionally requires all candidate task checks and EOS, preserving the best observed result from old prompting controls. No validation-driven candidate replacement."}
    plan_path=out/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError("Changed plan/source; choose a fresh output directory")
    write(plan_path,plan)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model,tok,env=load(args.device,args.dtype)
    write(out/"environment.json",env)
    sanity=infer(model,tok,"What is 7 + 5?","scalar","legacy")
    if not correct(sanity["text"],"12"):
        raise RuntimeError("Backend semantic sanity failed")
    data_path=out/"generations.jsonl"
    rows=[json.loads(s) for s in data_path.read_text().splitlines()] if data_path.exists() else []
    done={(r["stage"],r["task"]["id"],r["profile"]) for r in rows}
    def run(stage,tasks,profiles_by_kind):
        for index,task in enumerate(tasks):
            order=list(dict.fromkeys(profiles_by_kind[task["kind"]]))
            random.Random(SEED+index).shuffle(order)
            for profile in order:
                if (stage,task["id"],profile) in done:
                    continue
                result=infer(model,tok,task["prompt"],task["kind"],profile)
                row={"stage":stage,"task":task,**result,**grade(result,task)}
                with data_path.open("a") as f:
                    f.write(json.dumps(row,allow_nan=False)+"\n")
                rows.append(row)
                done.add((stage,task["id"],profile))
                print(f"{stage} {index+1}/{len(tasks)} {task['id']} {profile} in={row['input_tokens']} out={row['generated_tokens_including_eos']} checks={row['task_checks_passed']} eos={row['stopped_on_eos']} {row['total_seconds']:.2f}s",flush=True)
    run("development",DEV,{kind:PROFILES for kind in ["scalar","code"]})
    selection={}
    for kind in ["scalar","code"]:
        subset=[r for r in rows if r["stage"]=="development" and r["task"]["kind"]==kind]
        baseline=[r for r in subset if r["profile"]=="legacy"]
        comparisons={p:compare(baseline,[r for r in subset if r["profile"]==p]) for p in ["lean","checked"]}
        eligible=[p for p in comparisons if comparisons[p]["eligible"]]
        chosen=min(eligible,key=lambda p:(comparisons[p]["total_token_ratio"],p)) if eligible else "legacy"
        selection[kind]={"profile":chosen,"comparisons":comparisons}
    selection_path=out/"selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:
        raise ValueError("Selection differs from previously frozen development decision")
    write(selection_path,selection)
    print("Selection frozen: "+json.dumps({k:v["profile"] for k,v in selection.items()}),flush=True)
    chosen={k:["legacy",v["profile"]] for k,v in selection.items()}
    run("validation",VALIDATION,chosen)
    run("regression",old,chosen)
    gates={}
    for kind in ["scalar","code"]:
        p=selection[kind]["profile"]
        comparisons={}
        for stage in ["validation","regression"]:
            subset=[r for r in rows if r["stage"]==stage and r["task"]["kind"]==kind]
            comparisons[stage]=compare([r for r in subset if r["profile"]=="legacy"],[r for r in subset if r["profile"]==p])
        regression_rows=[r for r in rows if r["stage"]=="regression" and r["task"]["kind"]==kind and r["profile"]==p]
        regressions_all_pass=all(r["checks_and_eos"] for r in regression_rows)
        val=comparisons["validation"]
        promoted=p!="legacy" and val["eligible"] and not comparisons["regression"]["task_check_regressions"] and not comparisons["regression"]["checks_and_eos_regressions"] and regressions_all_pass
        gates[kind]={"profile":p,"promoted":promoted,"old_best_results_preserved":regressions_all_pass,"comparisons":comparisons}
    result={"summaries":summarize(rows),"selection":selection,"gates":gates,"rows":len(rows),
        "data_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),"plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "limitations":"Small synthetic/task-level tests, not universal code quality. Selection on development only, validation unchanged after selection. No post-generation stripping, oracle retries or extra model verification calls."}
    write(out/"summary.json",result)
    policy={"environment":{k:env[k] for k in ["device","dtype","model_revision"]},"kinds":gates,
        "inference_source_sha256":{n:source_hashes[n] for n in INFERENCE_SOURCES},"evidence":str(out.relative_to(ROOT)),
        "summary_sha256":hashlib.sha256((out/"summary.json").read_bytes()).hexdigest(),"scope":result["limitations"]}
    policy_path.parent.mkdir(parents=True,exist_ok=True)
    write(policy_path,policy)
    write(out/"completed.json",{"rows":len(rows),"promoted":{k:v["promoted"] for k,v in gates.items()}})
    lines=["# Token-light prompt benchmark","",result["limitations"],"",
        "| Stage | Kind | Profile | Checks / EOS | Mean input | Mean output | Mean total tokens | Mean request s |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in result["summaries"]:
        lines.append(f"| {r['stage']} | {r['kind']} | {r['profile']} | {r['task_checks_passed']}/{r['n']} / {r['checks_and_eos']}/{r['n']} | {r['mean_input_tokens']:.2f} | {r['mean_output_tokens']:.2f} | {r['mean_total_tokens']:.2f} | {r['mean_total_seconds']:.3f} |")
    lines += ["", "Policy decisions:", ""]+[f"- {k}: selected {v['profile']}; promoted={v['promoted']}; old best results preserved={v['old_best_results_preserved']}." for k,v in gates.items()]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"rows":len(rows),"gates":gates},indent=2),flush=True)


if __name__=="__main__":
    main()
