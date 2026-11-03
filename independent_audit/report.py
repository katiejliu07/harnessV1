"""Paired, suite-specific summaries. Never pool repeat timings as new tasks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .evaluation import canonical,quality_gate
from .model import ROOT
from .saved_audit import paired_stats
from . import scoring_v2


def task_passed(row):
    return row.get("task_checks_passed",row.get("final_answer_or_code_tests_passed",row["correct"]))


def dimensional_quality_gate(base,test,base_cost,test_cost):
    # Joint failure can hide a functional regression: two capped responses can
    # both have correct=False even when only one contains working code. Guard
    # task checks independently, and never endorse a batch with no usable output.
    joint=quality_gate([r["correct"] for r in base],[r["correct"] for r in test],base_cost,test_cost)
    checks=quality_gate([task_passed(r) for r in base],[task_passed(r) for r in test],base_cost,test_cost)
    return {**joint,"observed_task_check_regressions":checks["observed_correctness_regressions"],
            "at_least_one_checks_and_eos_success":any(r["correct"] for r in test),
            "eligible_on_observed_sample":joint["eligible_on_observed_sample"] and checks["eligible_on_observed_sample"] and any(r["correct"] for r in test),
            "claim":"Observed-sample multi-dimension guard only; task-check and budget-termination losses are separate. No population noninferiority or general completeness established."}


def apply_adjudications(rows,document,raw_bytes):
    """Apply explicit human-reviewed cases only to the bound response snapshot."""
    if document["input_sha256"]!=hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError("Manual adjudications refer to different benchmark data")
    names=["suite","task_index","condition","repeat","donor_variant"]
    lookup={tuple(r[k] for k in names):r for r in rows}
    seen=set()
    for item in document["records"]:
        key=tuple(item[k] for k in names)
        if key in seen or key not in lookup or type(item["task_checks_passed"]) is not bool:
            raise ValueError("Invalid or duplicate manual adjudication key/verdict")
        row=lookup[key]
        if hashlib.sha256(row["text"].encode()).hexdigest()!=item["text_sha256"]:
            raise ValueError("Manual adjudication text hash mismatch")
        row["automatic_task_checks_passed"]=task_passed(row)
        row["task_checks_passed"]=item["task_checks_passed"]
        row["correct"]=item["task_checks_passed"] and not row["truncated"]
        seen.add(key)


def summarize(directory):
    # README commands pass repo-relative directories; normalize before producing
    # source labels so the CLI behaves like the absolute-path serial runner.
    directory=Path(directory).resolve()
    rows=[json.loads(s) for s in (directory/"benchmark.jsonl").read_text().splitlines()]
    supplemental=directory/"quality_diagnostics.json"
    rescored=0
    grading_manifest=None
    if supplemental.exists():
        grading_manifest=json.loads((directory/"grading_manifest.json").read_text())
        if hashlib.sha256((directory/"benchmark.jsonl").read_bytes()).hexdigest()!=grading_manifest["input_sha256"] or hashlib.sha256(supplemental.read_bytes()).hexdigest()!=grading_manifest["diagnostics_sha256"]:
            raise ValueError("Supplemental grades do not match raw data/diagnostics; rerun quality_review")
        for name,sha in grading_manifest.get("source_sha256",{}).items():
            if hashlib.sha256((ROOT/"independent_audit"/name).read_bytes()).hexdigest()!=sha:
                raise ValueError("Grader source changed; rerun quality_review before reporting")
        diagnostics={tuple(r[k] for k in ["suite","task_index","condition","repeat","donor_variant"]):r
                     for r in json.loads(supplemental.read_text())}
        for row in rows:
            key=tuple(row[k] for k in ["suite","task_index","condition","repeat","donor_variant"])
            diag=diagnostics.get(key,{})
            if "executable_and_complexity" in diag:
                row["raw_primary_correct"]=row["correct"]
                row["task_checks_passed"]=diag["executable_and_complexity"]["passed"]
                row["correct"]=row["task_checks_passed"] and not row["truncated"]
                rescored+=1
            elif "scalar_evaluation_v2" in diag:
                row["raw_primary_correct"]=row["correct"]
                row["raw_extracted_answer"]=row.get("extracted_answer")
                row["extracted_answer"]=diag["scalar_evaluation_v2"]["answer"]
                row["task_checks_passed"]=diag["scalar_evaluation_v2"]["passed"]
                row["correct"]=row["task_checks_passed"] and not row["truncated"]
    manual_path=directory/"manual_adjudications.json"
    manual=json.loads(manual_path.read_text()) if manual_path.exists() else None
    if manual is not None:
        apply_adjudications(rows,manual,(directory/"benchmark.jsonl").read_bytes())
    groups=defaultdict(dict)
    for row in rows:
        if row["donor_variant"]=="same" and row["status"]=="ok":
            groups[(row["suite"],row["condition"])].setdefault(row["task_index"],[]).append(row)
    summaries=[]
    for (suite,condition), tasks in sorted(groups.items()):
        representatives={i:sorted(rr,key=lambda r:r["repeat"])[0] for i,rr in tasks.items()}
        row={"suite":suite,"condition":condition,"n_unique_tasks":len(tasks),
             "correct":sum(r["correct"] for r in representatives.values()),
             "task_checks_passed_ignoring_response_truncation":sum(r.get("task_checks_passed",r.get("final_answer_or_code_tests_passed",r["correct"])) for r in representatives.values()),
             "raw_primary_correct":sum(r.get("raw_primary_correct",r["correct"]) for r in representatives.values()),
             "automatic_task_checks_passed":sum(r.get("automatic_task_checks_passed",task_passed(r)) for r in representatives.values()),
             "truncated":sum(r["truncated"] for r in representatives.values()),
             "mean_output_tokens":float(np.mean([r["generated_tokens_including_eos"] for r in representatives.values()])),
             "mean_task_median_total_seconds":float(np.mean([np.median([r["total_seconds"] for r in rr]) for rr in tasks.values()])),
             "mean_task_median_model_seconds":float(np.mean([np.median([r["model_seconds_including_donor"] for r in rr]) for rr in tasks.values()])),
             "repeat_token_disagreements":sum(any(r["token_ids"]!=rr[0]["token_ids"] for r in rr[1:]) for rr in tasks.values())}
        base=groups.get((suite,"B_reasoning"),{})
        common=sorted(set(tasks)&set(base))
        if common:
            b=[base[i][0] for i in common]
            p=[representatives[i] for i in common]
            bt=[float(np.median([r["total_seconds"] for r in base[i]])) for i in common]
            pt=[float(np.median([r["total_seconds"] for r in tasks[i]])) for i in common]
            row["comparison_to_reasoning"]={
                "token_stats":paired_stats([r["generated_tokens_including_eos"] for r in b],[r["generated_tokens_including_eos"] for r in p]),
                "seconds_stats":paired_stats(bt,pt),
                "quality_gate":dimensional_quality_gate(b,p,sum(bt),sum(pt)),
                "token_identical_tasks":sum(x["token_ids"]==y["token_ids"] for x,y in zip(b,p)),
                "loss_task_indices":[i for i,x,y in zip(common,b,p) if x["correct"] and not y["correct"]]}
        summaries.append(row)
    # B is an intentionally verbose instruction. Preserve comparisons to A/C so
    # outperforming that instruction is never mistaken for beating prompting.
    for row in summaries:
        suite,condition=row["suite"],row["condition"]
        current=groups[(suite,condition)]
        row["reference_comparisons"]={}
        union_solved=set()
        for reference in ["A_normal","B_reasoning","C_concise"]:
            refs=groups.get((suite,reference),{})
            common=sorted(set(current)&set(refs))
            if not common:
                continue
            br=[refs[i][0] for i in common]
            pr=[current[i][0] for i in common]
            bc=[r["correct"] for r in br]
            pc=[r["correct"] for r in pr]
            bt=sum(float(np.median([r["total_seconds"] for r in refs[i]])) for i in common)
            pt=sum(float(np.median([r["total_seconds"] for r in current[i]])) for i in common)
            row["reference_comparisons"][reference]={"n_paired_tasks":len(common),
                "quality_gate":dimensional_quality_gate(br,pr,bt,pt),
                "task_check_regression_indices":[i for i,b,p in zip(common,br,pr) if task_passed(b) and not task_passed(p)],
                "regression_indices":[i for i,b,p in zip(common,bc,pc) if b and not p]}
            union_solved.update(i for i in common if refs[i][0]["correct"])
        row["regressions_against_union_of_prompting_controls"]=[i for i in sorted(union_solved) if not current[i][0]["correct"]]
    probes=[]
    lookup={(r["suite"],r["task_index"],r["condition"],r["repeat"],r["donor_variant"]):r for r in rows if r["status"]=="ok"}
    for key,r in lookup.items():
        if r["donor_variant"]=="same":
            continue
        ref=lookup.get((*key[:4],"same"))
        if ref:
            target=r["task"].get("answer")
            donor=r["donor_task"].get("answer")
            probes.append({"suite":r["suite"],"task_index":r["task_index"],"condition":r["condition"],"variant":r["donor_variant"],
              "answer_changed":ref.get("extracted_answer")!=r.get("extracted_answer"),
              "tokens_changed":ref["token_ids"]!=r["token_ids"],"target_answer":target,"donor_answer":donor,
              "probe_answer":r.get("extracted_answer"),"same_donor_answer":ref.get("extracted_answer"),
              "same_donor_token_length":r["donor_input_tokens"]==ref["donor_input_tokens"],
              "same_patch_position_map":r["target_positions"]==ref["target_positions"] and r["donor_positions"]==ref["donor_positions"],
              "attracted_to_donor_answer":bool(donor is not None and canonical(target)!=canonical(donor) and r.get("extracted_answer")==canonical(donor)),
              "functional_pass_changed":task_passed(ref)!=task_passed(r),
              "checks_and_eos_status_changed":ref["correct"]!=r["correct"],
              "donor_function_present_without_target_name":bool("function" in r["task"] and
                  r["donor_task"]["function"] in re.findall(r"\bdef\s+(\w+)\s*\(",r["text"]) and
                  r["task"]["function"] not in re.findall(r"\bdef\s+(\w+)\s*\(",r["text"])),
              "correct_after_donor_change":r["correct"]})
    result={"summaries":summaries,"donor_probes":probes,"errors_or_skips":[r for r in rows if r["status"]!="ok"],
            "grading_manifest":grading_manifest,
            "manual_adjudications":manual,
            "supplemental_code_rescoring":{"rows":rescored,"source":"quality_diagnostics.json" if supplemental.exists() else None,
              "policy":"Accept permitted safe standard-library solutions; enforce executable edge cases, immutability and counted binary-search accesses. Raw benchmark.jsonl remains unchanged."},
            "supplemental_scalar_rescoring":"If quality_diagnostics.json exists, apply scoring_v2.py. It avoids interpreting 'our answer is correct' as a final-answer anchor. Preserve raw_primary_correct alongside rescored values.",
            "coverage":{"saved_rows":len(rows),"completed_marker":(directory/"completed.json").exists()},
            "interpretation":"Task-paired descriptive intervals. Original set was reused during selection. Small heldout/coding sets do not establish noninferiority; code tests cannot prove general completeness."}
    full_parity=[]
    for r in rows:
        if r["status"]!="ok" or r["condition"] not in {"D_noop","G_alpha0"}:
            continue
        reference=lookup.get((r["suite"],r["task_index"],"B_reasoning",r["repeat"],"same"))
        if reference:
            full_parity.append({"suite":r["suite"],"task_index":r["task_index"],"condition":r["condition"],
                                "donor_variant":r["donor_variant"],"repeat":r["repeat"],
                                "token_identical":reference["token_ids"]==r["token_ids"]})
    result["full_noop_and_alpha0_parity"]={"comparisons":full_parity,"all_identical":all(r["token_identical"] for r in full_parity) if full_parity else None}
    legacy_path=directory/"v8_original_path_reproduction.jsonl"
    if legacy_path.exists():
        legacy=[json.loads(s) for s in legacy_path.read_text().splitlines()]
        saved_v8=json.loads((ROOT/"results/activation_patching/quality_assessment_v8.json").read_text())["detailed_results"]
        result["original_path_reproduction"]={"n":len(legacy),
          "timing_scope":"Legacy total_seconds is generation plus donor forward for the patch, excluding tokenization and decoding. Use matching model_seconds_including_donor from the cached matrix for before/after speed ratios; main-matrix total_seconds is full request time.",
          "baseline_correct":sum(scoring_v2.correct(r["results"]["baseline"]["text"],r["task"]["answer"]) for r in legacy),
          "patched_correct":sum(scoring_v2.correct(r["results"]["intervened"]["text"],r["task"]["answer"]) for r in legacy),
          "token_stats":paired_stats([r["results"]["baseline"]["tokens"] for r in legacy],[r["results"]["intervened"]["tokens"] for r in legacy]),
          "total_seconds_stats":paired_stats([r["results"]["baseline"]["total_seconds"] for r in legacy],[r["results"]["intervened"]["total_seconds"] for r in legacy]),
          "response_text_matches_saved_v8":{side:sum(r["results"][side]["text"]==saved_v8[r["task_index"]][side]["text"] for r in legacy) for side in ["baseline","intervened"]}}
        comparison=[]
        for lr in legacy:
            for side,condition in [("baseline","B_reasoning"),("intervened","E_v8_prefix")]:
                nr=lookup.get(("original",lr["task_index"],condition,0,"same"))
                if nr:
                    comparison.append({"task_index":lr["task_index"],"side":side,
                      "token_identical":nr["token_ids"]==lr["results"][side]["token_ids"],
                      "original_correct":scoring_v2.correct(lr["results"][side]["text"],lr["task"]["answer"]),"cached_correct":nr["correct"],
                      "original_generation_plus_donor_seconds":lr["results"][side]["total_seconds"],
                      "cached_generation_plus_donor_seconds":nr["model_seconds_including_donor"],
                      "cached_end_to_end_seconds":nr["total_seconds"]})
        result["full_sequence_original_vs_cached"]=comparison
    (directory/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    source_label=directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory
    lines=["# Independent benchmark results", "",f"Source: `{source_label}`. Task-specific costs include donor forwards; output tokens include EOS.","",
           f"Supplemental executable/complexity grading applied to {rescored} code rows; raw primary scores remain available in JSON. The success column requires passed task checks AND EOS before the cap. JSON separately reports checks passed even when the budget is reached. EOS is an operational stopping criterion, not proof of semantic completeness. Scalar scoring does not establish reasoning soundness.","",
           "| Suite | Condition | Task checks passed | Checks passed & EOS | Budget reached | Mean output tokens | Mean task median total s |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['suite']} | {r['condition']} | {r['task_checks_passed_ignoring_response_truncation']}/{r['n_unique_tasks']} | {r['correct']}/{r['n_unique_tasks']} | {r['truncated']} | {r['mean_output_tokens']:.2f} | {r['mean_task_median_total_seconds']:.3f} |")
    if manual is not None:
        lines += ["",f"Explicit manual adjudications applied to {len(manual['records'])} response(s); automatic task-check counts and all evidence remain in summary.json and manual_adjudications.json."]
    scalar_probes=[p for p in probes if p["target_answer"] is not None]
    code_probes=[p for p in probes if p["suite"]=="coding"]
    lines += ["",f"Scalar donor probes: {len(scalar_probes)}; answers changed in {sum(p['answer_changed'] for p in scalar_probes)}; donor-answer attraction in {sum(p['attracted_to_donor_answer'] for p in scalar_probes)}.",
              f"Coding donor probes: {len(code_probes)}; functional pass status changed in {sum(p['functional_pass_changed'] for p in code_probes)}; donor function name appeared without the target name in {sum(p['donor_function_present_without_target_name'] for p in code_probes)}.",
              "", "These are observed sample results, not evidence of a style-specific mechanism or broad preservation of code quality."]
    if "original_path_reproduction" in result:
        legacy=result["original_path_reproduction"]
        lines += ["","## Original v8 generation paths", "",
                  f"{legacy['n']} questions: baseline {legacy['baseline_correct']} correct, patched {legacy['patched_correct']} correct.",
                  f"Mean token reduction: {legacy['token_stats']['mean_difference']:.2f}. Mean generation-plus-donor time reduction: {legacy['total_seconds_stats']['mean_difference']:.3f} s (negative means patching is slower). Tokenization/decoding are excluded from these legacy timings.",
                  "", "Full-sequence native/manual/cached comparisons are retained in summary.json; short warmup parity alone does not establish long-generation parity."]
    (directory/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return result


def integrity():
    manifest=json.loads((ROOT/"independent_audit/results/original_manifest.json").read_text())
    changed=[name for name,sha in manifest.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    if changed:
        raise RuntimeError(f"Original artifacts changed: {changed}")
    return {"files_verified":len(manifest),"changed":changed}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("directory",type=Path)
    args=ap.parse_args()
    summarize(args.directory)
    print(integrity())
