"""Export verified summary tables and a final artifact-integrity receipt.

This does not rerun or overwrite original experiments. Completion receipts are
coverage checks, not evidence of causal isolation or population noninferiority.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from .model import ROOT
from .report import integrity

BASE=ROOT/"independent_audit/results"
COMMIT="6c54c6926f508bb7515b96c52f54e03eb3949327"


def read(path):
    return json.loads(path.read_text())


def main():
    for name in ["postprocess_status.json","additional_measurements_status.json"]:
        if read(BASE/name)["stage"]!="measurements_complete":
            raise RuntimeError(f"Incomplete measurements: {name}")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    changed=subprocess.check_output(["git","diff","HEAD","--name-only"],cwd=ROOT,text=True).splitlines()
    if commit!=COMMIT or changed:
        raise RuntimeError(f"Upstream commit or tracked files changed: {commit}, {changed}")
    full=BASE/"mps_torch271_full"
    plan=read(full/"plan.json")
    frozen={k:hashlib.sha256((ROOT/"independent_audit"/k).read_bytes()).hexdigest()==v
            for k,v in plan["source_sha256"].items()}
    rows=[json.loads(s) for s in (full/"benchmark.jsonl").read_text().splitlines()]
    keys={tuple(r[k] for k in ["suite","task_index","condition","repeat","donor_variant"]) for r in rows}
    summary=read(full/"summary.json")
    parity=read(BASE/"code_cache_parity.json")
    equal=read(BASE/"equal_length_donor_followup/summary.json")
    partition=read(BASE/"prefix_partition_followup/completed.json")
    donor_rows=[json.loads(s) for s in (BASE/"equal_length_donor_followup/comparisons.jsonl").read_text().splitlines()]
    budget_dir=BASE/"budget_1024_followup"
    budget_rows=[json.loads(s) for s in (budget_dir/"benchmark.jsonl").read_text().splitlines()]
    lookup={(r["suite"],r["task_index"],r["condition"]):r for r in rows if r["donor_variant"]=="same"}
    required={"frozen_inference_sources":all(frozen.values()),
              "main_790_unique_rows":len(rows)==len(keys)==790,
              "main_no_generation_errors":not any(r["status"]=="error" for r in rows),
              "full_noop_alpha0_parity":summary["full_noop_and_alpha0_parity"]["all_identical"] is True,
              "all_50_original_vs_cached":len(summary["full_sequence_original_vs_cached"])==50 and all(r["token_identical"] for r in summary["full_sequence_original_vs_cached"]),
              "coding_native_and_patched_parity":all(parity[k] for k in ["all_token_identical","all_cached_match_main","all_native_baselines_match_matrix"]),
              "early_donor_bitwise":read(BASE/"donor_early_stop.json")["all_bitwise_equal"],
              "partition_111_rows":partition["rows"]==111,
              "fixed_length_donors_24":equal["comparisons"]==24,
              "all_donor_lengths_maps_fixed":all(r["donor_length_matches"] and r["patch_map_unchanged"] for r in donor_rows),
              "budget_symmetric_coverage":len(budget_rows)==4*len(read(budget_dir/"plan.json")["targets"]),
              "budget_only_changes_stopping_cap":all(r["token_ids"][:len(lookup[(r["suite"],r["task_index"],r["condition"])]["token_ids"])]==lookup[(r["suite"],r["task_index"],r["condition"])]["token_ids"] for r in budget_rows)}
    if not all(required.values()):
        raise RuntimeError(f"Verification failed: {required}")
    for folder in [full,BASE/"mps_torch271_timing_repeats",BASE/"budget_1024_followup"]:
        if not (folder/"summary.json").exists():
            continue
        data=read(folder/"summary.json")
        manifest=read(folder/"grading_manifest.json")
        if hashlib.sha256((folder/"benchmark.jsonl").read_bytes()).hexdigest()!=manifest["input_sha256"]:
            raise RuntimeError("Stale grading input; regenerate quality review and report")
        fields=["suite","condition","n_unique_tasks","correct","task_checks_passed_ignoring_response_truncation",
                "raw_primary_correct","automatic_task_checks_passed","truncated","mean_output_tokens","mean_task_median_total_seconds","mean_task_median_model_seconds"]
        with (folder/"conditions.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data["summaries"])
    # Hash the evidence, not local model binaries/environments or this receipt
    # itself. Model weights have a separately verified remote LFS SHA-256.
    artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(BASE.rglob("*")) if p.is_file() and p.name!="verification.json"
               and p.suffix in {".json",".jsonl",".csv",".md",".txt"}}
    # Enumerate source directories directly. Recursing through .venv would hash
    # third-party files and dramatically slow this small provenance receipt.
    source_dirs=[ROOT/"independent_audit",ROOT/"independent_audit/tests",ROOT/"independent_audit/failed_environment"]
    sources=sorted(p for folder in source_dirs for p in folder.iterdir() if p.is_file() and p.suffix in {".py",".toml",".lock",".md"})
    receipt={"upstream_commit":commit,"original_files":integrity(),"tracked_changes":changed,
             "verification_checks":required,"frozen_source_checks":frozen,"artifact_sha256":artifacts,
             "audit_source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             "scope":"Integrity and measurement coverage only. Consult FINAL_REPORT.md for quality, timing scope, selection and causal limits."}
    (BASE/"verification.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in {"artifact_sha256","audit_source_sha256"}},indent=2))


if __name__=="__main__":
    main()
