"""Summarize fixed-map causal partitions, separating analytical identities.

These are nonadditive interventions, not an attribution summing to E's effect.
Never attach measured cost savings to the empty masks reused analytically.
"""
from collections import defaultdict
import hashlib
import json
import statistics
from pathlib import Path

from .model import ROOT
from .code_quality import evaluate_code
from .scoring_v2 import correct


def main():
    folder=ROOT/"independent_audit/results/prefix_partition_followup"
    if not (folder/"completed.json").exists():
        raise ValueError("Wait for all partition measurements")
    plan=json.loads((folder/"plan.json").read_text())
    for name,sha in plan["supporting_source_sha256"].items():
        if hashlib.sha256((ROOT/"independent_audit"/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f"Partition supporting source changed: {name}")
    rows=[json.loads(s) for s in (folder/"partitions.jsonl").read_text().splitlines()]
    main_path=ROOT/plan["arguments"]["main"]/"benchmark.jsonl"
    full=[json.loads(s) for s in main_path.read_text().splitlines()]
    lookup={(r["suite"],r["task_index"]):r for r in full if r["condition"]=="E_v8_prefix" and r["donor_variant"]=="same" and r["repeat"]==0}
    by_task=defaultdict(list)
    by_group=defaultdict(list)
    for row in rows:
        key=(row["suite"],row["task_index"])
        by_task[key].append(row)
        check=evaluate_code(row["text"],row["task"])["passed"] if "function" in row["task"] else correct(row["text"],row["task"]["answer"])
        by_group[(row["suite"],row["partition"])].append({**row,"task_checks_passed":check,
            "same_tokens_as_full_patch":row["token_ids"]==lookup[key]["token_ids"]})
    if len(rows)!=111 or len(by_task)!=37:
        raise ValueError("Incomplete or duplicate partition coverage")
    for key,parts in by_task.items():
        pairs=[(t,d) for r in parts for t,d in zip(r["target_positions"],r["donor_positions"])]
        expected=list(zip(lookup[key]["target_positions"],lookup[key]["donor_positions"]))
        if len(parts)!=3 or sorted(pairs)!=sorted(expected) or len({t for t,d in pairs})!=len(pairs):
            raise ValueError("Partitions do not disjointly recover original E map")
    summaries=[]
    for (suite,part),items in sorted(by_group.items()):
        summaries.append({"suite":suite,"partition":part,"n":len(items),
            "analytical_noops":sum(r["analytical_noop"] for r in items),
            "measured_generations":sum(not r["analytical_noop"] for r in items),
            "task_checks_passed":sum(r["task_checks_passed"] for r in items),
            "checks_and_eos":sum(r["correct"] for r in items),
            "budget_reached":sum(r["truncated"] for r in items),
            "same_tokens_as_baseline":sum(r["same_tokens_as_baseline"] for r in items),
            "same_tokens_as_full_patch":sum(r["same_tokens_as_full_patch"] for r in items),
            "mean_tokens":statistics.mean(len(r["token_ids"]) for r in items)})
    result={"rows":len(rows),"summaries":summaries,
        "partition_data_sha256":hashlib.sha256((folder/"partitions.jsonl").read_bytes()).hexdigest(),
        "main_data_sha256":hashlib.sha256(main_path.read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"Fixed original donor and position map. Disjoint masks, nonadditive effects; empty masks are analytical identities with no measured latency."}
    (folder/"summary.json").write_text(json.dumps(result,indent=2))
    lines=["# Fixed-map partition results","",result["scope"],"",
        "| Suite | Partition | Task checks | Checks + EOS | Measured / analytical | Same as B / E | Mean tokens |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['suite']} | {r['partition']} | {r['task_checks_passed']}/{r['n']} | {r['checks_and_eos']}/{r['n']} | {r['measured_generations']} / {r['analytical_noops']} | {r['same_tokens_as_baseline']} / {r['same_tokens_as_full_patch']} | {r['mean_tokens']:.2f} |")
    (folder/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
