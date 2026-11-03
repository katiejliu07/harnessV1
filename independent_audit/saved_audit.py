"""Recalculate saved evidence without rewriting any original artifact."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from .evaluation import correct, extract_answer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "independent_audit" / "results"


def original_function(name):
    # Execute only named, dependency-free AST definitions, never an experiment main().
    tree = ast.parse((ROOT / "code/quality_assessment_v8.py").read_text())
    wanted = {"normalize_number", "extract_answer", "count_reasoning_markers", "summarize_results"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign)) and
             (getattr(n, "name", None) in wanted or isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id in {"REASONING_CONNECTORS", "MARKER_METHOD"} for t in n.targets))]
    ns = {"re": __import__("re")}
    exec(compile(ast.Module(nodes, type_ignores=[]), "original_v8_definitions", "exec"), ns)
    return ns[name]


def paired_stats(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.
    test = stats.ttest_rel(a, b) if len(diff) > 1 and sd else None
    # Fixed resampling seed; these intervals condition on a reused, selected set.
    rng = np.random.default_rng(20261029)
    boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
    return {"n": len(diff), "baseline_mean": float(np.mean(a)), "intervened_mean": float(np.mean(b)),
            "mean_difference": float(diff.mean()), "sd_difference": sd,
            "paired_t": float(test.statistic) if test else None,
            "p_value_descriptive_post_selection": float(test.pvalue) if test else None,
            "cohens_dz": float(diff.mean()/sd) if sd else None,
            "paired_bootstrap_95": np.quantile(boot, [.025, .975]).tolist()}


def walks(obj, path="root"):
    if isinstance(obj, dict):
        if isinstance(obj.get("detailed_results"), list) and obj["detailed_results"]:
            yield path, obj
        for k, v in obj.items():
            if k != "detailed_results":
                yield from walks(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walks(v, f"{path}[{i}]")


def audit_record(data):
    rows = data["detailed_results"]
    rows = [r for r in rows if {"baseline", "intervened", "expected_answer"} <= r.keys()]
    if not rows:
        return None
    old_extract, markers = original_function("extract_answer"), original_function("count_reasoning_markers")
    discrepancies, bc, pc = [], [], []
    for i, row in enumerate(rows):
        for side in ["baseline", "intervened"]:
            item = row[side]
            if "text" not in item:
                continue
            score = correct(item["text"], row["expected_answer"])
            (bc if side == "baseline" else pc).append(score)
            checks = {"original_extraction_replay": old_extract(item["text"]),
                      "independent_extraction": extract_answer(item["text"]),
                      "saved_extraction": item.get("extracted_answer"),
                      "saved_correct": item.get("correct"), "independent_correct": score,
                      "saved_markers": item.get("reasoning_markers"), "replayed_markers": markers(item["text"])}
            if checks["saved_correct"] != score or checks["saved_markers"] != checks["replayed_markers"]:
                discrepancies.append({"index": i, "side": side, "problem": row.get("problem"), **checks})
    token_stats = paired_stats([r["baseline"]["tokens"] for r in rows], [r["intervened"]["tokens"] for r in rows])
    marker_stats = paired_stats([markers(r["baseline"]["text"]) for r in rows], [markers(r["intervened"]["text"]) for r in rows])
    losses = sum(b and not p for b, p in zip(bc, pc))
    gains = sum(p and not b for b, p in zip(bc, pc))
    return {"n": len(rows), "saved_summary": data.get("summary"), "token_stats": token_stats,
            "marker_stats": marker_stats, "baseline_correct": sum(bc), "intervened_correct": sum(pc),
            "marker_rescoring_scope": "Common v8 marker estimator. Earlier artifacts can intentionally use other marker definitions; differences alone are not historical bugs.",
            "answer_rescoring_scope": "Common independent final-scalar estimator; all 50 main-v8 final answers separately inspected/regression-tested. Other versions may require manual adjudication.",
            "paired_accuracy_losses": losses, "paired_accuracy_gains": gains,
            "mcnemar_exact_p": float(stats.binomtest(min(losses,gains),losses+gains,.5).pvalue) if losses+gains else 1.,
            "discrepancies": discrepancies,
            "token_count_limits": "Counts replay stored integers; text re-tokenization is separate and cannot recover discarded EOS/token IDs."}


def main():
    OUT.mkdir(exist_ok=True)
    reports, manifest = [], {}
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file():
            continue
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            for key, record in walks(data):
                result = audit_record(record)
                if result:
                    reports.append({"file": str(path.relative_to(ROOT)), "record": key, **result})
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            reports.append({"file": str(path.relative_to(ROOT)), "error": repr(exc)})
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT / "code").rglob("*.py")}
    (OUT / "original_manifest.json").write_text(json.dumps({**manifest, **sources}, indent=2))
    (OUT / "saved_results_recalculated.json").write_text(json.dumps(reports, indent=2, allow_nan=False))
    main = next(r for r in reports if r["file"].endswith("quality_assessment_v8.json"))
    print(json.dumps(main, indent=2))


if __name__ == "__main__":
    main()
