import json
import hashlib
from pathlib import Path
import pytest
from independent_audit import report


def test_raw_scores_preserved_and_repeats_not_counted_as_new_tasks(tmp_path,monkeypatch):
    monkeypatch.setattr(report,"ROOT",tmp_path)
    directory=tmp_path/"run"
    directory.mkdir()
    rows=[]
    for repeat in [0,1]:
        for condition,cost,raw in [("B_reasoning",2.,True),("E_v8_prefix",1.,False)]:
            rows.append({"suite":"original","task_index":0,"condition":condition,"repeat":repeat,
              "donor_variant":"same","status":"ok","task":{"prompt":"sqrt49","answer":"7"},
              "correct":raw,"truncated":False,"token_ids":[1,2],"extracted_answer":"7" if raw else "49",
              "generated_tokens_including_eos":2,"total_seconds":cost,"model_seconds_including_donor":cost})
    (directory/"benchmark.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    diagnostics=[{**{k:r[k] for k in ["suite","task_index","condition","repeat","donor_variant"]},
                  "scalar_evaluation_v2":{"passed":True,"answer":"7"}} for r in rows]
    (directory/"quality_diagnostics.json").write_text(json.dumps(diagnostics))
    manifest={"input_sha256":hashlib.sha256((directory/"benchmark.jsonl").read_bytes()).hexdigest(),
              "diagnostics_sha256":hashlib.sha256((directory/"quality_diagnostics.json").read_bytes()).hexdigest(),"source_sha256":{}}
    (directory/"grading_manifest.json").write_text(json.dumps(manifest))
    original=(directory/"benchmark.jsonl").read_bytes()
    monkeypatch.chdir(tmp_path)
    result=report.summarize(Path("run"))
    entry=next(r for r in result["summaries"] if r["condition"]=="E_v8_prefix")
    assert entry["n_unique_tasks"]==1
    assert entry["correct"]==1 and entry["raw_primary_correct"]==0
    assert entry["comparison_to_reasoning"]["quality_gate"]["total_cost_ratio"]==.5
    assert result["full_noop_and_alpha0_parity"]["all_identical"] is None
    assert (directory/"benchmark.jsonl").read_bytes()==original


def test_capped_joint_failure_must_not_hide_functional_regression():
    baseline={"correct":False,"task_checks_passed":True}
    patch={"correct":False,"task_checks_passed":False}
    gate=report.dimensional_quality_gate([baseline],[patch],2.,1.)
    assert gate["observed_correctness_regressions"]==0
    assert gate["observed_task_check_regressions"]==1
    assert not gate["eligible_on_observed_sample"]


def test_manual_review_is_bound_to_data_and_preserves_automatic_verdict():
    row={"suite":"coding","task_index":0,"condition":"B_reasoning","repeat":0,"donor_variant":"same",
         "text":"identical function repeated","correct":False,"task_checks_passed":False,"truncated":False}
    raw=b"bound input"
    review={"input_sha256":hashlib.sha256(raw).hexdigest(),"records":[{
        **{k:row[k] for k in ["suite","task_index","condition","repeat","donor_variant"]},
        "text_sha256":hashlib.sha256(row["text"].encode()).hexdigest(),"task_checks_passed":True}]}
    with pytest.raises(ValueError,match="different benchmark"):
        report.apply_adjudications([row.copy()],review,b"changed input")
    report.apply_adjudications([row],review,raw)
    assert row["correct"] and not row["automatic_task_checks_passed"]
