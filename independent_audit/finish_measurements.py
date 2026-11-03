"""Serialize the additional checks after postprocess.py; never race GPU runs.

The main and first followup plans are frozen. These diagnostics preserve their
data, and have separate manifests/results. A failed prerequisite stops this
queue rather than silently omitting the parity or common-budget experiments.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"independent_audit/results"


def main():
    status_path=BASE/"additional_measurements_status.json"
    completed=[]
    def status(stage,**extra):
        status_path.write_text(json.dumps({"stage":stage,"pid":os.getpid(),
            "completed_stages":completed,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**extra},indent=2))
    try:
        while True:
            previous=json.loads((BASE/"postprocess_status.json").read_text())
            if previous["stage"]=="failed":
                raise RuntimeError(f"Earlier pipeline failed: {previous.get('error')}")
            if previous["stage"]=="measurements_complete":
                break
            status("waiting_for_serial_pipeline",previous_stage=previous["stage"])
            time.sleep(15)
        logs=BASE/"followup_logs"
        logs.mkdir(exist_ok=True)
        def run(name,args):
            status(name)
            print(f"Starting {name}",flush=True)
            with (logs/(name+".txt")).open("w") as log:
                child=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                status(name,child_pid=child.pid)
                code=child.wait()
            if code:
                raise RuntimeError(f"{name} failed ({code}); inspect its log")
            completed.append(name)
            status(name+"_complete")
        run("code_cache_parity",["-u","-m","independent_audit.code_cache_parity"])
        parity=json.loads((BASE/"code_cache_parity.json").read_text())
        if not all(parity[k] for k in ["all_token_identical","all_cached_match_main","all_native_baselines_match_matrix"]):
            raise RuntimeError("Actual coding-path parity differs; adjudicate before proceeding")
        run("equal_length_donors",["-u","-m","independent_audit.equal_length_donors"])
        if json.loads((BASE/"equal_length_donor_followup/summary.json").read_text())["comparisons"]!=24:
            raise RuntimeError("Equal-length donor coverage incomplete")
        run("budget_1024",["-u","-m","independent_audit.budget_followup"])
        budget=BASE/"budget_1024_followup"
        if json.loads((budget/"completed.json").read_text())["rows"]:
            run("budget_quality_review",["-m","independent_audit.quality_review",str(budget)])
            run("budget_report",["-m","independent_audit.report",str(budget)])
        status("measurements_complete",note="Final interpretation and integrity receipt still required")
        print("Additional serial measurements complete",flush=True)
    except BaseException as exc:
        status("failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
