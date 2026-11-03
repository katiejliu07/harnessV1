"""Serial followups for the long-running benchmark; no concurrent GPU trials.

Measurements completing is not scientific adjudication completing. This writes
machine-readable stage status and never modifies an original experiment file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"independent_audit/results"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--main-pid",type=int,required=True)
    args=ap.parse_args()
    main_dir=BASE/"mps_torch271_full"
    status_path=BASE/"postprocess_status.json"
    logs=BASE/"followup_logs"
    logs.mkdir(exist_ok=True)
    stages=[]
    def status(stage,**extra):
        data={"stage":stage,"orchestrator_pid":os.getpid(),"main_pid":args.main_pid,
              "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"completed_stages":stages,**extra}
        status_path.write_text(json.dumps(data,indent=2))
    try:
        while not (main_dir/"completed.json").exists():
            alive=psutil.pid_exists(args.main_pid) and psutil.Process(args.main_pid).status()!=psutil.STATUS_ZOMBIE
            path=main_dir/"benchmark.jsonl"
            rows=len(path.read_text().splitlines()) if path.exists() else 0
            status("waiting_for_main",rows=rows)
            if not alive:
                raise RuntimeError("Main benchmark exited without a completion marker")
            time.sleep(15)
        main_rows=[json.loads(s) for s in (main_dir/"benchmark.jsonl").read_text().splitlines()]
        if len(main_rows)!=790 or any(r["status"]=="error" or r["status"]=="skipped" and r["condition"]!="C_length_matched" for r in main_rows):
            raise RuntimeError("Main benchmark coverage/error check failed")
        if len({tuple(r[k] for k in ["suite","task_index","condition","repeat","donor_variant"]) for r in main_rows})!=790:
            raise RuntimeError("Duplicate main benchmark keys")
        plan=json.loads((main_dir/"plan.json").read_text())
        if not all(hashlib.sha256((ROOT/"independent_audit"/k).read_bytes()).hexdigest()==sha for k,sha in plan["source_sha256"].items()):
            raise RuntimeError("Frozen inference source hashes changed")

        def run(name,args):
            status(name)
            print(f"Starting {name}",flush=True)
            with (logs/(name+".txt")).open("w") as log:
                child=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                status(name,child_pid=child.pid)
                code=child.wait()
            if code:
                raise RuntimeError(f"{name} failed ({code}); inspect {logs/(name+'.txt')}")
            stages.append(name)
            status(name+"_complete")

        run("unit_tests",["-m","pytest","independent_audit/tests","-q"])
        run("main_quality_review",["-m","independent_audit.quality_review",str(main_dir)])
        run("main_report",["-m","independent_audit.report",str(main_dir)])
        summary=json.loads((main_dir/"summary.json").read_text())
        if summary["full_noop_and_alpha0_parity"]["all_identical"] is not True:
            raise RuntimeError("Full-sequence no-op/alpha-zero parity failed; investigate before causal interpretation")
        run("donor_early_stop",["-u","-m","independent_audit.optimization","--device","mps","--dtype","float16"])
        run("prefix_partitions",["-u","-m","independent_audit.prefix_partitions"])
        timing=BASE/"mps_torch271_timing_repeats"
        run("timing_repeats",["-u","-m","independent_audit.benchmark","--device","mps","--dtype","float16",
             "--out",str(timing.relative_to(ROOT)),"--limit","5","--donor-probes","0","--repeats","3",
             "--conditions","A_normal","B_reasoning","C_concise","E_v8_prefix"])
        run("timing_quality_review",["-m","independent_audit.quality_review",str(timing)])
        run("timing_report",["-m","independent_audit.report",str(timing)])
        optimization=json.loads((BASE/"donor_early_stop.json").read_text())
        if not optimization["all_bitwise_equal"]:
            raise RuntimeError("Donor optimization changed captured activations")
        if json.loads((BASE/"prefix_partition_followup/completed.json").read_text())["rows"]!=111:
            raise RuntimeError("Prefix partition coverage incomplete")
        status("measurements_complete",note="Manual review and final scientific report still required")
        print("Serial measurements complete; adjudication/reporting remains",flush=True)
    except BaseException as exc:
        status("failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
