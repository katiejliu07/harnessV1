"""Harness overhead with an instant fake model (no tokens spent).

    python -m agent_harness.bench

Times what the harness itself adds per task: compaction, routing, checks, and
rung isolation (snapshot before the first rung, restore before each
escalation) on projects of 10, 1k and 5k files. Real model calls take seconds;
these numbers should stay in milliseconds.
"""
import os
import shutil
import statistics
import sys
import tempfile
import time

from .run import load_config,run_task
from .workspace import Snapshot,wait_for_cleanup

PY=f'"{sys.executable}"'


def _runner(answer):
    def run(step,prompt,config,workdir=None):
        return {"text":answer,"error":False,"usage":{"input_tokens":1,"output_tokens":1},"seconds":0.}
    return run


def _project(n,size):
    d=tempfile.mkdtemp(prefix="bench_wd_")
    blob=os.urandom(size)
    for i in range(n):
        sub=os.path.join(d,f"pkg{i%50}")
        os.makedirs(sub,exist_ok=True)
        with open(os.path.join(sub,f"f{i}.py"),"wb") as f:
            f.write(blob)
    return d


def _median_ms(fn,reps):
    xs=[]
    for _ in range(reps):
        s=time.perf_counter(); fn(); xs.append((time.perf_counter()-s)*1000)
    return statistics.median(xs)


def main():
    config=load_config()
    true=f'{PY} -c "raise SystemExit(0)"'
    false=f'{PY} -c "raise SystemExit(1)"'
    log="\n".join(f"2026-10-29 10:{i//60%60:02d}:{i%60:02d} INFO heartbeat ok {i%7}" for i in range(2000))
    rows=[("general task",_median_ms(lambda:run_task({"prompt":"Explain"},config,runner=_runner("ok")),9)),
          ("general task, 2k-line log",_median_ms(lambda:run_task({"prompt":"Why?","context":log},config,runner=_runner("ok")),9)),
          ("exact-answer task",_median_ms(lambda:run_task({"prompt":"3+4?","expected":"7"},config,runner=_runner("Final answer: 7")),9))]
    import subprocess
    check_ms=_median_ms(lambda:subprocess.run(true,shell=True),9)
    rows.append(("(one check command, for reference)",check_ms))
    for label,(n,size) in {"10 files":(10,2000),"1k files / 10 MB":(1000,10000),"5k files / 50 MB":(5000,10000)}.items():
        d=_project(n,size)
        try:
            reps=5 if n<5000 else 3
            rows.append((f"coding task passes, {label}",_median_ms(lambda:run_task({"prompt":"fix","check_cmd":true,"workdir":d},config,runner=_runner("done")),reps)))
            rows.append((f"coding task, 4 rungs fail, {label}",_median_ms(lambda:run_task({"prompt":"fix","check_cmd":false,"workdir":d},config,runner=_runner("done")),reps)))
            snap=Snapshot(d,10**10)
            rows.append((f"  snapshot: caller waits (scan), {label}",snap.seconds*1000))
            snap.ready()
            rows.append((f"  snapshot: background copy, {label} [{snap.method}]",(snap.build_seconds or 0)*1000))
            with open(os.path.join(d,"pkg0","f0.py"),"wb") as f:
                f.write(b"edited")
            rows.append((f"  restore 1 changed file, {label}",snap.restore()*1000))
            snap.close(wait=True)
        finally:
            shutil.rmtree(d,ignore_errors=True)
    wait_for_cleanup()
    from . import workspace
    print(f"platform: {sys.platform}, python {sys.version.split()[0]}, "
          f"windows change-time fingerprints: {bool(workspace._win) if os.name=='nt' else 'n/a'}")
    for name,ms in rows:
        print(f"{name:46s} {ms:9.2f} ms")


if __name__=="__main__":
    main()
