"""Objective checks for technical tasks. A task is technical when it brings one.

Escalating to a stronger model is only as safe as the check: weak tests can
accept a wrong cheap answer. Prefer real test suites; for functions, the
EvalPlus-style differential cases in lean_harness show how much hand-written
cases miss.
"""
import subprocess

from independent_audit.code_quality import evaluate_code
from independent_audit.scoring_v2 import correct


def has_check(task):
    return any(task.get(k) is not None for k in ("expected","check_cmd")) or bool(task.get("function") and task.get("cases"))


def check(task,text,workdir=None,timeout=600):
    """Returns {"passed": bool, "how": ...}. check_cmd runs in the task's workdir
    and passes on exit code 0; it is the swarm's own command, not model output."""
    if task.get("expected") is not None:
        return {"passed":correct(text,str(task["expected"])),"how":"expected"}
    if task.get("function") and task.get("cases"):
        verdict=evaluate_code(text,{"function":task["function"],"cases":task["cases"]})
        return {"passed":bool(verdict["passed"]),"how":"cases","detail":verdict.get("functional")}
    if task.get("check_cmd"):
        proc=subprocess.run(task["check_cmd"],shell=True,capture_output=True,text=True,timeout=timeout,cwd=workdir or task.get("workdir"))
        return {"passed":proc.returncode==0,"how":"check_cmd","tail":(proc.stdout+proc.stderr)[-800:]}
    return {"passed":None,"how":"none"}
