"""On GitHub Actions, report each failing test as an annotation. Annotations are
readable through the public API without signing in, unlike the job logs."""
import os
import sys


def pytest_runtest_logreport(report):
    if not os.environ.get("GITHUB_ACTIONS") or not report.failed:
        return
    text=str(report.longrepr)[-1500:]
    for a,b in (("%","%25"),("\r",""),("\n","%0A")):
        text=text.replace(a,b)
    sys.__stdout__.write(f"::error title={report.nodeid}::{text}\n")
    sys.__stdout__.flush()
