"""Entry point each swarm agent calls to spend its token allotment well.

How many tokens each parallel agent gets is decided by a separate allocator.
This harness makes one agent's task cost as little as possible inside that
allotment, without losing accuracy, and reports exactly what it spent.

One dispatcher sets independent settings per task from cheap signals (no model
call), starting small and escalating only when an objective check fails:

  model and effort - a ladder (e.g. Sonnet low -> Sonnet high -> Opus high)
  tools            - only the declared toolset first (none / read / code), then full
  context          - compacted (repetition out, never facts)
  instructions     - per agent, by evidence (hygiene for Codex tool runs)
  budget           - escalation stops once the task's token_budget is spent

Presets (config "profiles") bundle defaults for three situations:
  technical - the task brings a check (see verify.py): cheap first, escalate
              on a failed check.
  important - flagged important, no check: strongest model straight away,
              since nothing could catch a cheap model's mistake.
  general   - everything else: one call.
The swarm can force a preset with "kind".

Every run returns usage per attempt (for the allocator) and a compact handoff
for passing results between agents instead of full transcripts.

Task JSON: {"prompt": str, "context": str?, "context_files": [path]?,
            "kind": str?, "important": bool?,
            "toolset": "none"|"read"|"code"|"full"?, "tools": bool?,
            checks (verify.py): "expected", "function"+"cases", "check_cmd",
            "regex", "contains", "output_schema", "checks": [...],
            "workdir": str?, "token_budget": int?,
            "timeout": seconds per agent call?, "check_timeout": seconds?,
            "allowed_tools": ["Bash(pytest*)", ...]? (Claude permission rules),
            "max_budget_usd": float? (hard stop: each Claude call gets what is left)}
"""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from . import agents
from .compact import compact_checked
from .verify import check,has_check,has_strong_check,specs
from .workspace import Snapshot

CONFIG_PATH=Path(__file__).with_name("config.json")


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def route(task):
    if task.get("kind") in ("technical","important","general"):
        return task["kind"]
    if has_strong_check(task):
        return "technical"
    return "important" if task.get("important") else "general"


def toolset_of(task):
    """The smallest toolset the task needs: declared, or inferred from whether it
    has a workdir or check command to act on."""
    if task.get("toolset"):
        return task["toolset"]
    if task.get("tools") is not None:
        return "code" if task["tools"] else "none"
    has_cmd=any(s.get("type")=="cmd" for s in specs(task))
    return "code" if task.get("workdir") or has_cmd else "none"


def _read_context_file(path,workdir):
    p=Path(path)
    if not p.is_absolute() and workdir:
        p=Path(workdir)/p
    return p.read_text(errors="replace")


def build_prompt(task):
    """Prompt plus context. Free-form context (logs, notes) is compacted; files
    in context_files are included verbatim, since compaction treats unfenced
    text as prose and source files must reach the model byte for byte."""
    context=task.get("context")
    report=None
    if context:
        context,report=compact_checked(context)
    parts=[task["prompt"]]
    if context:
        parts.append(f"<context>\n{context}\n</context>")
    for path in task.get("context_files") or []:
        parts.append(f'<file path="{path}">\n{_read_context_file(path,task.get("workdir"))}\n</file>')
    return "\n\n".join(parts),report


def totals(attempts):
    out={}
    for a in attempts:
        for k,v in a["usage"].items():
            out[k]=out.get(k,0)+v
    out["input_total"]=agents.input_total(out)
    return out


def spent(attempts):
    t=totals(attempts)
    return t["input_total"]+t.get("output_tokens",0)


def handoff(path,text,verified):
    """What the next agent gets: the compacted final answer and its status,
    never the transcript that produced it."""
    body,_=compact_checked(text or "")
    return {"route":path,"verified":verified,"result":body}


def steps_for(path,task,profile,config):
    toolset=toolset_of(task)
    checked=path=="technical" and has_check(task)
    ladder=[dict(s) for s in (profile["ladder"] if checked else profile["ladder"][:1])]
    if not ladder:
        return [],toolset
    first=ladder[0]
    if not checked and task.get("toolset") in config.get("toolsets",{}) and task["toolset"]!="full" and first["agent"]=="claude":
        # Unchecked, but the caller *declared* the tools it needs (a chatbot turn
        # with toolset "none"): honor it. Claude Code's own system prompt stays;
        # only tools the caller said it doesn't need are dropped (no tools: about
        # 21k -> 3k input tokens). An inferred toolset is never applied here,
        # since nothing would catch a wrong guess.
        ladder[0]={**first,"toolset":config["toolsets"][task["toolset"]]}
    if not checked and has_check(task):
        # Format checks only (general/important): keep this preset's full first
        # setup, and on a failed format check climb to the technical ladder's
        # stronger rungs. Never a minimal rung: the check can't catch a wrong answer.
        tech=config["profiles"].get("technical",{}).get("ladder",[])
        key=lambda r:(r.get("agent"),r.get("model"),r.get("effort"))
        keys=[key(r) for r in tech]
        if key(ladder[0]) in keys:
            ladder+=[dict(r) for r in tech[keys.index(key(ladder[0]))+1:]]
        return ladder,toolset
    # Minimal first rung in front of a checked ladder. A miss is caught by the
    # check and falls through to the unchanged (full-tools) ladder, so accuracy
    # is the ladder's; the saving stays whenever the minimal rung passes.
    #  - no tools: lean mode (pilot_v2: ~21k -> ~0.6k input, 1 of 12 missed and caught)
    #  - some tools: only that toolset (fixed overhead 20,933 -> 6,031 for Bash,Read,Edit,Write)
    if checked and first["agent"]=="claude":
        if toolset=="none" and profile.get("lean_first_rung"):
            ladder.insert(0,{**first,"lean":True,"system_prompt":profile.get("lean_system_prompt")})
        elif toolset in config.get("toolsets",{}) and toolset!="full" and profile.get("toolset_first_rung"):
            ladder.insert(0,{**first,"toolset":config["toolsets"][toolset]})
    return ladder,toolset


def transient(result,config):
    """A provider/network hiccup, not a wrong answer: worth the same rung again.
    The harness's own timeout is never treated as transient (it already cost
    the full allowance)."""
    if not result.get("error") or result.get("timed_out"):
        return False
    text=" ".join(str(result.get(k) or "") for k in ("error_text","stderr_tail"))
    pats=config.get("retry",{}).get("patterns",[])
    return any(re.search(p,text,re.I) for p in pats)


def run_task(task,config=None,runner=None,sleep=None,on_text=None):
    config=config or load_config()
    runner=runner or agents.run              # looked up per call, so it can be swapped (tests, wrappers)
    sleep=sleep or time.sleep
    path=route(task)
    prompt,compaction=build_prompt(task)
    profile=config["profiles"][path]
    steps,toolset=steps_for(path,task,profile,config)
    tools=toolset!="none"
    hygiene=config["hygiene_instruction"].format(compact=Path(__file__).with_name("compact.py"),
                                                 python="python" if os.name=="nt" else "python3")
    budget=task.get("token_budget")
    retry=config.get("retry",{})
    max_retries=int(retry.get("transient_retries",0))
    backoff=retry.get("backoff_seconds") or [0]
    # What the runner needs to know about the task, beyond the step.
    run_config={**config,"task_toolset":toolset,
                **({"agent_timeout":task["timeout"]} if task.get("timeout") else {}),
                **({"output_schema":task["output_schema"]} if task.get("output_schema") else {}),
                **({"allowed_tools":list(task["allowed_tools"])} if task.get("allowed_tools") else {})}
    # Streaming only where the first answer is the answer: a checked rung's text
    # can still fail its check and be replaced by the next rung's.
    if on_text is not None and not has_check(task):
        run_config["on_text"]=on_text
    usd_budget=task.get("max_budget_usd")
    attempts=[]
    result=None
    budget_exhausted=False
    exception=None
    workdir=task.get("workdir")

    def usd_left():
        return None if usd_budget is None else usd_budget-sum(a.get("cost_usd") or 0 for a in attempts)

    def out_of_budget():
        if budget is not None and attempts and spent(attempts)>=budget:
            return True
        left=usd_left()
        return left is not None and left<=0
    # Each rung starts from the same files (see workspace.py). Only checked tasks
    # escalate, so only they pay for the snapshot.
    isolation=None
    snap=None
    checked=has_check(task)
    if workdir and checked and (len(steps)>1 or max_retries) and config.get("isolate_rungs",True):
        snap=Snapshot(workdir,int(config.get("isolate_rungs_max_mb",100))*1024*1024,
                      background=config.get("snapshot_background",True))
        isolation={"snapshot_seconds":snap.seconds,"skipped":snap.skipped,"restores":0,"restore_seconds":0.}

    def restore():
        nonlocal exception,snap
        if snap is None or snap.dir is None:
            return True
        try:
            took=snap.restore()
            if took is None:
                # The background copy couldn't be trusted (or the workdir is too
                # big): carry on unisolated, as before isolation existed, and say so.
                isolation["skipped"]=snap.skipped
                snap.close()
                snap=None
                return True
            isolation["restore_seconds"]+=took
        except OSError as exc:
            # A half-restored workdir is worse than none: stop and report.
            exception=f"restore: {exc!r}"[:500]
            return False
        isolation["restores"]+=1
        return True

    try:
        for step in steps:
            if out_of_budget():
                budget_exhausted=True          # the allocator's allotment is used up: report, don't overspend
                break
            if attempts and not restore():
                break
            tries=0
            while True:
                call_config=run_config if usd_budget is None else {**run_config,"max_budget_usd":usd_left()}
                result,verdict,exception=_attempt(step,prompt,call_config,profile,tools,hygiene,checked,task,runner)
                attempts.append({**step,"usage":result["usage"],"seconds":result["seconds"],"error":result["error"],"verified":verdict["passed"],
                                 **({"cost_usd":result["cost_usd"]} if result.get("cost_usd") is not None else {}),
                                 **({"timed_out":True} if result.get("timed_out") else {}),
                                 **({"check_timed_out":True} if verdict.get("timed_out") else {}),
                                 **({"retry":tries} if tries else {}),
                                 **({"exception":exception} if exception else {})})
                if exception or tries>=max_retries or not transient(result,config):
                    break
                if out_of_budget():
                    break
                attempts[-1]["transient"]=True
                sleep(backoff[min(tries,len(backoff)-1)])
                tries+=1
                if not restore():
                    break
            if exception or not checked or verdict["passed"]:
                break
    finally:
        if snap is not None:
            if isolation is not None and snap._thread is not None and not snap._thread.is_alive():
                isolation["build_seconds"]=snap.build_seconds
                isolation["method"]=snap.method
                isolation["skipped"]=isolation["skipped"] or snap.skipped
            snap.close()
    verified=attempts[-1]["verified"] if attempts else None
    text=result["text"] if result else ""
    report={"route":path,"toolset":toolset,"text":text,"verified":verified,"attempts":attempts,
            "usage_total":totals(attempts),"tokens_spent":spent(attempts),"token_budget":budget,
            "budget_exhausted":budget_exhausted,"compaction":compaction,"handoff":handoff(path,text,verified)}
    if not budget_exhausted and attempts and out_of_budget() and checked and not verified:
        report["budget_exhausted"]=True        # the last rung itself ran the allotment out
    if usd_budget is not None:
        report["cost_usd"]=sum(a.get("cost_usd") or 0 for a in attempts)
        report["max_budget_usd"]=usd_budget
    if isolation:
        report["rung_isolation"]=isolation
    if exception:
        report["exception"]=exception
    return report


def _attempt(step,prompt,config,profile,tools,hygiene,checked,task,runner):
    """One rung: (result, verdict, exception). An unexpected runner or check
    failure becomes a failed attempt, so earlier attempts' usage still reaches
    the report instead of being lost with a crash."""
    try:
        # Hygiene is per agent: pilot_v2 measured -36% tokens per correct for Codex
        # and +4% for Claude (6/6 correct either way), so only listed agents get it.
        system=profile.get("system_append")
        if tools and profile.get("hygiene") and step["agent"] in config.get("hygiene_agents",[]):
            system="\n".join(x for x in [system,hygiene] if x)
        result=runner(step,prompt,{**config,"system_append":system},task.get("workdir"))
    except Exception as exc:                   # noqa: BLE001 - reported, not swallowed
        return {"text":"","error":True,"usage":{},"seconds":0.},{"passed":False if checked else None},f"runner: {exc!r}"[:500]
    if not checked:
        return result,{"passed":None},None
    if result["error"]:
        return result,{"passed":False},None
    try:
        return result,check(task,result["text"],task.get("workdir")),None
    except Exception as exc:                   # noqa: BLE001
        return result,{"passed":False},f"check: {exc!r}"[:500]


def run_tasks(tasks,config=None,jobs=4,runner=None,on_done=None):
    """Run many tasks in parallel; reports come back in input order.

    Agent calls are subprocesses, so threads give real parallelism. Tasks that
    share a workdir run one after another in the same worker (they would see
    each other's edits, and rung isolation would undo the other task's files)."""
    config=config or load_config()
    groups={}
    for i,t in enumerate(tasks):
        key=os.path.realpath(t["workdir"]) if t.get("workdir") else f"#{i}"
        groups.setdefault(key,[]).append(i)
    reports=[None]*len(tasks)
    lock=threading.Lock()

    def work(indices):
        for i in indices:
            try:
                reports[i]=run_task(tasks[i],config,runner=runner)
            except Exception as exc:          # noqa: BLE001 - one bad task must not sink the batch
                reports[i]={"route":None,"text":"","verified":None,"attempts":[],"usage_total":{},"tokens_spent":0,
                            "exception":f"task: {exc!r}"[:500]}
            if on_done:
                with lock:
                    on_done(i,reports[i])

    from concurrent.futures import ThreadPoolExecutor        # ~7 ms import: only batches pay it
    with ThreadPoolExecutor(max_workers=max(1,jobs)) as pool:
        list(pool.map(work,groups.values()))
    return reports


def load_tasks(path):
    """A task object, a JSON list of tasks, or JSON Lines."""
    raw=Path(path).read_text() if str(path)!="-" else sys.stdin.read()
    try:
        data=json.loads(raw)
        return data if isinstance(data,list) else [data]
    except ValueError:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]


def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser(description="Run tasks through the harness")
    ap.add_argument("task_file",nargs="?",type=str,help="task JSON, JSON list or JSONL ('-' for stdin)")
    ap.add_argument("--prompt",help="run a single task given inline instead of a file")
    ap.add_argument("--prompt-file",type=Path)
    ap.add_argument("--context-file",action="append",default=[],help="file included verbatim (repeatable)")
    ap.add_argument("--workdir")
    ap.add_argument("--check-cmd")
    ap.add_argument("--expected")
    ap.add_argument("--regex")
    ap.add_argument("--schema",type=Path,help="JSON Schema file: structured output plus a validity check")
    ap.add_argument("--toolset",choices=["none","read","code","full"])
    ap.add_argument("--allow",action="append",default=[],help='Claude permission rule, e.g. "Bash(pytest*)" (repeatable)')
    ap.add_argument("--kind",choices=["technical","important","general"])
    ap.add_argument("--important",action="store_true")
    ap.add_argument("--budget",type=int,help="token budget (reported, checked between attempts)")
    ap.add_argument("--max-usd",type=float,help="hard spend limit in list-price USD, enforced inside Claude calls")
    ap.add_argument("--timeout",type=int,help="seconds per agent call")
    ap.add_argument("--jobs",type=int,default=4,help="parallel tasks for a batch")
    ap.add_argument("--config",type=Path,default=CONFIG_PATH)
    ap.add_argument("--report",type=Path,help="append the JSON report (usage per attempt) here")
    ap.add_argument("--json",action="store_true",help="print the full report(s) as JSON lines")
    args=ap.parse_args(argv)
    if args.prompt or args.prompt_file:
        task={"prompt":args.prompt or args.prompt_file.read_text()}
        for k,v in (("workdir",args.workdir),("check_cmd",args.check_cmd),("expected",args.expected),("regex",args.regex),
                    ("toolset",args.toolset),("kind",args.kind),("token_budget",args.budget),("timeout",args.timeout),("max_budget_usd",args.max_usd)):
            if v is not None:
                task[k]=v
        if args.important:
            task["important"]=True
        if args.allow:
            task["allowed_tools"]=args.allow
        if args.context_file:
            task["context_files"]=args.context_file
        if args.schema:
            task["output_schema"]=json.loads(args.schema.read_text())
        tasks=[task]
    elif args.task_file:
        tasks=load_tasks(args.task_file)
    else:
        ap.error("give a task file or --prompt")
    config=load_config(args.config)
    reports=run_tasks(tasks,config,jobs=args.jobs) if len(tasks)>1 else [run_task(tasks[0],config)]
    for r in reports:
        if args.json or len(reports)>1:
            print(json.dumps(r))
        else:
            print(r["text"])
    if args.report:
        with args.report.open("a") as f:
            for r in reports:
                f.write(json.dumps({k:v for k,v in r.items() if k!="text"})+"\n")
    # 0 all verified or unchecked, 1 a check failed, 2 an exception
    if any(r.get("exception") for r in reports):
        return 2
    return 1 if any(r.get("verified") is False for r in reports) else 0


if __name__=="__main__":
    sys.exit(main())
