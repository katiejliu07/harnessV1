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
  technical - the task brings a check (expected value, function cases or a
              check command): cheap first, escalate on a failed check.
  important - flagged important, no check: strongest model straight away,
              since nothing could catch a cheap model's mistake.
  general   - everything else: one call.
The swarm can force a preset with "kind".

Every run returns usage per attempt (for the allocator) and a compact handoff
for passing results between agents instead of full transcripts.

Task JSON: {"prompt": str, "context": str?, "kind": str?, "important": bool?,
            "toolset": "none"|"read"|"code"|"full"?, "tools": bool?,
            "expected": str?, "function": str?, "cases": [...]?,
            "check_cmd": str?, "workdir": str?, "token_budget": int?}
"""
import argparse
import json
from pathlib import Path

from . import agents
from .compact import compact_checked
from .verify import check,has_check

CONFIG_PATH=Path(__file__).with_name("config.json")


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def route(task):
    if task.get("kind") in ("technical","important","general"):
        return task["kind"]
    if has_check(task):
        return "technical"
    return "important" if task.get("important") else "general"


def toolset_of(task):
    """The smallest toolset the task needs: declared, or inferred from whether it
    has a workdir or check command to act on."""
    if task.get("toolset"):
        return task["toolset"]
    if task.get("tools") is not None:
        return "code" if task["tools"] else "none"
    return "code" if task.get("workdir") or task.get("check_cmd") else "none"


def build_prompt(task):
    context=task.get("context")
    report=None
    if context:
        context,report=compact_checked(context)
    prompt=task["prompt"] if not context else f"{task['prompt']}\n\n<context>\n{context}\n</context>"
    return prompt,report


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


def run_task(task,config=None,runner=agents.run):
    config=config or load_config()
    path=route(task)
    prompt,compaction=build_prompt(task)
    profile=config["profiles"][path]
    steps,toolset=steps_for(path,task,profile,config)
    tools=toolset!="none"
    hygiene=config["hygiene_instruction"].format(compact=Path(__file__).with_name("compact.py"))
    budget=task.get("token_budget")
    attempts=[]
    result=None
    budget_exhausted=False
    for step in steps:
        if budget is not None and attempts and spent(attempts)>=budget:
            budget_exhausted=True          # the allocator's allotment is used up: report, don't overspend
            break
        # Hygiene is per agent: pilot_v2 measured -36% tokens per correct for Codex
        # and +4% for Claude (6/6 correct either way), so only listed agents get it.
        system=profile.get("system_append")
        if tools and profile.get("hygiene") and step["agent"] in config.get("hygiene_agents",[]):
            system="\n".join(x for x in [system,hygiene] if x)
        result=runner(step,prompt,{**config,"system_append":system},task.get("workdir"))
        if path=="technical":
            verdict=check(task,result["text"],task.get("workdir")) if not result["error"] else {"passed":False}
        else:
            verdict={"passed":None}
        attempts.append({**step,"usage":result["usage"],"seconds":result["seconds"],"error":result["error"],"verified":verdict["passed"]})
        if path!="technical" or verdict["passed"]:
            break
    verified=attempts[-1]["verified"] if attempts else None
    text=result["text"] if result else ""
    return {"route":path,"toolset":toolset,"text":text,"verified":verified,"attempts":attempts,
            "usage_total":totals(attempts),"tokens_spent":spent(attempts),"token_budget":budget,
            "budget_exhausted":budget_exhausted,"compaction":compaction,"handoff":handoff(path,text,verified)}


def main():
    ap=argparse.ArgumentParser(description="Run one swarm task through the harness")
    ap.add_argument("task_file",type=Path)
    ap.add_argument("--config",type=Path,default=CONFIG_PATH)
    ap.add_argument("--report",type=Path,help="append the JSON report (usage per attempt) here")
    args=ap.parse_args()
    report=run_task(json.loads(args.task_file.read_text()),load_config(args.config))
    print(report["text"])
    if args.report:
        with args.report.open("a") as f:
            f.write(json.dumps({k:v for k,v in report.items() if k!="text"})+"\n")


if __name__=="__main__":
    main()
