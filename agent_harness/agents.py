"""Run one request through Claude Code or Codex headless and read its usage.

Commands come from config (see config.json) so model names, effort flags and
tool permissions can change without code changes. Usage parsing searches the
output for token fields wherever they appear, rather than assuming one schema.
Confirmed against real runs 2026-10-29: Claude Code --output-format json puts
the answer in "result" and tokens in "usage"; Codex exec --json reports tokens
in a "turn.completed" event and the answer via -o.
"""
import json
import os
import subprocess
import tempfile
import time

USAGE_KEYS=("input_tokens","output_tokens","cache_read_input_tokens","cache_creation_input_tokens","cached_input_tokens","cache_write_input_tokens","reasoning_output_tokens")
# Measured 2026-10-29 (Claude Code 2.1.314, one trivial question): default run
# 20,934 input tokens; no tools/MCP/skills/settings 2,814; plus a short
# replacement system prompt 579. The overhead is the built-in tool definitions
# and default system prompt. Lean mode is only for tasks that need no tools,
# and must pass the accuracy gate before it is enabled for a path.
CLAUDE_LEAN=["--tools","","--strict-mcp-config","--disable-slash-commands","--setting-sources",""]


def input_total(usage):
    """Every input token the model read: fresh + cache writes + cache reads."""
    return sum(usage.get(k,0) for k in ("input_tokens","cache_creation_input_tokens","cache_read_input_tokens","cache_write_input_tokens"))


def _usage_dicts(obj):
    if isinstance(obj,dict):
        if any(k in obj for k in ("input_tokens","output_tokens")):
            yield obj
        for v in obj.values():
            yield from _usage_dicts(v)
    elif isinstance(obj,list):
        for v in obj:
            yield from _usage_dicts(v)


def parse_claude(stdout):
    data=json.loads(stdout)
    usage=next(_usage_dicts(data.get("usage",data)),{})
    # total_cost_usd is a list-price estimate; on a subscription it is notional
    # but still a consistent weight across models for cost-per-correct.
    return {"text":data.get("result",""),"error":bool(data.get("is_error")),
            "usage":{k:usage.get(k,0) for k in USAGE_KEYS if k in usage},"cost_usd":data.get("total_cost_usd"),
            "raw":{k:data.get(k) for k in ("num_turns","modelUsage","total_cost_usd","usage")}}


def parse_codex(stdout,last_message):
    events=[json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    found=[u for e in events for u in _usage_dicts(e)]
    usage=found[-1] if found else {}      # the latest report; confirm cumulative vs per-turn in the pilot
    return {"text":last_message,"error":any(e.get("type") in ("error","turn.failed") for e in events),
            "usage":{k:usage.get(k,0) for k in USAGE_KEYS if k in usage}}


def run(step,prompt,config,workdir=None,timeout=1800):
    """step: {"agent": "claude"|"codex", "model": ..., "effort": ...}"""
    agent=config["agents"][step["agent"]]
    system=config.get("system_append")
    start=time.perf_counter()
    if step["agent"]=="claude":
        cmd=[agent["command"],"-p",prompt,"--output-format","json","--no-session-persistence"]
        if step.get("model"):
            cmd+=["--model",step["model"]]
        if step.get("effort"):
            cmd+=["--effort",step["effort"]]
        if step.get("lean"):
            cmd+=CLAUDE_LEAN
            if step.get("system_prompt"):
                cmd+=["--system-prompt",step["system_prompt"]]
        elif step.get("toolset") is not None:
            # Only these built-in tools, no MCP servers. Settings files stay loaded
            # so the swarm's permission rules still apply. Measured fixed input on a
            # trivial question: Read 3,905; Read,Grep,Glob 5,458; Bash,Read,Edit,Write
            # 6,031; all tools 20,933.
            cmd+=["--tools",step["toolset"],"--strict-mcp-config"]
        if system:
            cmd+=["--append-system-prompt",system]
        cmd+=agent.get("extra_args",[])
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,cwd=workdir,stdin=subprocess.DEVNULL)
        result=parse_claude(proc.stdout) if proc.returncode==0 and proc.stdout.strip() else {"text":"","error":True,"usage":{}}
    else:
        with tempfile.NamedTemporaryFile("r",suffix=".txt",delete=False) as f:
            out_path=f.name
        try:
            full=prompt if not system else f"{system}\n\n{prompt}"
            cmd=[agent["command"],"exec","--json","-o",out_path]
            if step.get("model"):
                cmd+=["-m",step["model"]]
            if step.get("effort"):
                cmd+=["-c",f"{agent.get('effort_key','model_reasoning_effort')}={step['effort']}"]
            cmd+=agent.get("extra_args",[])+[full]
            # Codex reads stdin when it is open ("Reading additional input from
            # stdin..."); a swarm worker must never hang on that.
            proc=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,cwd=workdir,stdin=subprocess.DEVNULL)
            last=open(out_path).read() if os.path.exists(out_path) else ""
            result=parse_codex(proc.stdout,last) if proc.returncode==0 else {"text":last,"error":True,"usage":{}}
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)
    result["seconds"]=time.perf_counter()-start
    result["returncode"]=proc.returncode
    if proc.returncode!=0:
        result["stderr_tail"]=proc.stderr[-500:]
    return result
