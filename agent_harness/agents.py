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


def _as_text(x):
    """TimeoutExpired carries partial output as bytes even with text=True."""
    if x is None:
        return ""
    return x.decode(errors="replace") if isinstance(x,bytes) else x


def _load_json(stdout):
    """The result object, even if other lines (warnings) were printed around it."""
    try:
        data=json.loads(stdout)
        if isinstance(data,dict):
            return data
    except ValueError:
        pass
    for line in reversed(stdout.splitlines()):
        line=line.strip()
        if line.startswith("{"):
            try:
                data=json.loads(line)
            except ValueError:
                continue
            if isinstance(data,dict):
                return data
    return None


def parse_claude(stdout):
    data=_load_json(stdout or "")
    if data is None:
        return {"text":"","error":True,"usage":{},"parse_error":True}
    usage=next(_usage_dicts(data.get("usage",data)),{})
    # total_cost_usd is a list-price estimate; on a subscription it is notional
    # but still a consistent weight across models for cost-per-correct.
    text=data.get("result","")
    if data.get("structured_output") is not None:     # --json-schema runs: the validated object
        text=json.dumps(data["structured_output"])
    out={"text":text if isinstance(text,str) else json.dumps(text),"error":bool(data.get("is_error")),
         "usage":{k:usage.get(k,0) for k in USAGE_KEYS if k in usage},"cost_usd":data.get("total_cost_usd"),
         "raw":{k:data.get(k) for k in ("num_turns","modelUsage","total_cost_usd","usage","subtype")}}
    if out["error"]:
        out["error_text"]=str(data.get("result") or data.get("subtype") or "")[:500]
    return out


def parse_codex(stdout,last_message):
    events=[]
    for l in (stdout or "").splitlines():
        if l.strip().startswith("{"):
            try:
                e=json.loads(l)
            except ValueError:
                continue                    # a cut-off line (e.g. killed on timeout)
            if isinstance(e,dict):
                events.append(e)
    found=[u for e in events for u in _usage_dicts(e)]
    usage=found[-1] if found else {}      # the latest report; confirm cumulative vs per-turn in the pilot
    failed=[e for e in events if e.get("type") in ("error","turn.failed")]
    out={"text":last_message,"error":bool(failed),"usage":{k:usage.get(k,0) for k in USAGE_KEYS if k in usage}}
    if failed:
        out["error_text"]=json.dumps(failed[-1])[:500]
    return out


def _exec(cmd,timeout,workdir):
    """(returncode, stdout, stderr, timed_out). A timeout keeps the partial
    output, so usage already reported is still counted, and never raises."""
    try:
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,cwd=workdir,stdin=subprocess.DEVNULL)
        return proc.returncode,proc.stdout,proc.stderr,False
    except subprocess.TimeoutExpired as exc:
        return None,_as_text(exc.stdout),_as_text(exc.stderr),True


# Codex sandbox per toolset: a read-only sandbox can never pass a coding task's
# check, so a code rung needs workspace-write (still confined to the workdir).
CODEX_SANDBOX={"none":"read-only","read":"read-only","code":"workspace-write","full":"workspace-write"}


def run(step,prompt,config,workdir=None,timeout=None):
    """step: {"agent": "claude"|"codex", "model": ..., "effort": ...}
    config may carry agent_timeout (seconds), output_schema (JSON Schema dict)
    and task_toolset (none/read/code/full, picks the Codex sandbox)."""
    timeout=timeout or config.get("agent_timeout") or 1800
    schema=config.get("output_schema")
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
        if schema:
            cmd+=["--json-schema",json.dumps(schema)]
        cmd+=agent.get("extra_args",[])
        rc,out,err,timed_out=_exec(cmd,timeout,workdir)
        # A failed run still printed its usage: count it, or the budget undercounts
        # exactly when calls fail and escalate.
        result=parse_claude(out)
        if rc!=0:
            result["text"]=""
            result["error"]=True
    else:
        with tempfile.NamedTemporaryFile("r",suffix=".txt",delete=False) as f:
            out_path=f.name
        schema_path=None
        try:
            if schema:
                with tempfile.NamedTemporaryFile("w",suffix=".json",delete=False) as f:
                    json.dump(schema,f)
                    schema_path=f.name
            full=prompt if not system else f"{system}\n\n{prompt}"
            cmd=[agent["command"],"exec","--json","-o",out_path]
            if step.get("model"):
                cmd+=["-m",step["model"]]
            if step.get("effort"):
                cmd+=["-c",f"{agent.get('effort_key','model_reasoning_effort')}={step['effort']}"]
            extra=list(agent.get("extra_args",[]))
            if "-s" not in extra and "--sandbox" not in extra:
                sandbox=agent.get("sandbox",CODEX_SANDBOX).get(config.get("task_toolset") or "none","read-only")
                extra+=["-s",sandbox]
            if schema_path:
                extra+=["--output-schema",schema_path]
            cmd+=extra+[full]
            # Codex reads stdin when it is open ("Reading additional input from
            # stdin..."); a swarm worker must never hang on that.
            rc,out,err,timed_out=_exec(cmd,timeout,workdir)
            last=open(out_path).read() if os.path.exists(out_path) else ""
            result=parse_codex(out,last)
            if rc!=0:
                result["error"]=True
        finally:
            for path in (out_path,schema_path):
                if path and os.path.exists(path):
                    os.remove(path)
    result["seconds"]=time.perf_counter()-start
    result["returncode"]=rc
    if timed_out:
        result["timed_out"]=True
    if rc!=0:
        result["stderr_tail"]=err[-500:]
        if not result.get("error_text"):
            result["error_text"]=(err or "")[-500:]
    return result
