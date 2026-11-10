"""Objective checks. A task is technical when it brings one.

Escalating to a stronger model is only as safe as the check: weak tests can
accept a wrong cheap answer. Prefer real test suites; for functions, the
EvalPlus-style differential cases in lean_harness show how much hand-written
cases miss.

A task can carry any of these (legacy top-level keys, or a "checks" list of
{"type": ..., ...} specs; every check must pass):

  expected      - final answer equals this (research grader, normalized)
  function+cases- generated function passes the cases in a sandboxed child
  check_cmd     - shell command in the workdir passes on exit code 0
  regex         - re.search(pattern, answer) matches            {"pattern", "flags"?}
  contains      - every string appears in the answer             {"values": [...]}
  json          - answer is JSON (optionally a fenced block) valid against a
                  schema                                         {"schema"?}
                  A top-level "output_schema" adds this check and is also passed
                  to the model as its structured-output schema.

Only expected/cases/cmd count as correctness checks (has_strong_check): they
route a task to the technical ladder and its minimal first rung. Format checks
alone keep the task's normal full setup; a format failure escalates upward.

Cheap checks run first and the first failure stops the rest, so a wrong answer
never pays for a slow test suite. The JSON validator covers a common subset of
JSON Schema and fails closed: a keyword it does not implement fails the check
rather than letting an unchecked answer through.
"""
import json
import re
import subprocess

ORDER={"contains":0,"regex":0,"json":1,"expected":2,"cases":3,"cmd":4}


def specs(task):
    out=[]
    if task.get("contains") is not None:
        out.append({"type":"contains","values":task["contains"] if isinstance(task["contains"],list) else [task["contains"]]})
    if task.get("regex") is not None:
        out.append({"type":"regex","pattern":task["regex"]})
    if task.get("output_schema") is not None:
        out.append({"type":"json","schema":task["output_schema"]})
    if task.get("expected") is not None:
        out.append({"type":"expected","value":task["expected"]})
    if task.get("function") and task.get("cases"):
        out.append({"type":"cases","function":task["function"],"cases":task["cases"]})
    if task.get("check_cmd"):
        out.append({"type":"cmd","cmd":task["check_cmd"]})
    out+=list(task.get("checks") or [])
    return sorted(out,key=lambda s:ORDER.get(s.get("type"),5))


def has_check(task):
    return bool(specs(task))


# Checks that establish the answer is right, not just well-formed. Only these may
# stand behind a minimal first rung: a format check (regex, contains, json)
# would accept a fluent wrong answer from an under-equipped attempt.
STRONG={"expected","cases","cmd"}


def has_strong_check(task):
    return any(s.get("type") in STRONG for s in specs(task))


def check(task,text,workdir=None,timeout=None):
    """Returns {"passed": bool|None, "how": ...}. Commands are the swarm's own,
    never model output."""
    timeout=timeout or task.get("check_timeout") or 600
    workdir=workdir or task.get("workdir")
    todo=specs(task)
    if not todo:
        return {"passed":None,"how":"none"}
    done=[]
    for spec in todo:
        verdict=_one(spec,text,workdir,timeout)
        if len(todo)==1:
            return verdict
        done.append(verdict["how"])
        if not verdict["passed"]:
            return {**verdict,"how":"+".join(done),"failed":spec.get("type")}
    return {"passed":True,"how":"+".join(done)}


def _one(spec,text,workdir,timeout):
    kind=spec.get("type")
    text=text or ""
    if kind=="expected":
        from independent_audit.scoring_v2 import correct      # research grader, loaded only when used
        return {"passed":correct(text,str(spec["value"])),"how":"expected"}
    if kind=="cases":
        from independent_audit.code_quality import evaluate_code
        verdict=evaluate_code(text,{"function":spec["function"],"cases":spec["cases"]})
        return {"passed":bool(verdict["passed"]),"how":"cases","detail":verdict.get("functional")}
    if kind=="cmd":
        try:
            proc=subprocess.run(spec["cmd"],shell=True,capture_output=True,text=True,timeout=spec.get("timeout",timeout),cwd=workdir)
        except subprocess.TimeoutExpired:
            # A hung check is a failed check, not a crashed task: the attempts
            # already made (and their usage) must still reach the report.
            return {"passed":False,"how":"check_cmd","timed_out":True}
        return {"passed":proc.returncode==0,"how":"check_cmd","tail":(proc.stdout+proc.stderr)[-800:]}
    if kind=="regex":
        flags=0
        for f in spec.get("flags",""):
            flags|={"i":re.I,"m":re.M,"s":re.S}[f]
        return {"passed":re.search(spec["pattern"],text,flags) is not None,"how":"regex"}
    if kind=="contains":
        missing=[v for v in spec["values"] if str(v) not in text]
        return {"passed":not missing,"how":"contains",**({"missing":missing[:5]} if missing else {})}
    if kind=="json":
        try:
            value=extract_json(text)
        except ValueError as exc:
            return {"passed":False,"how":"json","error":str(exc)[:200]}
        errors=validate(value,spec.get("schema") or {})
        return {"passed":not errors,"how":"json",**({"errors":errors[:5]} if errors else {})}
    return {"passed":False,"how":f"unknown check {kind!r}"}      # fail closed


FENCED=re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```",re.S)


def extract_json(text):
    """The answer as JSON: the whole text, else the last fenced block."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    for block in reversed(FENCED.findall(text)):
        try:
            return json.loads(block)
        except ValueError:
            continue
    raise ValueError("answer is not JSON")


TYPES={"object":dict,"array":list,"string":str,"boolean":bool,"null":type(None)}
KNOWN={"type","properties","required","items","enum","const","additionalProperties","minItems","maxItems",
       "minimum","maximum","exclusiveMinimum","exclusiveMaximum","minLength","maxLength","pattern",
       "title","description","$schema","$id","default","examples","uniqueItems"}


def _is(value,t):
    if t=="integer":
        return isinstance(value,int) and not isinstance(value,bool) or isinstance(value,float) and value.is_integer()
    if t=="number":
        return isinstance(value,(int,float)) and not isinstance(value,bool)
    return isinstance(value,TYPES[t]) if t in TYPES else False


def validate(value,schema,path="$"):
    """Errors for value against a JSON Schema subset; unknown keywords are errors."""
    if schema is True or schema=={}:
        return []
    if schema is False:
        return [f"{path}: no value allowed"]
    unknown=set(schema)-KNOWN
    if unknown:
        return [f"{path}: unsupported schema keyword(s) {sorted(unknown)} (check fails closed)"]
    errs=[]
    t=schema.get("type")
    if t is not None:
        ts=t if isinstance(t,list) else [t]
        if not any(_is(value,x) for x in ts):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: not in enum")
    if "const" in schema and value!=schema["const"]:
        errs.append(f"{path}: not the const value")
    if isinstance(value,(int,float)) and not isinstance(value,bool):
        for k,ok in (("minimum",lambda a,b:a>=b),("maximum",lambda a,b:a<=b),
                     ("exclusiveMinimum",lambda a,b:a>b),("exclusiveMaximum",lambda a,b:a<b)):
            if k in schema and not ok(value,schema[k]):
                errs.append(f"{path}: {k} {schema[k]}")
    if isinstance(value,str):
        if "minLength" in schema and len(value)<schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value)>schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"],value):
            errs.append(f"{path}: does not match pattern")
    if isinstance(value,list):
        if "minItems" in schema and len(value)<schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value)>schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and len({json.dumps(v,sort_keys=True) for v in value})<len(value):
            errs.append(f"{path}: items not unique")
        if "items" in schema:
            for i,v in enumerate(value):
                errs+=validate(v,schema["items"],f"{path}[{i}]")
    if isinstance(value,dict):
        props=schema.get("properties",{})
        for k in schema.get("required",[]):
            if k not in value:
                errs.append(f"{path}: missing {k!r}")
        for k,v in value.items():
            if k in props:
                errs+=validate(v,props[k],f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected {k!r}")
            elif isinstance(schema.get("additionalProperties"),dict):
                errs+=validate(v,schema["additionalProperties"],f"{path}.{k}")
    return errs
