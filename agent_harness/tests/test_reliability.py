"""Failure paths and the compaction fast path: timeouts, error exits, bad
output, crashing runners, rung isolation, and exactness of lost_facts."""
import os
import random
import stat
import sys
import time

from agent_harness import agents
from agent_harness import compact as C
from agent_harness.run import load_config,run_task
from agent_harness.verify import check


def _script(path,body):
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode|stat.S_IEXEC)
    return str(path)


def _usage_runner(texts):
    calls=[]
    def run(step,prompt,config,workdir=None):
        calls.append(step)
        return {"text":texts[len(calls)-1],"error":False,"usage":{"input_tokens":100,"output_tokens":10},"seconds":0.}
    return run,calls


# ---- agents.run --------------------------------------------------------------

def test_timeout_returns_error_and_keeps_partial_usage(tmp_path):
    cmd=_script(tmp_path/"claude",
        "import sys,time,json\n"
        "print(json.dumps({'result':'partial','usage':{'input_tokens':7,'output_tokens':2}}),flush=True)\n"
        "time.sleep(30)\n")
    start=time.perf_counter()
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}}},timeout=1)
    assert time.perf_counter()-start<10
    assert r["error"] and r["timed_out"] and r["returncode"] is None
    assert r["usage"]=={"input_tokens":7,"output_tokens":2}      # spent tokens still counted


def test_error_exit_keeps_usage_and_blanks_text(tmp_path):
    cmd=_script(tmp_path/"claude",
        "import sys,json\n"
        "print(json.dumps({'result':'API Error: overloaded','is_error':True,'usage':{'input_tokens':50,'output_tokens':0}}))\n"
        "sys.exit(1)\n")
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}}})
    assert r["error"] and r["text"]=="" and r["usage"]["input_tokens"]==50 and r["returncode"]==1


def test_non_json_output_is_an_error_not_a_crash(tmp_path):
    cmd=_script(tmp_path/"claude","print('Warning: something odd')\nprint('{not json')\n")
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}}})
    assert r["error"] and r["usage"]=={} and r.get("parse_error")
    # a warning line before the JSON result no longer breaks parsing
    cmd=_script(tmp_path/"claude2","import json\nprint('Warning: x')\nprint(json.dumps({'result':'ok','usage':{'input_tokens':1,'output_tokens':1}}))\n")
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}}})
    assert not r["error"] and r["text"]=="ok"


def test_codex_cut_off_event_line_is_skipped(tmp_path):
    cmd=_script(tmp_path/"codex",
        "import sys,json\nargs=sys.argv\nopen(args[args.index('-o')+1],'w').write('Final answer: 7')\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':20,'output_tokens':6}}))\nprint('{\"type\":\"item')\n")
    r=agents.run({"agent":"codex"},"q",{"agents":{"codex":{"command":cmd,"extra_args":[]}}})
    assert not r["error"] and r["usage"]["input_tokens"]==20


# ---- check and run_task ------------------------------------------------------

def test_check_cmd_timeout_fails_the_check(tmp_path):
    v=check({"check_cmd":f"{sys.executable} -c 'import time; time.sleep(30)'"},"",str(tmp_path),timeout=1)
    assert v["passed"] is False and v["timed_out"]


def test_runner_crash_keeps_earlier_usage():
    calls=[]
    def run(step,prompt,config,workdir=None):
        calls.append(step)
        if len(calls)==2:
            raise FileNotFoundError("claude")
        return {"text":"Final answer: 1","error":False,"usage":{"input_tokens":100,"output_tokens":10},"seconds":0.}
    report=run_task({"prompt":"What is 3+4?","expected":"7"},load_config(),runner=run)
    assert len(report["attempts"])==2 and report["tokens_spent"]==110
    assert report["verified"] is False and "FileNotFoundError" in report["exception"]


def test_unchecked_crash_reports_unverified():
    def run(step,prompt,config,workdir=None):
        raise RuntimeError("boom")
    report=run_task({"prompt":"Explain TCP"},load_config(),runner=run)
    assert report["verified"] is None and report["text"]=="" and "boom" in report["exception"]


# ---- rung isolation ----------------------------------------------------------

def _project(root):
    (root/"pkg").mkdir()
    (root/"pkg"/"a.py").write_text("x = 1\n")
    (root/"keep_empty").mkdir()
    (root/"node_modules").mkdir()
    (root/"node_modules"/"dep.js").write_text("dep\n")


def test_each_rung_starts_from_the_original_files(tmp_path):
    _project(tmp_path)
    seen=[]
    def run(step,prompt,config,workdir=None):
        seen.append(((tmp_path/"pkg"/"a.py").read_text(),sorted(os.listdir(tmp_path))))
        # every rung breaks things in a different way
        (tmp_path/"pkg"/"a.py").write_text(f"x = {len(seen)+1}  # broken\n")
        (tmp_path/f"junk{len(seen)}.txt").write_text("junk")
        (tmp_path/"newdir"/"deep").mkdir(parents=True,exist_ok=True)
        (tmp_path/"node_modules"/"dep.js").write_text("changed\n")    # SKIP dirs are left alone
        return {"text":"","error":False,"usage":{"input_tokens":1,"output_tokens":1},"seconds":0.}
    report=run_task({"prompt":"fix","check_cmd":"false","workdir":str(tmp_path)},load_config(),runner=run)
    assert len(seen)==4
    first=seen[0]
    assert all(s==first for s in seen)                                 # same starting files every rung
    assert first[0]=="x = 1\n" and "keep_empty" in first[1] and "newdir" not in first[1]
    iso=report["rung_isolation"]
    assert iso["restores"]==3 and iso["skipped"] is None
    assert (tmp_path/"keep_empty").is_dir()                            # original empty dir survives
    assert (tmp_path/"node_modules"/"dep.js").read_text()=="changed\n"


def test_passing_rung_keeps_its_edits(tmp_path):
    _project(tmp_path)
    def run(step,prompt,config,workdir=None):
        (tmp_path/"pkg"/"a.py").write_text("x = 2\n")
        return {"text":"","error":False,"usage":{},"seconds":0.}
    report=run_task({"prompt":"fix","check_cmd":"grep -q 'x = 2' pkg/a.py","workdir":str(tmp_path)},load_config(),runner=run)
    assert report["verified"] and (tmp_path/"pkg"/"a.py").read_text()=="x = 2\n"
    assert report["rung_isolation"]["restores"]==0


def test_isolation_off_or_oversized_shares_the_workdir(tmp_path):
    _project(tmp_path)
    config=load_config(); config["isolate_rungs_max_mb"]=0
    run,_=_usage_runner(["x"]*4)
    report=run_task({"prompt":"fix","check_cmd":"false","workdir":str(tmp_path)},config,runner=run)
    assert report["rung_isolation"]["skipped"] and report["rung_isolation"]["restores"]==0
    config["isolate_rungs"]=False
    run,_=_usage_runner(["x"]*4)
    assert "rung_isolation" not in run_task({"prompt":"fix","check_cmd":"false","workdir":str(tmp_path)},config,runner=run)


# ---- compaction fast path ----------------------------------------------------

def _slow_lost(text,out):
    return C.facts(C.BLOB.sub("",text))-C.facts(out)


def _random_text(rng):
    pool=["INFO ok","WARN retry db:5432","2026-10-29 10:00:01 heartbeat 3","2026-10-29 10:00:02 heartbeat 3",
          "value=42 at https://example.com/a?b=1","ValueError: bad","", "  ","path src/mod.py:12","x"*20,
          "A"*220,"deadbeef"*30,"```","print('same')","n=%d"]
    lines=[]
    for _ in range(rng.randint(0,60)):
        l=rng.choice(pool)
        if "%d" in l:
            l=l%rng.randint(0,5)
        lines+= [l]*rng.choice([1,1,1,2,5])
    return "\n".join(lines)


def test_lost_facts_matches_the_full_scan():
    rng=random.Random(0)
    for _ in range(400):
        text=_random_text(rng)
        out,_=C.compact(text)
        assert C.lost_facts(text,out)==_slow_lost(text,out)
        # and against arbitrary (buggy) outputs, where facts really go missing
        mangled="\n".join(l for l in out.split("\n") if rng.random()<0.7)
        assert C.lost_facts(text,mangled)==_slow_lost(text,mangled)


def _reference_collapse_blocks(lines,max_period=6):
    out,i=[],0
    while i<len(lines):
        for k in range(2,max_period+1):
            block=lines[i:i+k]
            if len(block)<k or not any(l.strip() for l in block):
                continue
            reps=1
            while lines[i+reps*k:i+(reps+1)*k]==block:
                reps+=1
            if reps>1:
                out+=block+[f"  [previous {k} lines repeated {reps}x]"]
                i+=reps*k
                break
        else:
            out.append(lines[i])
            i+=1
    return out


def test_collapse_blocks_unchanged_by_the_precheck():
    rng=random.Random(1)
    for _ in range(3000):
        lines=[rng.choice(["a","b","c",""," "]) for _ in range(rng.randint(0,40))]
        assert C._collapse_blocks(lines)==_reference_collapse_blocks(lines)


def test_compact_checked_unique_lines_is_fast():
    text="\n".join(f"line {i} value={i*7} path=src/mod{i}.py" for i in range(10000))
    start=time.perf_counter()
    out,rep=C.compact_checked(text)
    assert out==text and rep["fallback"]=="no saving"
    assert time.perf_counter()-start<0.5


def test_zero_usage_falls_back_to_model_usage():
    import json as _j
    out=_j.dumps({"result":"","is_error":True,"subtype":"error_max_budget_usd","total_cost_usd":0.05,
                  "usage":{"input_tokens":0,"output_tokens":0},
                  "modelUsage":{"claude-sonnet":{"inputTokens":3,"outputTokens":40,"cacheReadInputTokens":100,"cacheCreationInputTokens":9000}}})
    r=agents.parse_claude(out)
    assert r["usage"]=={"input_tokens":3,"output_tokens":40,"cache_read_input_tokens":100,"cache_creation_input_tokens":9000}
    assert r["error_text"]=="error_max_budget_usd" and r["cost_usd"]==0.05
