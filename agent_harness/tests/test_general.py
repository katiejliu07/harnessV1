"""General-task features: check types, structured output, transient retries,
batches, verbatim context files, Codex sandbox, CLI."""
import json
import sys
import threading
import time

from agent_harness import agents
from agent_harness.run import load_config,main,run_task,run_tasks,toolset_of,transient
from agent_harness.verify import check,has_check,validate
from _portable import TRUE,fake_cli


def runner_of(texts,errors=None,usage=None):
    calls=[]
    def run(step,prompt,config,workdir=None):
        calls.append({"step":step,"prompt":prompt,"config":config})
        i=len(calls)-1
        err=(errors or {}).get(i)
        return {"text":"" if err else texts[i],"error":bool(err),"error_text":err or "",
                "usage":usage or {"input_tokens":100,"output_tokens":10},"seconds":0.}
    return run,calls


# ---- checks ------------------------------------------------------------------

def test_format_checks_never_unlock_the_minimal_rung():
    from agent_harness.run import route
    for t in ({"regex":r"\b42\b"},{"contains":["ok"]},{"output_schema":{"type":"object"}},{"checks":[{"type":"regex","pattern":"x"}]}):
        task={"prompt":"p",**t}
        assert has_check(task) and route(task)=="general"
        run,calls=runner_of(["no match"]*5)
        report=run_task(task,load_config(),runner=run)
        # full default setup first (no lean, no toolset), then only stronger rungs
        assert not calls[0]["step"].get("lean") and "toolset" not in calls[0]["step"]
        assert [(c["step"]["model"],c["step"]["effort"]) for c in calls]==[("sonnet","low"),("sonnet","high"),("opus","high")]
        assert report["verified"] is False
    # a passing format check stops after the one normal call
    run,calls=runner_of(['{"a": 1}'])
    assert run_task({"prompt":"p","output_schema":{"type":"object"}},load_config(),runner=run)["verified"] and len(calls)==1
    # important + format check: Opus high, nothing stronger to climb to
    run,calls=runner_of(["x"]*3)
    run_task({"prompt":"p","important":True,"regex":"y"},load_config(),runner=run)
    assert len(calls)==1 and calls[0]["step"]["model"]=="opus"
    # a correctness check still gets the technical ladder
    assert route({"prompt":"p","regex":"x","expected":"7"})=="technical"
    assert not has_check({"prompt":"p"})


def test_regex_contains_and_combined_checks():
    assert check({"regex":r"answer: \d+"},"the answer: 12")["passed"]
    assert not check({"regex":r"answer: \d+"},"no number")["passed"]
    assert check({"checks":[{"type":"regex","pattern":"HELLO","flags":"i"}]},"hello")["passed"]
    v=check({"contains":["alpha","beta"]},"alpha only")
    assert not v["passed"] and v["missing"]==["beta"]
    # every check must pass; cheap ones run first and stop the expensive ones
    v=check({"contains":["x"],"check_cmd":TRUE},"no")
    assert not v["passed"] and v["failed"]=="contains" and "check_cmd" not in v["how"]
    assert check({"contains":["x"],"check_cmd":TRUE},"x")["passed"]


def test_unknown_check_fails_closed():
    assert check({"checks":[{"type":"llm_judge"}]},"anything")["passed"] is False


def test_json_schema_validation():
    schema={"type":"object","required":["name","n"],"additionalProperties":False,
            "properties":{"name":{"type":"string","minLength":1},"n":{"type":"integer","minimum":0},
                          "tags":{"type":"array","items":{"enum":["a","b"]},"uniqueItems":True}}}
    assert validate({"name":"x","n":3,"tags":["a"]},schema)==[]
    assert validate({"name":"x"},schema)                       # missing n
    assert validate({"name":"x","n":-1},schema)                # below minimum
    assert validate({"name":"x","n":1,"extra":1},schema)       # additional property
    assert validate({"name":"x","n":True},schema)              # bool is not an integer
    assert validate({"name":"x","n":1,"tags":["a","a"]},schema)
    # unsupported keywords never let an answer through unchecked
    assert validate({"a":1},{"type":"object","oneOf":[{"required":["a"]}]})
    assert check({"output_schema":schema},'Here you go:\n```json\n{"name":"x","n":2}\n```')["passed"]
    assert not check({"output_schema":schema},"not json at all")["passed"]


def test_toolset_inferred_from_cmd_in_checks_list():
    assert toolset_of({"prompt":"p","checks":[{"type":"cmd","cmd":"pytest"}]})=="code"
    assert toolset_of({"prompt":"p","regex":"x"})=="none"


# ---- structured output and CLI flags ----------------------------------------

def _fake(tmp_path,name,body):
    return fake_cli(tmp_path/f"{name}.py",body)


def _args_file(cmd):
    return cmd[-1]+".args"                      # the fake writes sys.argv[0]+'.args'


def test_schema_reaches_claude_and_structured_output_is_the_answer(tmp_path):
    cmd=_fake(tmp_path,"claude","import sys,json\nopen(sys.argv[0]+'.args','w').write(json.dumps(sys.argv))\n"
              "print(json.dumps({'result':'','structured_output':{'n':3},'usage':{'input_tokens':1,'output_tokens':1}}))\n")
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}},"output_schema":{"type":"object"}})
    argv=json.loads(open(_args_file(cmd)).read())
    assert argv[argv.index("--json-schema")+1]=='{"type": "object"}' and json.loads(r["text"])=={"n":3}


def test_codex_sandbox_follows_the_toolset(tmp_path):
    cmd=_fake(tmp_path,"codex","import sys,json\nopen(sys.argv[0]+'.args','w').write(json.dumps(sys.argv))\n"
              "a=sys.argv\nopen(a[a.index('-o')+1],'w').write('ok')\n")
    config={"agents":{"codex":{"command":cmd,"extra_args":[]}}}
    for toolset,mode in (("none","read-only"),("read","read-only"),("code","workspace-write")):
        agents.run({"agent":"codex"},"q",{**config,"task_toolset":toolset})
        argv=json.loads(open(_args_file(cmd)).read())
        assert argv[argv.index("-s")+1]==mode
    # an explicit sandbox in extra_args wins
    agents.run({"agent":"codex"},"q",{"agents":{"codex":{"command":cmd,"extra_args":["-s","read-only"]}},"task_toolset":"code"})
    argv=json.loads(open(_args_file(cmd)).read())
    assert argv.count("-s")==1 and argv[argv.index("-s")+1]=="read-only"


def test_task_timeout_reaches_the_runner():
    run,calls=runner_of(["x"])
    run_task({"prompt":"Explain","timeout":42},load_config(),runner=run)
    assert calls[0]["config"]["agent_timeout"]==42


# ---- transient retries -------------------------------------------------------

def test_transient_error_retries_same_rung_before_escalating():
    config=load_config()
    run,calls=runner_of(["","Final answer: 7"],errors={0:"API Error: 529 Overloaded"})
    slept=[]
    report=run_task({"prompt":"What is 3+4?","expected":"7"},config,runner=run,sleep=slept.append)
    assert len(calls)==2 and calls[0]["step"]==calls[1]["step"]          # same rung, not a pricier one
    assert report["verified"] and report["attempts"][0]["transient"] and report["attempts"][1]["retry"]==1
    assert report["tokens_spent"]==220 and slept==[5]                   # both tries are paid for


def test_real_failures_and_harness_timeouts_are_not_retried():
    config=load_config()
    assert not transient({"error":True,"error_text":"Invalid model name"},config)
    assert not transient({"error":True,"timed_out":True,"error_text":"overloaded"},config)
    assert not transient({"error":False},config)
    run,calls=runner_of(["","Final answer: 7"],errors={0:"bad request"})
    report=run_task({"prompt":"What is 3+4?","expected":"7"},config,runner=run,sleep=lambda s:None)
    assert calls[1]["step"]!=calls[0]["step"] and "retry" not in report["attempts"][1]


def test_retries_are_bounded():
    config=load_config()
    errs={i:"rate limit exceeded" for i in range(20)}
    run,calls=runner_of([""]*20,errors=errs)
    report=run_task({"prompt":"Explain"},config,runner=run,sleep=lambda s:None)
    assert len(calls)==1+config["retry"]["transient_retries"] and report["verified"] is None


# ---- context files -----------------------------------------------------------

def test_context_files_are_verbatim(tmp_path):
    src="def f():\n    pass\n    pass\n    pass\n"           # repeated lines compaction would collapse
    (tmp_path/"m.py").write_text(src)
    run,calls=runner_of(["x"])
    run_task({"prompt":"Review","context":"\n".join(["WARN connection retry pending"]*3),"context_files":["m.py"],"workdir":str(tmp_path)},load_config(),runner=run)
    assert src in calls[0]["prompt"] and "[line repeated 3x]" in calls[0]["prompt"]


# ---- batches -----------------------------------------------------------------

def test_batch_runs_in_parallel_and_keeps_order():
    active=[0]; peak=[0]; lock=threading.Lock()
    def run(step,prompt,config,workdir=None):
        with lock:
            active[0]+=1; peak[0]=max(peak[0],active[0])
        time.sleep(0.2)
        with lock:
            active[0]-=1
        return {"text":prompt,"error":False,"usage":{"input_tokens":1,"output_tokens":1},"seconds":0.2}
    tasks=[{"prompt":f"task {i}"} for i in range(6)]
    start=time.perf_counter()
    reports=run_tasks(tasks,load_config(),jobs=6,runner=run)
    assert [r["text"] for r in reports]==[t["prompt"] for t in tasks]
    assert peak[0]>1 and time.perf_counter()-start<3.0


def test_batch_serializes_tasks_sharing_a_workdir(tmp_path):
    active=[0]; peak=[0]; lock=threading.Lock()
    def run(step,prompt,config,workdir=None):
        with lock:
            active[0]+=1; peak[0]=max(peak[0],active[0])
        time.sleep(0.1)
        with lock:
            active[0]-=1
        return {"text":"ok","error":False,"usage":{},"seconds":0.}
    tasks=[{"prompt":"p","workdir":str(tmp_path)} for _ in range(4)]
    run_tasks(tasks,load_config(),jobs=4,runner=run)
    assert peak[0]==1


def test_batch_survives_a_bad_task():
    run,_=runner_of(["ok","ok"])
    reports=run_tasks([{"prompt":"fine"},{"no_prompt":True}],load_config(),jobs=2,runner=run)
    assert reports[0]["text"]=="ok" and "exception" in reports[1]


# ---- CLI ---------------------------------------------------------------------

def test_cli_inline_task_and_exit_codes(tmp_path,capsys,monkeypatch):
    from agent_harness import run as R
    monkeypatch.setattr(R.agents,"run",lambda step,prompt,config,workdir=None:
                        {"text":"Final answer: 7","error":False,"usage":{"input_tokens":1,"output_tokens":1},"seconds":0.})
    assert main(["--prompt","What is 3+4?","--expected","7"])==0
    assert capsys.readouterr().out.strip()=="Final answer: 7"
    assert main(["--prompt","What is 3+4?","--expected","8"])==1
    f=tmp_path/"tasks.jsonl"
    f.write_text('{"prompt":"a"}\n{"prompt":"b","expected":"7"}\n')
    capsys.readouterr()
    assert main([str(f),"--jobs","2"])==0
    lines=[json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [l["route"] for l in lines]==["general","technical"]


def test_code_tasks_can_edit_and_shell_stays_opt_in(tmp_path):
    cmd=_fake(tmp_path,"claude","import sys,json\nopen(sys.argv[0]+'.args','a').write(json.dumps(sys.argv)+'\\n')\n"
              "print(json.dumps({'result':'done','usage':{'input_tokens':1,'output_tokens':1}}))\n")
    config=load_config(); config["agents"]["claude"]["command"]=cmd
    run_task({"prompt":"fix","check_cmd":TRUE,"workdir":str(tmp_path),"allowed_tools":["Bash(pytest*)"]},config)
    argv=json.loads(open(_args_file(cmd)).read().splitlines()[0])
    assert argv[argv.index("--permission-mode")+1]=="acceptEdits"
    assert argv[argv.index("--allowedTools")+1]=="Bash(pytest*)" and TRUE not in argv   # check_cmd never leaked
    open(_args_file(cmd),"w").close()
    run_task({"prompt":"What is 3+4?","expected":"7"},config)            # lean rung, no tools: no permissions
    argv=json.loads(open(_args_file(cmd)).read().splitlines()[0])
    assert "--permission-mode" not in argv and "--allowedTools" not in argv


def test_usd_budget_is_passed_down_and_stops_the_ladder():
    calls=[]
    def run(step,prompt,config,workdir=None):
        calls.append(config.get("max_budget_usd"))
        return {"text":"Final answer: 1","error":False,"usage":{"input_tokens":1,"output_tokens":1},"seconds":0.,"cost_usd":0.03}
    report=run_task({"prompt":"What is 3+4?","expected":"7","max_budget_usd":0.05},load_config(),runner=run)
    assert calls[0]==0.05 and abs(calls[1]-0.02)<1e-9 and len(calls)==2      # each call gets what is left
    assert report["budget_exhausted"] and abs(report["cost_usd"]-0.06)<1e-9
    # without the key nothing is passed
    run2,c2=runner_of(["Final answer: 7"])
    run_task({"prompt":"What is 3+4?","expected":"7"},load_config(),runner=run2)
    assert "max_budget_usd" not in c2[0]["config"]


def test_declared_toolset_is_honored_on_unchecked_tasks_inferred_is_not(tmp_path):
    run,calls=runner_of(["Paris"])
    run_task({"prompt":"Capital of France?","toolset":"none"},load_config(),runner=run)
    assert calls[0]["step"]["toolset"]=="" and not calls[0]["step"].get("lean")      # tools off, default system prompt kept
    run,calls=runner_of(["ok"])
    run_task({"prompt":"Explain this repo","workdir":str(tmp_path)},load_config(),runner=run)   # inferred "code": not applied
    assert "toolset" not in calls[0]["step"]
    run,calls=runner_of(["ok"])
    run_task({"prompt":"Research X","toolset":"full"},load_config(),runner=run)
    assert "toolset" not in calls[0]["step"]


def test_streaming_only_for_unchecked_tasks():
    run,calls=runner_of(["a","Final answer: 7"])
    cb=lambda t:None
    run_task({"prompt":"Hi"},load_config(),runner=run,on_text=cb)
    assert calls[0]["config"]["on_text"] is cb
    run_task({"prompt":"3+4?","expected":"7"},load_config(),runner=run,on_text=cb)
    assert "on_text" not in calls[1]["config"]


def test_stream_parsing_and_bad_callback(tmp_path):
    ev=lambda d:json.dumps({"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":d}}})
    body=("import json,sys,time\n"
          f"print({ev('Par')!r},flush=True)\ntime.sleep(0.3)\nprint({ev('is')!r},flush=True)\n"
          "print(json.dumps({'type':'result','result':'Paris','usage':{'input_tokens':5,'output_tokens':2}}))\n")
    cmd=_fake(tmp_path,"claude",body)
    got=[]; t0=time.perf_counter()
    def cb(x): got.append((x,time.perf_counter()-t0))
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}},"on_text":cb})
    assert [g[0] for g in got]==["Par","is"] and got[0][1]<got[1][1]-0.2          # first words arrive before the end
    assert r["text"]=="Paris" and r["usage"]=={"input_tokens":5,"output_tokens":2} and not r["error"]
    def boom(x): raise RuntimeError("ui died")
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}},"on_text":boom})
    assert r["text"]=="Paris" and not r["error"]                                    # answer survives a broken callback


def test_stream_timeout(tmp_path):
    cmd=_fake(tmp_path,"claude","import time\ntime.sleep(30)\n")
    t=time.perf_counter()
    r=agents.run({"agent":"claude"},"q",{"agents":{"claude":{"command":cmd}},"on_text":lambda x:None},timeout=1)
    assert r["timed_out"] and r["error"] and time.perf_counter()-t<10
