import json
import os

from agent_harness.compact import compact,compact_checked,facts
from agent_harness.run import load_config,route,run_task
from agent_harness import agents
from _portable import FALSE,TRUE,fake_cli


def fake_runner(answers):
    """Replays answers per call and records what was asked; no real agent runs."""
    calls=[]
    def run(step,prompt,config,workdir=None):
        calls.append((step,prompt,config.get("system_append")))
        text=answers[len(calls)-1]
        return {"text":text,"error":False,"usage":{"input_tokens":100,"output_tokens":len(text)},"seconds":0.}
    return run,calls


def test_routing_needs_an_objective_check():
    assert route({"prompt":"Explain how TCP slow start works"})=="general"
    assert route({"prompt":"What is 3+4?","expected":"7"})=="technical"
    assert route({"prompt":"fix it","check_cmd":"pytest -q"})=="technical"
    assert route({"prompt":"write f","function":"f","cases":[([1],[1])]})=="technical"
    assert route({"prompt":"anything","kind":"general","expected":"1"})=="general"
    # important without a check goes straight to the strongest model
    assert route({"prompt":"Pick the database architecture","important":True})=="important"
    assert route({"prompt":"What is 3+4?","expected":"7","important":True})=="technical"


def test_escalates_only_on_failed_check_and_stops_at_first_pass():
    config=load_config()
    run,calls=fake_runner(["Final answer: 6","Final answer: 7","Final answer: 7"])
    report=run_task({"prompt":"What is 3+4?","expected":"7"},config,runner=run)
    assert len(calls)==2 and report["verified"] is True
    assert [a["verified"] for a in report["attempts"]]==[False,True]
    assert report["usage_total"]["input_tokens"]==200
    # first try passes: no escalation spend at all
    run,calls=fake_runner(["Final answer: 7"])
    assert run_task({"prompt":"What is 3+4?","expected":"7"},config,runner=run)["attempts"][0]["verified"] and len(calls)==1


def test_all_rungs_failing_is_reported_unverified_not_hidden():
    run,calls=fake_runner(["Final answer: 1"]*4)
    report=run_task({"prompt":"What is 3+4?","expected":"7"},load_config(),runner=run)
    assert len(calls)==4 and report["verified"] is False   # lean rung + 3 ladder rungs


def test_general_path_is_one_call_and_compacts_context():
    log="\n".join(["2026-10-29 10:00:%02d,000 INFO heartbeat ok"%i for i in range(30)]+["ERROR disk full on /dev/sda1"])
    run,calls=fake_runner(["ok"])
    report=run_task({"prompt":"Why did it stop?","context":log},load_config(),runner=run)
    assert len(calls)==1 and report["route"]=="general"
    assert "disk full on /dev/sda1" in calls[0][1] and report["compaction"]["chars_out"]<report["compaction"]["chars_in"]


def test_code_cases_check():
    run,_=fake_runner(["```python\ndef f(x):\n    return x[::-1]\n```"])
    report=run_task({"prompt":"reverse","function":"f","cases":[([[1,2]],[2,1])]},load_config(),runner=run)
    assert report["verified"] is True


def test_compaction_never_loses_facts_or_touches_code():
    code="```python\nfor i in range(3):\n    print('same')\n    print('same')\n```"
    text="\n".join(["WARN retry db:5432"]*20+[code,"x"*0+"value=42 at https://example.com/a?b=1"])
    out,_=compact(text)
    assert code in out and facts(text)<=facts(out)
    assert "[line repeated 20x]" in out
    # the guard returns the original if anything factual would be lost
    out2,rep=compact_checked("A 1\nA 1\nB 2")
    assert facts("A 1\nA 1\nB 2")<=facts(out2) and rep.get("fallback")!="fact lost"


def test_cli_output_parsing_with_fake_commands(tmp_path):
    claude=fake_cli(tmp_path/"claude.py","import json\nprint(json.dumps({'type':'result','result':'Final answer: 7','is_error':False,'usage':{'input_tokens':12,'output_tokens':5,'cache_read_input_tokens':3}}))\n")
    codex=fake_cli(tmp_path/"codex.py","import json,sys\na=sys.argv\nopen(a[a.index('-o')+1],'w').write('Final answer: 7\\n')\n"
                   "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':20,'cached_input_tokens':4,'output_tokens':6}}))\n")
    config={"agents":{"claude":{"command":claude},"codex":{"command":codex,"extra_args":[]}}}
    a=agents.run({"agent":"claude","model":"sonnet","effort":"low"},"q",config)
    assert a["text"]=="Final answer: 7" and a["usage"]=={"input_tokens":12,"output_tokens":5,"cache_read_input_tokens":3}
    b=agents.run({"agent":"codex","model":"m","effort":"low"},"q",config)
    assert b["text"].strip()=="Final answer: 7" and b["usage"]["input_tokens"]==20 and b["usage"]["cached_input_tokens"]==4


def test_important_path_is_one_call_on_the_strongest_model():
    run,calls=fake_runner(["Use Postgres."])
    report=run_task({"prompt":"Pick the database architecture","important":True},load_config(),runner=run)
    assert len(calls)==1 and calls[0][0]["model"]=="opus" and report["verified"] is None


def test_toolset_rung_only_for_tool_tasks_and_hygiene_only_with_tools(tmp_path):
    config=load_config()
    for p in config["profiles"].values():
        p["toolset_first_rung"]=True; p["hygiene"]=True
    # tool task: the declared/inferred toolset first, then the full-tools ladder
    run,calls=fake_runner(["x"]*4)            # check always fails: toolset rung + 3 ladder rungs
    report=run_task({"prompt":"fix the bug","check_cmd":FALSE,"workdir":str(tmp_path)},config,runner=run)
    assert calls[0][0]["toolset"]=="Bash,Read,Edit,Write" and "toolset" not in calls[1][0]
    assert report["toolset"]=="code"
    run,calls=fake_runner(["x"]*4)
    run_task({"prompt":"look","check_cmd":FALSE,"workdir":str(tmp_path),"toolset":"read"},config,runner=run)
    assert calls[0][0]["toolset"]=="Read,Grep,Glob"
    # tool-free task gets the lean rung instead, never a toolset rung
    run,calls=fake_runner(["Final answer: 7"])
    run_task({"prompt":"What is 3+4?","expected":"7"},config,runner=run)
    assert calls[0][0].get("lean") and "toolset" not in calls[0][0] and not calls[0][2]
    run,calls=fake_runner(["done"])
    run_task({"prompt":"fix the bug","check_cmd":TRUE,"workdir":str(tmp_path)},config,runner=run)
    step,_,system=calls[0]
    assert not step.get("lean") and not system          # claude: hygiene measured no saving
    config["hygiene_agents"]=["claude","codex"]
    run,calls=fake_runner(["done"])
    run_task({"prompt":"fix the bug","check_cmd":TRUE,"workdir":str(tmp_path)},config,runner=run)
    assert "compact.py" in calls[0][2]


def test_handoff_is_compact_result_not_transcript():
    run,_=fake_runner(["\n".join(["WARN retry"]*30+["Final answer: 7"])])
    report=run_task({"prompt":"What is 3+4?","expected":"7"},load_config(),runner=run)
    assert report["handoff"]["verified"] is True and "[line repeated 30x]" in report["handoff"]["result"]
    assert report["usage_total"]["input_total"]==100


def test_compaction_never_grows_text():
    out,rep=compact_checked("a\na\na")
    assert out=="a\na\na" or len(out)<5


def test_every_pilot_task_is_graded():
    from agent_harness.pilot import hygiene_task,lean_tasks
    from agent_harness.verify import check,has_check
    import tempfile
    tasks=lean_tasks()
    assert len(tasks)==12 and all(has_check(t) for t in tasks)
    # the checker really grades: a right and a wrong answer to the first word problem
    t=tasks[0]
    assert check(t,f"Final answer: {t['expected']}")["passed"] and not check(t,"Final answer: 123456")["passed"]
    with tempfile.TemporaryDirectory() as root:
        for name in ["log","tests","data"]:
            task=hygiene_task(name,root)
            assert has_check(task) and check(task,"",task["workdir"])["passed"] is False   # unsolved task fails its check


def test_lean_first_rung_falls_through_to_the_unchanged_ladder(tmp_path):
    config=load_config()
    assert config["profiles"]["technical"]["lean_first_rung"] is True
    # lean miss -> default rung passes: same answer the ladder would give, one cheap extra try
    run,calls=fake_runner(["Final answer: 6","Final answer: 7"])
    report=run_task({"prompt":"What is 3+4?","expected":"7"},config,runner=run)
    assert [c[0].get("lean",False) for c in calls]==[True,False] and report["verified"] is True
    assert calls[1][0]["model"]=="sonnet" and calls[1][0]["effort"]=="low"
    # tool tasks never get a lean rung; unchecked paths never do either
    run,calls=fake_runner(["done"])
    run_task({"prompt":"fix","check_cmd":TRUE,"workdir":str(tmp_path)},config,runner=run)
    assert not calls[0][0].get("lean")
    run,calls=fake_runner(["x"])
    run_task({"prompt":"Pick the database architecture","important":True},config,runner=run)
    assert not calls[0][0].get("lean") and len(calls)==1


def test_hygiene_reaches_codex_tool_runs_only(tmp_path):
    config=load_config()
    assert config["hygiene_agents"]==["codex"]
    config["profiles"]["technical"]["ladder"]=[{"agent":"codex","model":None,"effort":"low"}]
    run,calls=fake_runner(["done"])
    run_task({"prompt":"fix","check_cmd":TRUE,"workdir":str(tmp_path)},config,runner=run)
    assert "compact.py" in calls[0][2]
    run,calls=fake_runner(["Final answer: 7"])
    run_task({"prompt":"What is 3+4?","expected":"7"},config,runner=run)   # no tools: no hygiene
    assert not calls[0][2]


def test_budget_stops_escalation_and_says_so():
    config=load_config()
    run,calls=fake_runner(["Final answer: 1"]*4)                 # each attempt "costs" 100 input + 15 output
    report=run_task({"prompt":"What is 3+4?","expected":"7","token_budget":200},config,runner=run)
    assert len(calls)==2 and report["budget_exhausted"] is True and report["verified"] is False
    assert report["tokens_spent"]==2*(100+len("Final answer: 1"))
    run,calls=fake_runner(["Final answer: 7"])
    report=run_task({"prompt":"What is 3+4?","expected":"7","token_budget":200},config,runner=run)
    assert report["budget_exhausted"] is False and report["verified"] is True


def test_cli_toolset_flags(tmp_path):
    claude=fake_cli(tmp_path/"claude.py","import json,os,sys\nopen(os.path.join(os.path.dirname(sys.argv[0]),'args.txt'),'w').write(' '.join(sys.argv[1:]))\n"
                    "print(json.dumps({'result':'ok','usage':{'input_tokens':1,'output_tokens':1}}))\n")
    config={"agents":{"claude":{"command":claude}}}
    agents.run({"agent":"claude","model":"sonnet","toolset":"Bash,Read,Edit,Write"},"q",config)
    args=(tmp_path/"args.txt").read_text()
    assert "--tools Bash,Read,Edit,Write --strict-mcp-config" in args and "--setting-sources" not in args   # permissions kept
