"""Independent recheck of the toolset first rung on harder, multi-file tasks.

Why: pilot_toolset_v1 used three easy single-file tasks written by the person
testing the idea, both conditions scored 6/6 (a ceiling), and usage
aggregation over multi-turn Claude runs was never verified. Here:
  - three small multi-file projects (bug fixes across modules, and a feature
    implemented from a docstring), each graded by HIDDEN tests kept outside
    the agent's working directory and never mentioned to it;
  - every project is self-checked first: it must fail its hidden tests as
    shipped and pass them once correctly fixed;
  - each run keeps Claude's raw JSON, so top-level usage can be compared with
    the per-model totals (does "usage" cover every turn?);
  - results are reported raw and price-weighted, and the decision uses the same
    zero-loss rule as every other gate.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]      # repo root; avoids importing torch via independent_audit.model
from . import agents
from .pilot import CLAUDE_TOOL_ARGS,single
from .run import load_config,run_task

TEST_CMD="python3 -m unittest discover -s tests -q"

PROJECTS={
"inventory":{
 "prompt":"The unit tests in this project fail. Find and fix the bug(s) in the package so the code matches its docstrings. Check with: "+TEST_CMD,
 "files":{
  "inventory/__init__.py":"",
  "inventory/models.py":'from dataclasses import dataclass\n\n\n@dataclass\nclass Item:\n    sku: str\n    price_cents: int\n    qty: int\n',
  "inventory/pricing.py":'"""Pricing rules.\n\nA percentage discount rounds half up to the nearest cent.\nOrders of 10 or more units of one item get the bulk discount.\n"""\nBULK_THRESHOLD = 10\nBULK_PERCENT = 15\n\n\ndef apply_discount(price_cents, percent):\n    """Price after a percent discount, rounded half up to the nearest cent."""\n    return int(price_cents * (100 - percent) / 100)\n\n\ndef line_total(item):\n    """Total for one line: the bulk discount applies at BULK_THRESHOLD units or more."""\n    unit = apply_discount(item.price_cents, BULK_PERCENT) if item.qty > BULK_THRESHOLD else item.price_cents\n    return unit * item.qty\n',
  "inventory/store.py":'from .pricing import line_total\n\n\nclass Store:\n    def __init__(self):\n        self.items = []\n\n    def add(self, item):\n        self.items.append(item)\n\n    def total(self):\n        """Sum of line totals in cents."""\n        return sum(line_total(i) for i in self.items)\n',
  "inventory/report.py":'from .pricing import line_total\n\n\ndef summary(store):\n    """One line per item: \'<sku> x<qty> = <cents>\', then \'TOTAL <cents>\'."""\n    lines = [f"{i.sku} x{i.qty} = {line_total(i)}" for i in store.items]\n    lines.append(f"TOTAL {store.total()}")\n    return "\\n".join(lines)\n',
  "tests/test_store.py":'import unittest\nfrom inventory.models import Item\nfrom inventory.store import Store\n\n\nclass TestStore(unittest.TestCase):\n    def test_bulk_at_threshold(self):\n        s = Store()\n        s.add(Item("A", 199, 10))\n        self.assertEqual(s.total(), 1690)\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
 },
 "hidden":'import sys\nsys.path.insert(0, ".")\nfrom inventory.models import Item\nfrom inventory.pricing import apply_discount, line_total\nfrom inventory.store import Store\nfrom inventory.report import summary\nassert apply_discount(199, 15) == 169\nassert apply_discount(250, 10) == 225\nassert apply_discount(5, 50) == 3\nassert apply_discount(1, 50) == 1\nassert apply_discount(1000, 0) == 1000\nassert line_total(Item("A", 199, 9)) == 1791\nassert line_total(Item("A", 199, 10)) == 1690\nassert line_total(Item("A", 199, 11)) == 1859\ns = Store(); s.add(Item("A", 199, 10)); s.add(Item("B", 5, 2))\nassert s.total() == 1700\nassert summary(s) == "A x10 = 1690\\nB x2 = 10\\nTOTAL 1700"\nprint("hidden ok")\n',
 "fix":lambda d:(d/"inventory/pricing.py").write_text((d/"inventory/pricing.py").read_text()
     .replace("return int(price_cents * (100 - percent) / 100)","return (price_cents * (100 - percent) + 50) // 100")
     .replace("if item.qty > BULK_THRESHOLD","if item.qty >= BULK_THRESHOLD")),
},
"exprcalc":{
 "prompt":"This calculator gives wrong answers for some expressions. Fix the package so every expression follows normal arithmetic rules (precedence, left-to-right for operators of equal precedence, parentheses, unary minus). Check with: "+TEST_CMD,
 "files":{
  "exprcalc/__init__.py":"",
  "exprcalc/tokenizer.py":'def tokenize(text):\n    """Integers become int tokens; + - * / ( ) become one-character strings."""\n    out, num = [], ""\n    for ch in text + " ":\n        if ch.isdigit():\n            num += ch\n            continue\n        if num:\n            out.append(int(num))\n            num = ""\n        if ch in "+-*/()":\n            out.append(ch)\n        elif not ch.isspace():\n            raise ValueError(f"bad character {ch!r}")\n    return out\n',
  "exprcalc/parser.py":'"""Recursive-descent parser producing nested tuples (op, left, right) or ("neg", x)."""\n\n\nclass Parser:\n    def __init__(self, tokens):\n        self.tokens, self.i = tokens, 0\n\n    def peek(self):\n        return self.tokens[self.i] if self.i < len(self.tokens) else None\n\n    def take(self):\n        tok = self.peek()\n        self.i += 1\n        return tok\n\n    def expr(self):\n        left = self.term()\n        if self.peek() in ("+", "-"):\n            op = self.take()\n            return (op, left, self.expr())\n        return left\n\n    def term(self):\n        left = self.factor()\n        while self.peek() in ("*", "/"):\n            op = self.take()\n            left = (op, left, self.factor())\n        return left\n\n    def factor(self):\n        tok = self.take()\n        if tok == "-":\n            return ("neg", self.factor())\n        if tok == "(":\n            node = self.expr()\n            if self.take() != ")":\n                raise ValueError("expected )")\n            return node\n        if isinstance(tok, int):\n            return tok\n        raise ValueError(f"unexpected {tok!r}")\n\n\ndef parse(tokens):\n    p = Parser(tokens)\n    node = p.expr()\n    if p.peek() is not None:\n        raise ValueError("trailing tokens")\n    return node\n',
  "exprcalc/evaluator.py":'from fractions import Fraction\nfrom .tokenizer import tokenize\nfrom .parser import parse\n\n\ndef evaluate(node):\n    if isinstance(node, int):\n        return Fraction(node)\n    if node[0] == "neg":\n        return -evaluate(node[1])\n    op, a, b = node\n    a, b = evaluate(a), evaluate(b)\n    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]\n\n\ndef calc(text):\n    """Exact value of an arithmetic expression, as a Fraction."""\n    return evaluate(parse(tokenize(text)))\n',
  "tests/test_calc.py":'import unittest\nfrom exprcalc.evaluator import calc\n\n\nclass TestCalc(unittest.TestCase):\n    def test_precedence(self):\n        self.assertEqual(calc("2+3*4"), 14)\n\n    def test_subtraction_chain(self):\n        self.assertEqual(calc("10-4-3"), 3)\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
 },
 "hidden":'import sys\nsys.path.insert(0, ".")\nfrom fractions import Fraction\nfrom exprcalc.evaluator import calc\ncases = {"2+3*4": 14, "10-4-3": 3, "20-5+2": 17, "100/10/5": 2, "8/4*2": 4, "-3+5": 2, "-(2+3)*2": -10, "2*(3+4)-5": 9, "1-2-3-4": -8, "7": 7, "2-3*4+5": -5}\nfor e, v in cases.items():\n    assert calc(e) == v, (e, calc(e), v)\nassert calc("1/3+1/6") == Fraction(1, 2)\nprint("hidden ok")\n',
 "fix":lambda d:(d/"exprcalc/parser.py").write_text((d/"exprcalc/parser.py").read_text().replace(
     '        left = self.term()\n        if self.peek() in ("+", "-"):\n            op = self.take()\n            return (op, left, self.expr())\n        return left',
     '        left = self.term()\n        while self.peek() in ("+", "-"):\n            op = self.take()\n            left = (op, left, self.term())\n        return left')),
},
"slots":{
 "prompt":"Implement free_slots in slots/calendar.py exactly as its docstring describes, reusing the helpers in slots/timeutil.py. Check with: "+TEST_CMD,
 "files":{
  "slots/__init__.py":"",
  "slots/timeutil.py":'def to_minutes(hhmm):\n    """\'09:30\' -> 570."""\n    h, m = hhmm.split(":")\n    return int(h) * 60 + int(m)\n\n\ndef to_hhmm(minutes):\n    """570 -> \'09:30\'."""\n    return f"{minutes // 60:02d}:{minutes % 60:02d}"\n',
  "slots/calendar.py":'from .timeutil import to_minutes, to_hhmm\n\n\ndef free_slots(events, day_start, day_end, min_minutes):\n    """Free time between day_start and day_end ("HH:MM" strings).\n\n    events: list of (start, end) "HH:MM" pairs, in any order; they may overlap,\n    touch, or extend outside the day. Busy time is the union of the events,\n    clipped to the day. Return the gaps as ("HH:MM", "HH:MM") pairs in time\n    order, keeping only gaps of at least min_minutes.\n    """\n    raise NotImplementedError\n',
  "tests/test_calendar.py":'import unittest\nfrom slots.calendar import free_slots\n\n\nclass TestSlots(unittest.TestCase):\n    def test_basic(self):\n        self.assertEqual(free_slots([("10:00", "11:00")], "09:00", "12:00", 30),\n                         [("09:00", "10:00"), ("11:00", "12:00")])\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
 },
 "hidden":'import sys\nsys.path.insert(0, ".")\nfrom slots.calendar import free_slots\nassert free_slots([("10:00", "11:00")], "09:00", "12:00", 30) == [("09:00", "10:00"), ("11:00", "12:00")]\nassert free_slots([("10:30", "11:30"), ("10:00", "11:00")], "09:00", "12:00", 30) == [("09:00", "10:00"), ("11:30", "12:00")]\nassert free_slots([("10:00", "11:00"), ("11:00", "11:20")], "09:00", "12:00", 30) == [("09:00", "10:00"), ("11:20", "12:00")]\nassert free_slots([("08:00", "09:30"), ("11:45", "13:00")], "09:00", "12:00", 30) == [("09:30", "11:45")]\nassert free_slots([("09:10", "09:50")], "09:00", "10:30", 15) == [("09:50", "10:30")]\nassert free_slots([], "09:00", "10:00", 60) == [("09:00", "10:00")]\nassert free_slots([], "09:00", "10:00", 61) == []\nassert free_slots([("07:00", "08:00")], "09:00", "10:00", 30) == [("09:00", "10:00")]\nassert free_slots([("09:00", "12:00")], "09:00", "12:00", 1) == []\nprint("hidden ok")\n',
 "fix":lambda d:(d/"slots/calendar.py").write_text((d/"slots/calendar.py").read_text().replace("    raise NotImplementedError\n",
     "    lo, hi = to_minutes(day_start), to_minutes(day_end)\n    busy = sorted((max(lo, to_minutes(a)), min(hi, to_minutes(b))) for a, b in events)\n    out, cur = [], lo\n    for a, b in busy:\n        if b <= a:\n            continue\n        if a - cur >= min_minutes:\n            out.append((to_hhmm(cur), to_hhmm(a)))\n        cur = max(cur, b)\n    if hi - cur >= min_minutes:\n        out.append((to_hhmm(cur), to_hhmm(hi)))\n    return out\n")),
},
}


def materialize(name,root):
    d=Path(root)/name
    for rel,text in PROJECTS[name]["files"].items():
        (d/rel).parent.mkdir(parents=True,exist_ok=True)
        (d/rel).write_text(text)
    return d


def hidden_check(name,hidden_dir):
    path=Path(hidden_dir)/f"hidden_{name}.py"
    path.write_text(PROJECTS[name]["hidden"])
    return f"python3 {path}"


def self_check():
    """Every project must fail its hidden tests as shipped and pass once fixed."""
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as hid:
        for name in PROJECTS:
            d=materialize(name,root)
            cmd=hidden_check(name,hid)
            before=subprocess.run(cmd,shell=True,cwd=d,capture_output=True).returncode
            PROJECTS[name]["fix"](d)
            after=subprocess.run(cmd,shell=True,cwd=d,capture_output=True,text=True)
            visible=subprocess.run(TEST_CMD,shell=True,cwd=d,capture_output=True).returncode
            if before==0 or after.returncode!=0 or visible!=0:
                raise RuntimeError(f"{name}: shipped passes={before==0}, fixed passes={after.returncode==0}, visible after fix={visible==0} {after.stderr[-300:]}")
    return True


def raw_runner(store):
    """agents.run, keeping the raw Claude JSON for the usage-aggregation check."""
    def run(step,prompt,config,workdir=None):
        result=agents.run(step,prompt,config,workdir)
        store.append(result.get("raw"))
        return result
    return run


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="agent_harness/results/recheck_toolset_v1")
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()
    self_check()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    base=load_config()
    minimal=base["toolsets"]["code"]
    rows=[]
    root=Path(tempfile.mkdtemp(prefix="recheck_"))
    hid=Path(tempfile.mkdtemp(prefix="hidden_"))
    try:
        with (out/"rows.jsonl").open("a") as f:
            for rep in range(args.repeats):
                for name in PROJECTS:
                    for condition in random.Random(f"R{rep}{name}").sample(["full_tools","code_toolset"],2):
                        d=materialize(name,root/f"{condition}_{rep}")
                        task={"prompt":PROJECTS[name]["prompt"],"check_cmd":hidden_check(name,hid),"workdir":str(d)}
                        cfg=single(base,"claude","sonnet","low",extra=CLAUDE_TOOL_ARGS+["Bash(python3 -m unittest*)"],toolset=minimal if condition=="code_toolset" else None)
                        raws=[]
                        report=run_task(task,cfg,runner=raw_runner(raws))
                        raw=raws[-1] or {}
                        mu=raw.get("modelUsage",{})
                        mu_input=sum(v.get("inputTokens",0)+v.get("cacheReadInputTokens",0)+v.get("cacheCreationInputTokens",0) for v in mu.values())
                        row={"condition":condition,"project":name,"repeat":rep,"verified":report["verified"],"usage":report["usage_total"],
                             "num_turns":raw.get("num_turns"),"modelUsage_input_total":mu_input,"cost_usd":raw.get("total_cost_usd"),
                             "seconds":report["attempts"][-1]["seconds"] if report["attempts"] else None}
                        f.write(json.dumps(row)+"\n"); f.flush()
                        rows.append(row)
                        print(f"R {condition:13} {name:10} rep={rep} verified={row['verified']} turns={row['num_turns']} input={row['usage']['input_total']} modelUsage_input={mu_input} out={row['usage'].get('output_tokens')} cost=${row['cost_usd']}",flush=True)
    finally:
        shutil.rmtree(root,ignore_errors=True)
        shutil.rmtree(hid,ignore_errors=True)
    by={}
    for r in rows:
        e=by.setdefault(r["condition"],{"n":0,"correct":0,"tokens":0,"cost_usd":0.,"turns":0})
        e["n"]+=1; e["correct"]+=bool(r["verified"]); e["tokens"]+=r["usage"]["input_total"]+r["usage"].get("output_tokens",0)
        e["cost_usd"]+=r["cost_usd"] or 0; e["turns"]+=r["num_turns"] or 0
    pairs={(r["project"],r["repeat"]):{} for r in rows}
    for r in rows:
        pairs[(r["project"],r["repeat"])][r["condition"]]=bool(r["verified"])
    lost=sorted(f"{p}#{k}" for (p,k),v in pairs.items() if v.get("full_tools") and not v.get("code_toolset"))
    usage_consistent=all(abs(r["usage"]["input_total"]-r["modelUsage_input_total"])<=1 for r in rows)
    summary={"by_condition":by,"losses":lost,"usage_matches_modelUsage":usage_consistent}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
