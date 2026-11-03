"""Frozen extension tasks: no layer/prompt search on these outputs is permitted."""
import ast
import json
import re
import subprocess
import sys

HELDOUT = [
    {"prompt":"What is 38 plus 47?", "answer":"85"},
    {"prompt":"What is 23 times 17?", "answer":"391"},
    {"prompt":"What is 18 percent of 350?", "answer":"63"},
    {"prompt":"Solve 7x - 11 = 80 for x.","answer":"13"},
    {"prompt":"A car travels 154 miles in 2 hours and 86 miles in 3 hours. What is its average speed in miles per hour?","answer":"48"},
    {"prompt":"A 120 dollar item is discounted 25 percent, then taxed 8 percent. What is its final price?","answer":"97.2"},
    {"prompt":"What is 13/16 of 240?","answer":"195"},
    {"prompt":"What is 47 squared minus 39 squared?","answer":"688"},
]
CODING = [
 {"prompt":"Write a Python function dedup(items) returning a list of first occurrences in original order. Items are hashable. Handle empty input. Do not modify input.",
  "function":"dedup","cases":[[[[]],[]],[[[3,1,3,2,1]],[3,1,2]],[[[None,False,None,False]],[None,False]]]},
 {"prompt":"Write a Python function runs(values) that returns a list of (value, count) tuples for consecutive equal values. Handle empty input. Do not modify input.",
  "function":"runs","cases":[[[[]],[]],[[[1,1,2,1]],[[1,2],[2,1],[1,1]]],[[['a','a','a']],[['a',3]]]]},
 {"prompt":"Write a Python function lower_bound(a, x) returning the first index i where a[i] >= x in a sorted list, or len(a) if none. Use O(log n) time, no imports, and handle empty lists.",
  "function":"lower_bound","cases":[[[[],3],0],[[[1,2,2,4],2],1],[[[1,2,4],5],3],[[[1,2],0],0]]},
 {"prompt":"Write a Python function balanced(s) that returns whether all (), [], {} brackets are balanced and correctly nested, ignoring all other characters. Handle empty strings.",
  "function":"balanced","cases":[[[""],True],[["a{b[(c)]}"],True],[["([)]"],False],[[")("],False],[["(()"],False]]},
]


def score_code(text, task):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```",text,re.S)
    source = blocks[-1] if blocks else text
    try:
        tree=ast.parse(source)
        # This limited harness rejects imports/dunders/dynamic execution before
        # a fresh bounded child process runs code. It is NOT a general sandbox
        # for arbitrary Python; rejected constructs count as unsupported/fail.
        if any(not isinstance(n,(ast.FunctionDef,ast.Expr)) or isinstance(n,ast.Expr) and not isinstance(n.value,ast.Constant) for n in tree.body):
            raise ValueError("Only function definitions and docstrings supported")
        forbidden={"eval","exec","compile","open","input","getattr","setattr","globals","locals","__import__","breakpoint"}
        for n in ast.walk(tree):
            if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.AsyncFunctionDef,ast.Global,ast.Nonlocal)):
                raise ValueError("Unsupported construct")
            if isinstance(n,ast.Name) and (n.id in forbidden or "__" in n.id):
                raise ValueError("Unsafe name")
            if isinstance(n,ast.Attribute) and n.attr.startswith("_"):
                raise ValueError("Private attribute")
            if isinstance(n,ast.FunctionDef) and (n.decorator_list or n.args.defaults or n.args.kw_defaults and any(n.args.kw_defaults)):
                raise ValueError("Decorators/default expressions unsupported")
        payload={"source":source,"function":task["function"],"cases":task["cases"]}
        child = """
import json,sys,resource
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
data=json.load(sys.stdin)
safe={k:__builtins__.__dict__[k] for k in ['len','range','enumerate','zip','list','tuple','set','dict','min','max','sum','abs','sorted','reversed','all','any','isinstance','int','str','bool','float','ValueError']}
ns={'__builtins__':safe}
exec(compile(data['source'],'<generated>','exec'),ns)
results=[]
for args,expected in data['cases']:
    before=json.dumps(args,sort_keys=True)
    try:
        actual=ns[data['function']](*args)
        results.append(json.loads(json.dumps(actual))==expected and before==json.dumps(args,sort_keys=True))
    except Exception:
        results.append(False)
print(json.dumps(results))
"""
        run=subprocess.run([sys.executable,"-I","-c",child],input=json.dumps(payload),text=True,capture_output=True,timeout=4)
        results=json.loads(run.stdout) if run.returncode == 0 else []
        return {"passed":len(results)==len(task["cases"]) and all(results),"case_results":results,"error":run.stderr[-500:] or None}
    except (SyntaxError,ValueError,subprocess.TimeoutExpired) as exc:
        return {"passed":False,"case_results":[],"error":str(exc)}
