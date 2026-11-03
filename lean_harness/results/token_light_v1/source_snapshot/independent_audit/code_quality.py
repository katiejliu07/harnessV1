"""Supplemental executable grader, authored before coding generations began.

Do not count a valid itertools/typing solution as incorrect merely because the
initial restricted grader rejected imports. Keep raw scores, then independently
regrade with explicit safe module facades. Never expose a real module (typing.sys
would bypass a naive import allowlist). This remains a restricted task harness,
not a security boundary for arbitrary untrusted Python.
"""
import ast
import json
import re
import subprocess
import sys

ALLOWED={"itertools":{"groupby","chain","zip_longest","repeat"},
         "collections":{"Counter","defaultdict","deque"},
         "math":{"ceil","floor","sqrt","isqrt","log2"},
         "re":{"sub","findall","compile","match","fullmatch","IGNORECASE","MULTILINE","DOTALL"},
         "functools":{"reduce"},"bisect":{"bisect_left","bisect_right"},
         "typing":{"List","Tuple","Dict","Set","Optional","Union","Sequence","Iterable","Any"}}


def solution_source(text,function):
    # A response can contain a complete function but hit its budget in later
    # examples before closing the Markdown fence. Test the unmodified Python
    # body too; response truncation is independently retained by report.py.
    # Do not repair incomplete Python or manufacture a missing function return.
    blocks=re.findall(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)",text,re.S)
    candidates=[b for b in blocks if re.search(r"\bdef\s+"+re.escape(function)+r"\s*\(",b)]
    if len(candidates)>1:
        raise ValueError("Multiple solution blocks require manual adjudication")
    source=candidates[0] if candidates else blocks[0] if len(blocks)==1 else text
    tree=ast.parse(source)
    body=[]
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)):
            body.append(n)
        elif isinstance(n,ast.Assign):
            ast.literal_eval(n.value)  # permit harmless module-level constants only
            body.append(n)
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant):
            body.append(n)
        # Examples after the requested function are not part of function tests.
    if sum(isinstance(n,ast.FunctionDef) and n.name==function for n in body)!=1:
        raise ValueError("Exactly one requested function is required")
    tree.body=body
    forbidden={"eval","exec","compile","open","input","getattr","setattr","globals","locals","__import__","breakpoint"}
    for n in ast.walk(tree):
        if isinstance(n,(ast.ClassDef,ast.AsyncFunctionDef,ast.Global,ast.Nonlocal)):
            raise ValueError("Unsupported construct")
        if isinstance(n,ast.Name) and (n.id in forbidden or "__" in n.id):
            raise ValueError("Unsafe name")
        if isinstance(n,ast.Attribute) and n.attr.startswith("_"):
            raise ValueError("Private attribute")
        if isinstance(n,ast.Import):
            if any(x.name not in ALLOWED for x in n.names):
                raise ValueError("Unsupported import")
        if isinstance(n,ast.ImportFrom):
            if n.level or n.module not in ALLOWED or any(x.name not in ALLOWED[n.module] for x in n.names):
                raise ValueError("Unsupported from-import")
        if isinstance(n,ast.FunctionDef) and (n.decorator_list or n.args.defaults or any(n.args.kw_defaults)):
            raise ValueError("Decorators/default expressions require manual adjudication")
    if function=="lower_bound" and any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree)):
        raise ValueError("This task explicitly prohibits imports")
    return ast.unparse(tree)


CHILD=r'''
import json,sys,resource,types,itertools,collections,math,functools,bisect,typing,re
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
# Darwin rejects RLIMIT_AS here; do not turn every valid solution into a false
# failure. CPU/timeout limits remain, and this harness is not a general sandbox.
if sys.platform!='darwin':
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
data=json.load(sys.stdin)
modules={'itertools':itertools,'collections':collections,'math':math,'functools':functools,'bisect':bisect,'typing':typing,'re':re}
facades={name:types.SimpleNamespace(**{attr:getattr(module,attr) for attr in data['allowed'][name]}) for name,module in modules.items()}
def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
    if level or name not in facades: raise ImportError(name)
    return facades[name]
safe={k:__builtins__.__dict__[k] for k in ['len','range','enumerate','zip','iter','next','map','filter','list','tuple','set','dict','min','max','sum','abs','sorted','reversed','all','any','isinstance','int','str','bool','float','ValueError','Exception']}
safe['__import__']=safe_import
ns={'__builtins__':safe}
exec(compile(data['source'],'<generated>','exec'),ns)
functional=[]
for args,expected in data['cases']:
    before=json.dumps(args,sort_keys=True)
    try:
        actual=ns[data['function']](*args)
        shape_ok=(type(actual) is bool if data['function']=='balanced' else
                  type(actual) is int if data['function']=='lower_bound' else
                  type(actual) is list and all(type(pair) is tuple and len(pair)==2 for pair in actual) if data['function']=='runs' else
                  type(actual) is list)
        # Python equality makes True==1 and False==0. Serialized value equality
        # plus explicit return contracts prevents those false-positive scores.
        functional.append(shape_ok and json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)
                          and before==json.dumps(args,sort_keys=True))
    except Exception:
        functional.append(False)
complexity=[]
class CountedList(list):
    def charge(self,n):
        self.accesses+=n
        if self.accesses>self.limit: raise RuntimeError('Nonlogarithmic access budget exceeded')
    def __getitem__(self,key):
        self.charge(len(range(*key.indices(len(self)))) if isinstance(key,slice) else 1)
        return super().__getitem__(key)
    def __iter__(self):
        for i in range(len(self)): yield self[i]
    def index(self,*args):
        self.charge(len(self))
        return super().index(*args)
    def copy(self):
        self.charge(len(self))
        return super().copy()
if data['function']=='lower_bound':
    for n in [256,4096,65536]:
        raw=[i//3 for i in range(n)]
        for x in [-1,0,n//6,raw[-1],raw[-1]+1]:
            a=CountedList(raw)
            a.accesses=0
            a.limit=2*math.ceil(math.log2(n+1))+8
            try:
                value=ns['lower_bound'](a,x)
                passed=type(value) is int and value==bisect.bisect_left(raw,x)
                complexity.append({'n':n,'x':x,'accesses':a.accesses,'budget':a.limit,'passed':passed})
            except Exception as e:
                complexity.append({'n':n,'x':x,'accesses':a.accesses,'budget':a.limit,'passed':False,'error':str(e)})
print(json.dumps({'functional':functional,'complexity':complexity}))
'''


def evaluate_code(text,task):
    try:
        source=solution_source(text,task["function"])
        payload={"source":source,"function":task["function"],"cases":task["cases"],"allowed":{k:sorted(v) for k,v in ALLOWED.items()}}
        proc=subprocess.run([sys.executable,"-I","-c",CHILD],input=json.dumps(payload),text=True,capture_output=True,timeout=4)
        result=json.loads(proc.stdout) if proc.returncode==0 else {"functional":[],"complexity":[],"error":proc.stderr[-300:]}
        result["passed"]=len(result["functional"])==len(task["cases"]) and all(result["functional"]) and all(x["passed"] for x in result["complexity"])
        if task["function"]=="lower_bound":
            result["passed"] &= len(result["complexity"])==15
        result["normalized_tested_source"]=source
        return result
    except (ValueError,SyntaxError,subprocess.TimeoutExpired) as exc:
        return {"passed":False,"error":str(exc),"unsupported_or_unexecutable":True}
