"""Post-generation quality checks, explicitly separate from frozen raw scores.

Passing scalar tests does not establish instruction adherence or sound reasoning.
These diagnostics preserve every raw score and label their scope. Complexity is
checked with counted sequence accesses, not flaky wall-clock cutoffs.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import hashlib

from .model import ROOT
from .tasks import score_code
from .code_quality import evaluate_code
from . import scoring_v2


def final_format(text):
    return bool(re.search(r"Final answer:\s*[-+]?\d+(?:\.\d+)?\.?\s*$",text.strip().replace("**", ""),re.I))


def logarithmic_access_test(text,task):
    base=score_code(text,task)
    if not base["passed"]:
        return {"passed":False,"reason":"Basic executable tests failed","basic":base}
    if task["function"]!="lower_bound":
        return {"passed":True,"reason":"No asymptotic constraint specified for this task","basic":base}
    blocks=re.findall(r"```(?:python)?\s*\n(.*?)```",text,re.S)
    source=blocks[-1] if blocks else text
    child=r'''
import json,sys,resource,math,bisect
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
# macOS rejects this limit; retaining it would falsely fail every code sample.
if sys.platform!='darwin':
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
data=json.load(sys.stdin)
safe={k:__builtins__.__dict__[k] for k in ['len','range','enumerate','zip','list','tuple','set','dict','min','max','sum','abs','sorted','reversed','all','any','isinstance','int','str','bool','float','ValueError']}
ns={'__builtins__':safe}
exec(compile(data['source'],'<generated>','exec'),ns)
class CountedList(list):
    def charge(self,n):
        self.accesses+=n
        if self.accesses>self.limit: raise RuntimeError('Nonlogarithmic sequence access budget exceeded')
    def __getitem__(self,key):
        self.charge(len(range(*key.indices(len(self)))) if isinstance(key,slice) else 1)
        return super().__getitem__(key)
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    def index(self,*args):
        self.charge(len(self))
        return super().index(*args)
    def copy(self):
        self.charge(len(self))
        return super().copy()
results=[]
for n in [256,4096,65536]:
    raw=[i//3 for i in range(n)]
    for x in [-1,0,n//6,raw[-1],raw[-1]+1]:
        a=CountedList(raw)
        a.accesses=0
        a.limit=2*math.ceil(math.log2(n+1))+8
        try:
            answer=ns['lower_bound'](a,x)
            passed=answer==bisect.bisect_left(raw,x)
            results.append({'n':n,'x':x,'accesses':a.accesses,'budget':a.limit,'passed':passed})
        except Exception as e:
            results.append({'n':n,'x':x,'accesses':a.accesses,'budget':a.limit,'passed':False,'error':str(e)})
print(json.dumps(results))
'''
    try:
        run=subprocess.run([sys.executable,"-I","-c",child],input=json.dumps({"source":source}),text=True,capture_output=True,timeout=4)
        details=json.loads(run.stdout) if run.returncode==0 else []
        return {"passed":len(details)==15 and all(r["passed"] for r in details),"probes":details,
                "error":run.stderr[-300:] or None,"basic":base,
                "scope":"Counted sequence-access bound on tested sizes; not a universal proof of asymptotic complexity."}
    except (ValueError,subprocess.TimeoutExpired) as exc:
        return {"passed":False,"error":str(exc),"basic":base}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("directory",type=Path)
    args=ap.parse_args()
    raw=(args.directory/"benchmark.jsonl").read_bytes()
    rows=[json.loads(s) for s in raw.decode().splitlines()]
    reports=[]
    for row in rows:
        if row["status"]!="ok":
            continue
        coding="function" in row["task"]
        result={k:row[k] for k in ["suite","task_index","condition","repeat","donor_variant"]}
        result["raw_primary_correct"]=row["correct"]
        if coding:
            result["executable_and_complexity"]=evaluate_code(row["text"],row["task"])
            blocks=re.findall(r"```(?:python)?\s*\n.*?```",row["text"],re.S)
            result["has_requested_code_fence"]=bool(blocks)
            outside=re.sub(r"```(?:python)?\s*\n.*?```","",row["text"],flags=re.S).strip()
            result["only_code_if_concise_requested"]=(not outside) if row["condition"] in {"C_concise","C_length_matched"} else None
        else:
            result["scalar_evaluation_v2"]={"answer":scoring_v2.extract_answer(row["text"]),
                                            "passed":scoring_v2.correct(row["text"],row["task"]["answer"])}
            result["requested_final_format"]=final_format(row["text"]) if row["condition"]!="A_normal" else None
            result["known_false_intermediate_statement"]="90 is less than 81" in row["text"]
            result["reasoning_soundness"]="Not established by scalar scoring or marker counts; requires adjudication"
        reports.append(result)
    serialized=json.dumps(reports,indent=2)
    (args.directory/"quality_diagnostics.json").write_text(serialized)
    manifest={"input_sha256":hashlib.sha256(raw).hexdigest(),"diagnostics_sha256":hashlib.sha256(serialized.encode()).hexdigest(),
              "graded_rows":len(reports),"source_sha256":{name:hashlib.sha256((ROOT/"independent_audit"/name).read_bytes()).hexdigest()
                    for name in ["quality_review.py","code_quality.py","scoring_v2.py","evaluation.py","tasks.py"]}}
    (args.directory/"grading_manifest.json").write_text(json.dumps(manifest,indent=2))
    print({"rows":len(reports),"known_false_intermediate_statements":sum(r.get("known_false_intermediate_statement",False) for r in reports),
           "final_format_failures":sum(r.get("requested_final_format") is False for r in reports),
           "code_or_complexity_failures":sum(not r["executable_and_complexity"]["passed"] for r in reports if "executable_and_complexity" in r)})


if __name__=="__main__":
    main()
