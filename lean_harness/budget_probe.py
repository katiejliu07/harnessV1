"""Simulate a larger output budget on rows that hit the current one.

Greedy decoding makes this a narrow, checkable change: a larger budget leaves
every output that ended before the old cap token-for-token identical, so only
truncated rows can change. This reruns exactly those rows with the larger
budget, asserts the old tokens are a prefix of the new ones (proving the
property on real data), and regrades. Unused budget costs nothing; the extra
spend is only the tokens these rows now generate.
"""
import argparse
import json

import torch

from independent_audit.model import ROOT,load
from .backtoback import run_condition
from .benchmark import SEED,grade,write
from .differential import differential


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("generations")
    ap.add_argument("--out",required=True)
    ap.add_argument("--scalar-budget",type=int,default=1024)
    ap.add_argument("--code-budget",type=int,default=2048)
    ap.add_argument("--device",default="mps")
    ap.add_argument("--dtype",default="float16")
    args=ap.parse_args()
    rows=[json.loads(s) for s in (ROOT/args.generations).read_text().splitlines()]
    truncated=[r for r in rows if r["truncated"]]
    budgets={"scalar":args.scalar_budget,"code":args.code_budget}
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model,tok,_=load(args.device,args.dtype)
    results=[]
    for r in truncated:
        result=run_condition(model,tok,r["task"],r["profile"],budgets)
        if result["token_ids"][:len(r["token_ids"])]!=r["token_ids"]:
            raise RuntimeError(f"Old output is not a prefix of the new one: {r['task']['id']} {r['profile']}")
        row={**result,"task":r["task"],"profile":r["profile"],**grade(result,r["task"])}
        row["differential"]=differential({**row,"stage":"budget"})
        results.append({"task_id":r["task"]["id"],"profile":r["profile"],"old_tokens":len(r["token_ids"]),
            "new_tokens":result["generated_tokens_including_eos"],"still_truncated":result["truncated"],
            "now_correct":row["checks_and_eos"],"text":result["text"]})
        print(f"{r['task']['id']} {r['profile']} {len(r['token_ids'])}->{result['generated_tokens_including_eos']} truncated={result['truncated']} correct={row['checks_and_eos']}",flush=True)
    by_profile={}
    for x in results:
        e=by_profile.setdefault(x["profile"],{"truncated_rows":0,"fixed":0,"still_truncated":0,"extra_tokens":0})
        e["truncated_rows"]+=1
        e["fixed"]+=x["now_correct"]
        e["still_truncated"]+=x["still_truncated"]
        e["extra_tokens"]+=x["new_tokens"]-x["old_tokens"]
    write(ROOT/args.out,{"budgets":budgets,"source":args.generations,"by_profile":by_profile,"rows":results})
    print(json.dumps(by_profile,indent=2),flush=True)


if __name__=="__main__":
    main()
