"""Post-hoc audit of finished results: error bars and differential code tests.

No model calls. For each stage/kind/profile it reports accuracy with a 95%
Wilson interval, paired wins/losses against legacy with an exact McNemar
p-value, and the per-item loss rate a zero-loss result cannot rule out. Code
rows are regraded on EvalPlus-style differential cases; a profile that loses
any item to legacy under them is reported blocked.
"""
import argparse
from collections import defaultdict
import json

from independent_audit.model import ROOT
from .differential import differential
from .stats import paired,wilson


def audit(paths):
    rows=[json.loads(s) for p in paths for s in (ROOT/p).read_text().splitlines()]
    groups=defaultdict(list)
    for r in rows:
        groups[(r["stage"],r["task"]["kind"])].append(r)
    report=[]
    for (stage,kind),rs in sorted(groups.items()):
        by_profile=defaultdict(list)
        for r in rs:
            by_profile[r["profile"]].append(r)
        diff={p:[differential(r) for r in prs] for p,prs in by_profile.items()} if kind=="code" else {}
        for profile,prs in sorted(by_profile.items()):
            k=sum(r["checks_and_eos"] for r in prs)
            entry={"stage":stage,"kind":kind,"profile":profile,"n":len(prs),"correct":k,"wilson95":wilson(k,len(prs)),
                   "total_tokens":sum(r["total_tokens"] for r in prs)}
            if profile!="legacy" and "legacy" in by_profile:
                entry["vs_legacy"]=paired(by_profile["legacy"],prs)
            if kind=="code":
                d=[x for x in diff[profile] if x]
                entry["differential_passed"]=sum(x["differential_passed"] for x in d)
                entry["differential_n"]=len(d)
                if profile!="legacy" and "legacy" in diff:
                    legacy_ok={x["task_id"] for x in diff["legacy"] if x and x["differential_passed"]}
                    entry["differential_losses_vs_legacy"]=sorted(x["task_id"] for x in d if x["task_id"] in legacy_ok and not x["differential_passed"])
                    entry["blocked_by_differential"]=bool(entry["differential_losses_vs_legacy"])
            report.append(entry)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("generations",nargs="+",help="generations.jsonl paths relative to the repository root")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    report=audit(args.generations)
    (ROOT/args.out).write_text(json.dumps(report,indent=2))
    for e in report:
        lo,hi=e["wilson95"]
        line=f"{e['stage']:12} {e['kind']:6} {e['profile']:7} {e['correct']:2}/{e['n']:<2} [{lo:.2f},{hi:.2f}] tokens={e['total_tokens']}"
        if "vs_legacy" in e:
            v=e["vs_legacy"]
            line+=f" | vs legacy +{v['candidate_wins']}/-{v['candidate_losses']} p={v['mcnemar_exact_p']:.3f} unseen-loss<= {v['loss_rate_a_zero_loss_result_cannot_rule_out_95']:.0%}"
        if "differential_n" in e:
            line+=f" | differential {e['differential_passed']}/{e['differential_n']}"
            if e.get("blocked_by_differential"):
                line+=f" BLOCKED {e['differential_losses_vs_legacy']}"
        print(line)


if __name__=="__main__":
    main()
