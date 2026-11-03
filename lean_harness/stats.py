"""Uncertainty for the gates' observed counts (reporting only; gates unchanged).

Follows Miller, "Adding Error Bars to Evals" (arXiv 2411.00640): compare
profiles on question-level paired outcomes, not two summary means, and say how
large an effect the sample could miss. The promotion gate stays stricter than
any significance test: zero item losses against legacy.
"""
from math import comb,sqrt


def wilson(k,n,z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n==0:
        return (0.,1.)
    p=k/n
    centre=(p+z*z/(2*n))/(1+z*z/n)
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return (max(0.,centre-half),min(1.,centre+half))


def mcnemar_exact(wins,losses):
    """Two-sided exact McNemar p-value from discordant pairs."""
    n=wins+losses
    if n==0:
        return 1.
    tail=sum(comb(n,i) for i in range(min(wins,losses)+1))/2**n
    return min(1.,2*tail)


def undetectable_loss_rate(n,chance=.05):
    """Largest per-item loss rate that still shows zero losses on n items with
    probability >= chance: (1-r)^n >= chance. With n=24, about 11.7%."""
    return 1-chance**(1/n) if n else 1.


def paired(baseline,candidate,key="checks_and_eos"):
    by_id={r["task"]["id"]:r for r in baseline}
    pairs=[(by_id[r["task"]["id"]][key],r[key]) for r in candidate if r["task"]["id"] in by_id]
    wins=sum(1 for b,c in pairs if c and not b)
    losses=sum(1 for b,c in pairs if b and not c)
    n=len(pairs)
    kb,kc=sum(b for b,_ in pairs),sum(c for _,c in pairs)
    return {"n":n,"baseline_correct":kb,"candidate_correct":kc,
            "baseline_wilson95":wilson(kb,n),"candidate_wilson95":wilson(kc,n),
            "candidate_wins":wins,"candidate_losses":losses,"mcnemar_exact_p":mcnemar_exact(wins,losses),
            "loss_rate_a_zero_loss_result_cannot_rule_out_95":undetectable_loss_rate(n)}
