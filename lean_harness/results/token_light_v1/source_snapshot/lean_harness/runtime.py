"""Small prompts with complete task text and the audited KV-cached generator.

Critical: never shorten the user's task, delete code/comments after generation,
or stop at a syntactically valid partial function. Such edits can hide lost
requirements and do not save generation tokens already spent. Decode to EOS,
report budget termination explicitly, and preserve the raw output for review.
"""
from pathlib import Path
import hashlib
import json
import re
import time

from independent_audit.engine import generate,synchronize
from independent_audit.model import ROOT,REVISION
from independent_audit.prompts import CONCISE,CODE_CONCISE

PROFILES=("legacy","lean","checked")
WRAPPERS={
    "legacy":{"scalar":CONCISE,"code":CODE_CONCISE},
    "lean":{"scalar":"{prompt}\nAnswer only.","code":"{prompt}\nPython only."},
    "checked":{"scalar":"Calculate carefully. Return only the result.\n{prompt}",
               "code":"Complete Python only; preserve all requirements.\n{prompt}"},
}
POLICY_PATH=ROOT/"lean_harness/policy.json"


def make_prompt(task,kind,profile):
    if not isinstance(task,str) or not task.strip():
        raise ValueError("A nonempty task is required")
    if profile not in PROFILES or kind not in {"scalar","code","general"}:
        raise ValueError("Unknown task kind or prompt profile")
    # General requests preserve explicit explanation, formatting and caveats.
    # The caller opts into scalar/code-only contracts; no unreliable classifier.
    return task if kind=="general" else WRAPPERS[profile][kind].format(prompt=task)


def recommended_profile(kind,path=POLICY_PATH,*,environment=None):
    if not path.exists() or kind=="general":
        return "legacy"
    policy=json.loads(path.read_text())
    entry=policy["kinds"].get(kind,{})
    # Failed development/validation must never quietly turn into a new default.
    # Promotion is specific to measured source/model/backend, not a guarantee on
    # a different GPU or after changing the prompt/decoding implementation.
    if not environment or any(policy["environment"][k]!=environment[k] for k in ["device","dtype","model_revision"]):
        return "legacy"
    if environment["model_revision"]!=REVISION:
        return "legacy"
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha for name,sha in policy["inference_source_sha256"].items()):
        return "legacy"
    return entry["profile"] if entry.get("promoted") else "legacy"


def infer(model,tokenizer,task,kind,profile="legacy",max_new_tokens=None):
    budget=max_new_tokens if max_new_tokens is not None else 1024 if kind=="code" else 320
    if budget<=0:
        raise ValueError("Output budget must be positive")
    synchronize(model.cfg.device)
    start=time.perf_counter()
    prompt=make_prompt(task,kind,profile)
    # Unpatched inference needs no donor, offset maps or second tokenization.
    ids=tokenizer.apply_chat_template([{"role":"user","content":prompt}],
        add_generation_prompt=True,return_tensors="pt").to(model.cfg.device)
    result=generate(model,ids,budget,tokenizer.eos_token_id)
    result.pop("last_logits")
    text=tokenizer.decode(result["token_ids"],skip_special_tokens=True).strip()
    synchronize(model.cfg.device)
    elapsed=time.perf_counter()-start
    return {**result,"text":text,"prompt":prompt,"input_ids":ids[0].tolist(),
        "input_tokens":ids.shape[1],"total_tokens":ids.shape[1]+len(result["token_ids"]),
        "total_seconds":elapsed,"kind":kind,"profile":profile,"max_new_tokens":budget,
        "status":"truncated" if result["truncated"] else "empty" if not text else "complete_unverified",
        "answer_only_format":bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?\.?",text)) if kind=="scalar" else None}
