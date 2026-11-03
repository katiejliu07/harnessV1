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

PROFILES=("legacy","nosys","terse","steps")
# v1/v2 (results/token_light_v1, v2) showed that removing the worked steps
# ("Answer only", "Reply only: Final answer") or the code-block contract loses
# arithmetic accuracy and can lengthen code. v3 candidates keep both and target
# what measured outputs actually spend tokens on:
#   nosys: legacy wording without Qwen's injected default system turn (21 input
#          tokens on every request; the user's text is unchanged).
#   terse: nosys plus working as bare equations (legacy scalar output is mostly
#          numbered-plan prose, LaTeX \text{} and PEMDAS narration) and code without
#          docstrings/comments/tests (40-70% of legacy code output) unless asked.
#   steps: v3 terse skipped the working on arithmetic and lost items, matching
#          Chain of Draft's drop for this very model (Qwen2.5-1.5B, GSM8K 32.5% ->
#          24.2%; arXiv 2502.18600) and the finding that weaker models need longer
#          reasoning (Wu et al., ICLR 2026). steps requires every calculation as an
#          equation and bans only the measured waste. Code reuses terse's wording.
WRAPPERS={
    "legacy":{"scalar":CONCISE,"code":CODE_CONCISE},
    "nosys":{"scalar":CONCISE,"code":CODE_CONCISE},
    "terse":{"scalar":"Solve with brief equations only, no prose. End with: Final answer: <number>.\n\nQuestion: {prompt}",
             "code":"Provide only the complete Python function in a code block, without docstrings, comments, tests or explanation unless the question asks for them.\n\nQuestion: {prompt}"},
}
WRAPPERS["steps"]={"scalar":"Show every calculation as a short equation, one per line. Do not restate the question or add headings. End with: Final answer: <number>.\n\nQuestion: {prompt}",
                   "code":WRAPPERS["terse"]["code"]}
SYSTEM_FREE={"nosys","terse","steps"}
EMPTY_SYSTEM_TURN="<|im_start|>system\n<|im_end|>\n"
POLICY_PATH=ROOT/"lean_harness/policy.json"


def make_prompt(task,kind,profile):
    if not isinstance(task,str) or not task.strip():
        raise ValueError("A nonempty task is required")
    if profile not in PROFILES or kind not in {"scalar","code","general"}:
        raise ValueError("Unknown task kind or prompt profile")
    # General requests preserve explicit explanation, formatting and caveats.
    # The caller opts into scalar/code-only contracts; no unreliable classifier.
    return task if kind=="general" else WRAPPERS[profile][kind].format(prompt=task)


def render(tokenizer,prompt,profile):
    """Chat-template token ids. Without a system message Qwen's template inserts
    its default one; an explicit empty system turn still costs 5 tokens. System-
    free profiles render with an empty system turn and cut exactly that turn, so
    the rest is byte-identical to the template's own user/assistant framing."""
    messages=[{"role":"user","content":prompt}]
    if profile not in SYSTEM_FREE:
        return tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt")
    text=tokenizer.apply_chat_template([{"role":"system","content":""},*messages],tokenize=False,add_generation_prompt=True)
    if not text.startswith(EMPTY_SYSTEM_TURN):
        raise ValueError("Unexpected chat template; refusing to guess a system-free rendering")
    return tokenizer(text[len(EMPTY_SYSTEM_TURN):],add_special_tokens=False,return_tensors="pt")["input_ids"]


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
    return entry["profile"] if entry.get("promoted") and entry.get("profile") in PROFILES else "legacy"


def infer(model,tokenizer,task,kind,profile="legacy",max_new_tokens=None):
    # 1024 everywhere: hard_backtoback_v1 + budget_probe_1024 showed 320 cut off 9 of 24 hard
    # problems under legacy; 1024 let 7 finish correctly, and every finished output was
    # unchanged (greedy prefix check). Unused budget costs nothing.
    budget=max_new_tokens if max_new_tokens is not None else 1024
    if budget<=0:
        raise ValueError("Output budget must be positive")
    synchronize(model.cfg.device)
    start=time.perf_counter()
    prompt=make_prompt(task,kind,profile)
    # Unpatched inference needs no donor, offset maps or second tokenization.
    ids=render(tokenizer,prompt,profile).to(model.cfg.device)
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
