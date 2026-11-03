"""Token-level alignments recorded in every result; no inferred semantic matches."""
from __future__ import annotations

REASONING = ("Think step-by-step to solve this problem. Show your reasoning process.\n"
             "End with: Final answer: <number>.\n\nQuestion: {prompt}\n\n"
             "Let me think through this step by step:")
CONCISE = "Give a concise, complete answer. End with: Final answer: <number>.\n\nQuestion: {prompt}"
CODE_REASONING = "Solve this programming task step by step. Then provide the complete Python function in a code block.\n\nQuestion: {prompt}\n\nLet me think through this step by step:"
CODE_CONCISE = "Provide only the complete Python function in a code block.\n\nQuestion: {prompt}"


def tokenize(tokenizer, prompt, question):
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    start = rendered.index(prompt)
    qstart = start + prompt.index(question)
    qend = qstart + len(question)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    # Fully contained tokens avoid copying a token spanning question/instruction.
    q = [i for i,(a,b) in enumerate(offsets) if qstart <= a < b <= qend]
    inst = [i for i,(a,b) in enumerate(offsets) if start <= a < b <= start+len(prompt) and (b <= qstart or a >= qend)]
    ids = encoded["input_ids"]
    reference = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
    if ids.tolist() != reference.tolist():
        raise ValueError("Offset tokenizer differs from original chat template")
    return {"ids": ids, "question": q, "instruction": inst, "rendered": rendered}


def position_map(target, donor, mode):
    nt, nd = target["ids"].shape[1], donor["ids"].shape[1]
    if mode == "prefix":
        return list(range(min(nt,nd))), list(range(min(nt,nd)))
    if mode in {"question", "instruction"}:
        a, b = target[mode], donor[mode]
        n = min(len(a),len(b))
        return (a[-n:], b[-n:]) if n else ([],[])
    if mode == "suffix":
        a,b = target["ids"][0].tolist(),donor["ids"][0].tolist()
        n = 0
        while n < min(nt,nd) and a[-n-1] == b[-n-1]:
            n += 1
        return list(range(nt-n,nt)),list(range(nd-n,nd))
    raise ValueError(mode)


def equal_length_concise(tokenizer, question, target_len, coding=False):
    base = (CODE_CONCISE if coding else CONCISE).format(prompt=question)
    # Padding is a separate semantic control, never assumed neutral. Search exact
    # token length after rendering; record text instead of padding with masked IDs.
    for count in range(target_len+1):
        candidate = base + "\n" + " " .join(["Please"]*count)
        item = tokenize(tokenizer,candidate,question)
        if item["ids"].shape[1] == target_len:
            return item
    return None
