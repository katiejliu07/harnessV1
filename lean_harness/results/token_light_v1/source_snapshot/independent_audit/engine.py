"""Same-path, KV-cached residual-post interventions with explicit position maps.

Critical invariant: prompt-only patches must NEVER be reapplied to decode position
zero. A local decode index zero is not absolute prompt position zero. For a
time-limited patch, stale patched KV states are not equivalent to unpatching;
this engine intentionally rejects that schedule instead of silently changing it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


@dataclass
class Patch:
    donor: dict
    target_positions: list[int]
    donor_positions: list[int]
    alpha: float = 1.

    def __post_init__(self):
        if len(self.target_positions) != len(self.donor_positions):
            raise ValueError("Position maps must have equal cardinality")
        if len(set(self.target_positions)) != len(self.target_positions):
            raise ValueError("Duplicate target positions")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0,1]")


def capture(model, ids, layers):
    states = {}
    def save(activation, hook):
        states[int(hook.name.split(".")[1])] = activation.detach().clone()
        return activation
    with torch.inference_mode():
        model.run_with_hooks(ids, return_type=None,
                             fwd_hooks=[(f"blocks.{i}.hook_resid_post", save) for i in layers])
    return states


def hooks_for(patch, phase, noop_layers=()):
    if patch is None:
        return [(f"blocks.{i}.hook_resid_post", lambda x, hook: x) for i in noop_layers]
    def hook_for(layer):
        def apply(x, hook):
            if not phase["prefill"] or patch.alpha == 0 or not patch.target_positions:
                return x
            ti = torch.tensor(patch.target_positions, device=x.device)
            di = torch.tensor(patch.donor_positions, device=x.device)
            if min(patch.target_positions) < 0 or max(patch.target_positions) >= x.shape[1]:
                raise ValueError("Target position outside prompt")
            donor = patch.donor[layer]
            if min(patch.donor_positions) < 0 or max(patch.donor_positions) >= donor.shape[1]:
                raise ValueError("Donor position outside donor prompt")
            # Out-of-place avoids mutating a captured/aliased activation in tests.
            y = x.clone()
            y[:, ti, :] = x[:, ti, :] * (1-patch.alpha) + donor[:, di, :] * patch.alpha
            return y
        return apply
    return [(f"blocks.{i}.hook_resid_post", hook_for(i)) for i in patch.donor]


def generate(model, ids, max_new_tokens, eos_token_id, *, patch=None, noop_layers=(), cached=True,
             max_patch_steps=None):
    if cached and max_patch_steps is not None:
        raise ValueError("Time-limited patch requires KV invalidation; use uncached reproduction")
    phase = {"prefill": True}
    hooks = hooks_for(patch, phase, noop_layers)
    cache = HookedTransformerKeyValueCache.init_cache(model.cfg, ids.device, batch_size=1) if cached else None
    tokens = ids.clone()
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    query_tokens = attention_pairs = 0
    synchronize(ids.device)
    start = time.perf_counter()
    first = None
    final_logits = None
    with torch.inference_mode():
        for step in range(max_new_tokens):
            phase["prefill"] = not cached or step == 0
            inp = tokens if not cached or step == 0 else tokens[:, -1:]
            qlen, klen = inp.shape[1], tokens.shape[1]
            query_tokens += qlen
            attention_pairs += qlen*klen
            active_hooks = hooks if max_patch_steps is None or step < max_patch_steps else []
            with model.hooks(fwd_hooks=active_hooks):
                # Batch-one benchmark has no padding. Explicit masks also allow
                # tiny tokenizer-free fixtures to exercise real KV semantics.
                logits = model(inp, past_kv_cache=cache, attention_mask=torch.ones_like(inp))
            final_logits = logits[:, -1, :]
            if not bool(final_logits.isfinite().all()):
                raise FloatingPointError("Nonfinite logits: reject run before scoring/timing claims")
            next_id = final_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_id], dim=1)
            if first is None:
                synchronize(ids.device)
                first = time.perf_counter()-start
            if int(next_id.item()) in eos_ids:
                break
    synchronize(ids.device)
    wall = time.perf_counter()-start
    new = tokens[0, ids.shape[1]:].tolist()
    return {"token_ids": new, "generated_tokens_including_eos": len(new),
            "stopped_on_eos": bool(new and new[-1] in eos_ids),
            "truncated": bool(new and new[-1] not in eos_ids and len(new) == max_new_tokens),
            "generation_seconds": wall, "ttft_seconds": first,
            "decode_seconds": wall-(first or 0), "forward_query_tokens": query_tokens,
            "attention_pairs_per_layer_proxy": attention_pairs,
            "last_logits": final_logits.detach().cpu() if final_logits is not None else None}
