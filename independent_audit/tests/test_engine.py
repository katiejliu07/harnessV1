import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from independent_audit.engine import Patch, capture, generate, hooks_for
from independent_audit.prompts import position_map


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(42)
    return HookedTransformer(HookedTransformerConfig(n_layers=8, d_model=32, n_ctx=64,
        d_head=8,n_heads=4,d_mlp=64,d_vocab=101,act_fn="silu",device="cpu")).eval()


def test_cached_noop_and_alpha_zero_parity(model):
    ids = torch.tensor([[1,4,7,9,12]])
    donor = capture(model,torch.tensor([[1,8,3]]),[2,3,6])
    kw = dict(max_new_tokens=8,eos_token_id=-1)
    a = generate(model,ids,**kw)
    for b in [generate(model,ids,noop_layers=[2,3,6],**kw),
              generate(model,ids,patch=Patch(donor,[0,1,2],[0,1,2],0),**kw)]:
        assert a["token_ids"] == b["token_ids"]
        torch.testing.assert_close(a["last_logits"],b["last_logits"],rtol=0,atol=0)


def test_cached_patch_matches_full_recompute(model):
    ids=torch.tensor([[1,4,7,9,12]])
    donor=capture(model,torch.tensor([[1,8,3]]),[2,3,6])
    patch=Patch(donor,[0,1,2],[0,1,2])
    a=generate(model,ids,8,-1,patch=patch,cached=True)
    b=generate(model,ids,8,-1,patch=patch,cached=False)
    assert a["token_ids"] == b["token_ids"]
    torch.testing.assert_close(a["last_logits"],b["last_logits"],rtol=1e-4,atol=1e-5)
    assert a["forward_query_tokens"] == 5+7
    assert b["forward_query_tokens"] == sum(range(5,13))


def test_no_patch_into_local_decode_zero():
    patch=Patch({2:torch.ones(1,4,8)},[0,1],[0,1])
    hook=hooks_for(patch,{"prefill":False})[0][1]
    x=torch.zeros(1,1,8)
    assert torch.equal(hook(x,None),x)


def test_absolute_prefix_transplants_instruction_and_question():
    target={"ids":torch.tensor([[1,2,3,4,5,6,7]]),"instruction":[1,2,3],"question":[4,5]}
    donor={"ids":torch.tensor([[1,8,9,7]]),"instruction":[],"question":[1,2]}
    ti,di=position_map(target,donor,"prefix")
    assert 1 in ti and 1 in di  # donor question -> target instruction
    ti,di=position_map(target,donor,"question")
    assert ti == [4,5] and di == [1,2]
    assert not set(ti).intersection(target["instruction"])
    assert position_map(target,donor,"suffix") == ([6],[3])


def test_reject_unsupported_schedule_and_invalid_maps(model):
    with pytest.raises(ValueError,match="KV invalidation"):
        generate(model,torch.tensor([[1,2]]),2,-1,max_patch_steps=10)
    with pytest.raises(ValueError):
        Patch({},[1,1],[0,1])


def test_changing_donor_changes_activations(model):
    ids=torch.tensor([[1,4,7,9,12]])
    d1=capture(model,torch.tensor([[1,8,3]]),[2])
    d2=capture(model,torch.tensor([[1,9,4]]),[2])
    a=generate(model,ids,1,-1,patch=Patch(d1,[0,1,2],[0,1,2]))
    b=generate(model,ids,1,-1,patch=Patch(d2,[0,1,2],[0,1,2]))
    assert not torch.equal(a["last_logits"],b["last_logits"])
    # Different logits alone do not prove answer leakage; real-model donor-answer
    # attraction is measured separately, with target and donor gold answers saved.


def test_original_generation_probe_repatches_decode_local_zero():
    import ast
    from independent_audit.saved_audit import ROOT
    tree=ast.parse((ROOT/"code/generation_phase_analysis.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="patch_hook")
    ns={}
    exec(compile(ast.Module([fn],type_ignores=[]),"original_generation_probe","exec"),ns)
    old=ns["patch_hook"](2,torch.ones(1,4,8))
    assert torch.equal(old(torch.zeros(1,1,8),None),torch.ones(1,1,8))
    replay=old(torch.zeros(1,6,8),None)
    assert torch.equal(replay[:,-1,:],torch.zeros(1,8))
    # Offline full-sequence replay leaves generated positions alone, unlike
    # the online cached hook. Such replay is not a faithful activation trace.
    # With the pinned library's default KV-cached generate(), the first token of
    # the donor is injected into EVERY generated-token residual in that probe.


def test_early_donor_capture_exactly_matches_full(model):
    from independent_audit.optimization import capture_early
    ids=torch.tensor([[1,7,8,3]])
    full=capture(model,ids,[6,2,3])
    early=capture_early(model,ids,[6,2,3])
    assert all(torch.equal(full[i],early[i]) for i in [6,2,3])


def test_original_answer_locator_misses_multitoken_numbers():
    import ast
    import re
    from independent_audit.saved_audit import ROOT
    tree=ast.parse((ROOT/"code/generation_phase_analysis.py").read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"extract_digits","locate_positions"}]
    ns={"re":re}
    exec(compile(ast.Module(nodes,type_ignores=[]),"original_answer_locator","exec"),ns)
    final,answer=ns["locate_positions"](["Final"," answer",":"," ","1","2","."],0,"12")
    assert final==0 and answer is None


def test_saved_generation_probe_degenerate_outputs():
    import json
    from independent_audit.saved_audit import ROOT
    d=json.loads((ROOT/"results/activation_patching/generation_phase_probe.json").read_text())
    assert len(d["examples"])==25
    assert all(not e["conditions"]["intervened"]["output_preview"].strip() for e in d["examples"])


def test_rotary_grouped_query_cached_patch_parity():
    torch.manual_seed(123)
    cfg=HookedTransformerConfig(n_layers=8,d_model=32,n_ctx=64,d_head=8,n_heads=4,
        n_key_value_heads=2,d_mlp=64,d_vocab=101,act_fn="silu",device="cpu",
        positional_embedding_type="rotary",rotary_dim=8,rotary_base=1000000,
        normalization_type="RMS",gated_mlp=True)
    model=HookedTransformer(cfg).eval()
    ids=torch.tensor([[1,4,7,9,12]])
    donor=capture(model,torch.tensor([[1,8,3]]),[2,3,6])
    patch=Patch(donor,[0,1,2],[0,1,2])
    cached=generate(model,ids,8,-1,patch=patch)
    full=generate(model,ids,8,-1,patch=patch,cached=False)
    assert cached["token_ids"]==full["token_ids"]
    torch.testing.assert_close(cached["last_logits"],full["last_logits"],rtol=1e-4,atol=1e-5)
