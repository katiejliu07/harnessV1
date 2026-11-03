import torch
from independent_audit.prefix_partitions import partition_map


def test_original_prefix_partition_does_not_change_donor_or_alignment():
    target={"ids":torch.tensor([[1,2,3,4,5,6,7]]),"instruction":[1,2,3],"question":[4,5]}
    donor={"ids":torch.tensor([[1,8,9,7]]),"instruction":[],"question":[1,2]}
    parts=partition_map(target,donor)
    assert parts["instruction"]==([1,2,3],[1,2,3])
    assert parts["question"]==([],[])
    assert parts["chat_or_boundary"]==([0],[0])
    assert sorted(t for pair in parts.values() for t in pair[0])==list(range(4))
