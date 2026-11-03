from independent_audit.benchmark import choose_donor


def test_distinct_donors_even_when_neighbors_share_gold_answer():
    tasks=[{"prompt":str(i),"answer":a} for i,a in enumerate(["12","12","24","25","7"])]
    a=choose_donor(tasks,0,"E_v8_prefix","different_1")
    b=choose_donor(tasks,0,"E_v8_prefix","different_2")
    assert a["prompt"]!=b["prompt"]
    assert a["answer"]!=tasks[0]["answer"] and b["answer"]!=tasks[0]["answer"]
    unrelated=[choose_donor(tasks,0,"G_unrelated",v)["prompt"] for v in ["same","different_1","different_2"]]
    assert len(set(unrelated))==3
