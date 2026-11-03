"""Export actual tokenizer maps, without loading weights or running inference."""
import csv
import importlib.util
import json

from transformers import AutoTokenizer

from .model import MODEL_DIR,ROOT
from .prompts import REASONING,CODE_REASONING,tokenize,position_map
from .tasks import HELDOUT,CODING


def region(item,index):
    return "question" if index in item["question"] else "instruction" if index in item["instruction"] else "chat_or_boundary"


def main():
    spec=importlib.util.spec_from_file_location("original_problem_sets",ROOT/"code/problem_sets.py")
    original=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    out=ROOT/"independent_audit/results/alignment_maps"
    out.mkdir(exist_ok=True)
    counts=[]
    for suite,tasks in [("original",original.MAIN_PROBLEMS),("heldout_arithmetic",HELDOUT),("coding",CODING)]:
        for index,task in enumerate(tasks):
            q=task["prompt"]
            target=tokenize(tokenizer,(CODE_REASONING if "function" in task else REASONING).format(prompt=q),q)
            donor=tokenize(tokenizer,q,q)
            ti,di=position_map(target,donor,"prefix")
            rows=[]
            for t,d in zip(ti,di):
                target_id=int(target["ids"][0,t])
                donor_id=int(donor["ids"][0,d])
                rows.append({"target_position":t,"donor_position":d,"target_id":target_id,"donor_id":donor_id,
                    "target_token":tokenizer.convert_ids_to_tokens(target_id),"donor_token":tokenizer.convert_ids_to_tokens(donor_id),
                    "target_region":region(target,t),"donor_region":region(donor,d),"identical_id":target_id==donor_id})
            with (out/f"{suite}_{index:02}.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            counts.append({"suite":suite,"task_index":index,"prompt":q,"target_length":target["ids"].shape[1],
                           "donor_length":donor["ids"].shape[1],"patched_tokens":len(rows),
                           "target_instruction_tokens":sum(r["target_region"]=="instruction" for r in rows),
                           "target_question_tokens":sum(r["target_region"]=="question" for r in rows),
                           "donor_question_to_target_instruction":sum(r["donor_region"]=="question" and r["target_region"]=="instruction" for r in rows)})
    (out/"summary.json").write_text(json.dumps(counts,indent=2))
    print(json.dumps([r for r in counts if r["task_index"]==0],indent=2))


if __name__=="__main__":
    main()
