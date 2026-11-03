"""Audit non-quality JSON layouts; explicitly mark missing response data."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .evaluation import correct,extract_answer
from .model import ROOT


def strict_json(text):
    def no_duplicates(pairs):
        result={}
        for k,v in pairs:
            if k in result:
                raise ValueError(f"Duplicate JSON key {k}")
            result[k]=v
        return result
    return json.loads(text,object_pairs_hook=no_duplicates,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {x}")))


def nodes(obj,path="root"):
    if isinstance(obj,dict):
        yield path,obj
        for k,v in obj.items():
            yield from nodes(v,f"{path}.{k}")
    elif isinstance(obj,list):
        for i,v in enumerate(obj):
            yield from nodes(v,f"{path}[{i}]")


def main():
    folder=ROOT/"results/activation_patching"
    parsed=[]
    errors=[]
    deltas=[]
    grouped=defaultdict(list)
    unscorable=[]
    for path in sorted(folder.glob("*.json")):
        try:
            data=strict_json(path.read_text(encoding="utf-8-sig"))
            parsed.append(path.name)
        except (ValueError,UnicodeError) as exc:
            errors.append({"file":path.name,"error":str(exc)})
            continue
        for loc,row in nodes(data):
            baseline=row.get("reasoning_baseline_tokens",row.get("baseline_tokens"))
            for field in ["layer_results","head_results"]:
                if isinstance(baseline,(int,float)) and isinstance(row.get(field),dict):
                    if not any(k in row for k in ["text","response","output"]):
                        unscorable.append({"file":path.name,"path":loc,"reason":"Only token counts saved; correctness cannot be recovered"})
                    for index,item in row[field].items():
                        if not isinstance(item,dict) or "tokens_generated" not in item:
                            continue
                        computed=item["tokens_generated"]-baseline
                        deltas.append({"file":path.name,"path":loc,"component":index,
                                       "recalculated_change":computed,"saved_change":item.get("token_change"),
                                       "consistent":computed==item.get("token_change")})
                        grouped[(path.name,field,index)].append(computed)
    probe=json.loads((folder/"generation_phase_probe.json").read_text())
    probe_audit={"n":len(probe["examples"]),
      "whitespace_only_intervened_previews":sum(not e["conditions"]["intervened"]["output_preview"].strip() for e in probe["examples"]),
      "missing_answer_positions":{side:sum(e["conditions"][side]["answer_pos"] is None for e in probe["examples"]) for side in ["baseline","intervened"]},
      "readme_rank_claim_source":{side:probe["summary"][side]["logit_lens_end"][-1] for side in ["baseline","intervened"]},
      "interpretation":"Generation-phase hooks differ from v8 and overwrite decode-local zero under cached generation. All saved intervened previews are degenerate; rank changes do not validate v8 reasoning suppression."}
    validation=json.loads((folder/"validation_results.json").read_text())
    gold=["8","5","3","15","30"]
    validation_scores=[{"prompt":row["prompt"],"expected":expected,
                        "baseline_correct":correct(row["baseline"],expected),"reasoning_correct":correct(row["reasoning"],expected),
                        "token_ratio_recalculated":row["reasoning_tokens"]/row["baseline_tokens"],
                        "token_ratio_saved":row["token_ratio"]}
                       for row,expected in zip(validation["results"],gold)]
    result={"strictly_parsed_json_files":parsed,"json_errors":errors,
      "replayed_token_changes":deltas,"token_delta_inconsistencies":sum(not r["consistent"] for r in deltas),
      "mean_token_changes_by_component":[{"file":k[0],"field":k[1],"component":k[2],"n":len(v),"mean":statistics.mean(v)} for k,v in grouped.items()],
      "unrecoverable_correctness_records":unscorable,"generation_phase_probe":probe_audit,
      "setup_validation_rescoring":validation_scores,
      "limits":"Attention/logit summaries cannot be independently recomputed from aggregate JSON without original cached activations and exact historical environment. Main quality arrays are separately recalculated."}
    (ROOT/"independent_audit/results/auxiliary_artifact_audit.json").write_text(json.dumps(result,indent=2))
    print({"json_files":len(parsed),"json_errors":errors,"token_delta_fields":len(deltas),"inconsistent_deltas":result["token_delta_inconsistencies"],"generation_probe":probe_audit})


if __name__=="__main__":
    main()
