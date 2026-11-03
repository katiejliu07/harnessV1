"""Check published model hash, saved JSON arithmetic and text tokenization limits."""
import hashlib
import json
from collections import defaultdict

from transformers import AutoTokenizer

from .model import ROOT,MODEL_DIR,REVISION
from .saved_audit import original_function


def main():
    out=ROOT/"independent_audit/results"
    remote=json.loads((out/"model_remote_manifest.json").read_text())
    model_file=next(x for x in remote["siblings"] if x["rfilename"]=="model.safetensors")
    h=hashlib.sha256()
    with (MODEL_DIR/"model.safetensors").open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    actual=h.hexdigest()
    expected=model_file["lfs"]["sha256"]
    if actual!=expected or remote["sha"]!=REVISION:
        raise RuntimeError("Model checksum/revision mismatch")
    tok=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    record=json.loads((ROOT/"results/activation_patching/quality_assessment_v8.json").read_text())
    rows=record["detailed_results"]
    discrepancies=[]
    retoken=[]
    markers=original_function("count_reasoning_markers")
    for i,row in enumerate(rows):
        for field,a,b in [("token_reduction",row["baseline"]["tokens"],row["intervened"]["tokens"]),
                          ("marker_reduction",markers(row["baseline"]["text"]),markers(row["intervened"]["text"]))]:
            if row["changes"][field]!=a-b:
                discrepancies.append({"index":i,"field":field,"saved":row["changes"][field],"recalculated":a-b})
        for side in ["baseline","intervened"]:
            item=row[side]
            nt=len(tok.encode(item["text"],add_special_tokens=False))
            retoken.append({"index":i,"side":side,"saved_tokens":item["tokens"],"decoded_text_retokenized":nt,
                            "saved_minus_retokenized":item["tokens"]-nt})
    groups=defaultdict(list)
    for path in (ROOT/"results").rglob("*.json"):
        groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(str(path.relative_to(ROOT)))
    result={"model_sha256_actual":actual,"model_sha256_published":expected,"model_revision":REVISION,
            "v8_change_field_discrepancies":discrepancies,"v8_text_retokenization":retoken,
            "exact_duplicate_json_artifact_groups":[v for v in groups.values() if len(v)>1],
            "limits":"Decoded/stripped text omits EOS and may lose leading whitespace. Retokenization cannot prove exact historical generated token IDs; new runs save IDs including EOS."}
    (out/"artifact_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="v8_text_retokenization"},indent=2))


if __name__=="__main__":
    main()
