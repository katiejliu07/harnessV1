"""Retokenize every saved quality response, not just the selected v8 file.

Historical model/tokenizer revisions and token IDs are unknown. Differences from
this pinned tokenizer are observations, not automatically count bugs. Duplicate
artifacts remain separate provenance records, never independent replications.
Only saved full quality responses are scored; truncated probe previews cannot
substitute for missing generated sequences.
"""
from collections import Counter
import hashlib
import json

from transformers import AutoTokenizer

from .model import ROOT,MODEL_DIR,REVISION
from .saved_audit import walks,paired_stats
from .scoring_v2 import extract_answer,correct


def main():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    records=[]
    for path in sorted((ROOT/"results").rglob("*.json")):
        data=json.loads(path.read_text(encoding="utf-8-sig"))
        for location,record in walks(data):
            rows=[r for r in record["detailed_results"] if {"baseline","intervened","expected_answer"}<=r.keys()]
            if not rows:
                continue
            responses=[]
            missing=[]
            for index,row in enumerate(rows):
                for side in ["baseline","intervened"]:
                    item=row[side]
                    if "text" not in item:
                        missing.append({"index":index,"side":side})
                        continue
                    text=item["text"]
                    count=len(tokenizer.encode(text,add_special_tokens=False))
                    responses.append({"index":index,"side":side,"expected_answer":row["expected_answer"],
                        "text_sha256":hashlib.sha256(text.encode()).hexdigest(),"saved_tokens":item["tokens"],
                        "retokenized_decoded_text_tokens":count,"saved_minus_retokenized":item["tokens"]-count,
                        "answer_v2":extract_answer(text),"correct_v2":correct(text,row["expected_answer"]),
                        "saved_correct":item.get("correct")})
            records.append({"file":str(path.relative_to(ROOT)),"record":location,"n":len(rows),
                "token_stats":paired_stats([r["baseline"]["tokens"] for r in rows],[r["intervened"]["tokens"] for r in rows]),
                "baseline_correct_v2":sum(r["side"]=="baseline" and r["correct_v2"] for r in responses),
                "intervened_correct_v2":sum(r["side"]=="intervened" and r["correct_v2"] for r in responses),
                "missing_full_responses":missing,"responses":responses})
    counts=Counter(r["saved_minus_retokenized"] for record in records for r in record["responses"])
    result={"tokenizer_revision":REVISION,"quality_records":len(records),
            "response_records":sum(len(r["responses"]) for r in records),
            "count_delta_histogram":dict(sorted(counts.items())),"records":records,
            "scope":"All text-bearing original quality arrays; duplicate files retained by path. V2 scalar scores are conditional on final-answer extraction and do not validate reasoning. Decoded text cannot recover historical token IDs/EOS. Missing responses cannot be rescored."}
    (ROOT/"independent_audit/results/historical_text_validation.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))


if __name__=="__main__":
    main()
