"""Sanity checks are prerequisites: matching two broken paths is not validation."""
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformer_lens import HookedTransformer

from .engine import generate
from .model import MODEL,REVISION,MODEL_DIR,ROOT


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    ids=tok.apply_chat_template([{"role":"user","content":"What is 7 + 5?"}],add_generation_prompt=True,return_tensors="pt")
    hf=AutoModelForCausalLM.from_pretrained(MODEL_DIR,local_files_only=True,torch_dtype=torch.float16,attn_implementation="eager").eval()
    records=[]
    for device,dtype in [("cpu",torch.float32),("mps",torch.float32),("mps",torch.float16)]:
        hf=hf.to(device=device,dtype=dtype)
        logits=hf(ids.to(device)).logits[:,-1,:]
        output=hf.generate(ids.to(device),max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
        row={"backend":"huggingface","device":device,"dtype":str(dtype),"text":tok.decode(output[0,ids.shape[1]:]),
             "argmax":int(logits.argmax()),"finite":bool(logits.isfinite().all())}
        records.append(row)
        print(json.dumps(row),flush=True)
    hf=hf.to(device="cpu",dtype=torch.float32)
    for processed in [True,False]:
        factory=HookedTransformer.from_pretrained if processed else HookedTransformer.from_pretrained_no_processing
        model=factory(MODEL,hf_model=hf,tokenizer=tok,device="cpu",dtype=torch.float32,revision=REVISION).eval()
        for device in ["cpu","mps"]:
            model=model.to(device)
            logits=model(ids.to(device))[:,-1,:]
            output=generate(model,ids.to(device),16,tok.eos_token_id)
            row={"backend":"transformer_lens","processed":processed,"device":device,"dtype":"float32",
                 "argmax":int(logits.argmax()),"finite":bool(logits.isfinite().all()),
                 "text":tok.decode(output["token_ids"]),"seconds":output["generation_seconds"]}
            records.append(row)
            print(json.dumps(row),flush=True)
        del model
        gc.collect()
        torch.mps.empty_cache()
    (ROOT/"independent_audit/results/backend_diagnostic.json").write_text(json.dumps(records,indent=2))


if __name__=="__main__":
    main()
