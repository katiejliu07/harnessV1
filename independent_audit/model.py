"""Immutable model revision; local cache avoids accidental updates between runs."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / ".audit-model"


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL,revision=REVISION,local_dir=MODEL_DIR,
                      allow_patterns=["*.json","*.txt","*.safetensors","*.model","*.jinja"])
    print(f"Downloaded {MODEL}@{REVISION} to {MODEL_DIR}",flush=True)


def load(device="auto", dtype="float16"):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformer_lens import HookedTransformer
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dt = getattr(torch,dtype)
    if device == "cpu" and dtype == "float16":
        raise ValueError("Request float32 explicitly for CPU; do not silently change dtype")
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    hf=AutoModelForCausalLM.from_pretrained(MODEL_DIR,local_files_only=True,torch_dtype=dt,attn_implementation="eager")
    # Keep TransformerLens default centering/folding, as used by original shared.py.
    # hf_model and tokenizer ensure weights/tokenizer use the same locked snapshot.
    model=HookedTransformer.from_pretrained(MODEL,hf_model=hf,tokenizer=tok,device=device,dtype=dt,move_to_device=True,revision=REVISION)
    del hf
    gc.collect()
    model.eval()
    torch.set_grad_enabled(False)
    meta={"repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
          "model":MODEL,"model_revision":REVISION,"original_model_revision":"not recorded in repository",
          "device":device,"dtype":str(dt),"platform":platform.platform(),"python":platform.python_version(),
          "cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,
          "mps_available":torch.backends.mps.is_available(),"torch_threads":torch.get_num_threads(),
          "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
          "model_load_seconds":time.perf_counter()-start,
          "generation":{"do_sample":False,"temperature":1.,"top_p":1.,"max_new_tokens":320,"batch_size":1,"eos_token_id":tok.eos_token_id},
          "transformer_lens_config":model.cfg.to_dict()}
    meta["transformer_lens_config"]={k:str(v) if isinstance(v,(torch.dtype,torch.device)) else v for k,v in meta["transformer_lens_config"].items()}
    return model,tok,meta


if __name__ == "__main__":
    download()
