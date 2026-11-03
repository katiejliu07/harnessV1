"""CLI uses measured policy defaults, with no hidden retries or output stripping."""
import argparse
import json
from pathlib import Path
import sys
import torch

from independent_audit.model import load
from .runtime import PROFILES,infer,recommended_profile


def main():
    ap=argparse.ArgumentParser(description="Compact prompts; preserve the complete task and raw generated answer.")
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file",type=Path)
    ap.add_argument("--kind",choices=["scalar","code","general"],required=True)
    ap.add_argument("--profile",choices=["auto",*PROFILES],default="auto")
    ap.add_argument("--device",default="auto")
    ap.add_argument("--dtype",default="float16")
    ap.add_argument("--max-new-tokens",type=int)
    ap.add_argument("--metadata",type=Path)
    args=ap.parse_args()
    task=args.prompt if args.prompt is not None else args.prompt_file.read_text()
    torch.set_num_threads(4)
    # Loading libraries may print diagnostics; keep them out of the answer stream.
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        model,tok,environment=load(args.device,args.dtype)
    profile=recommended_profile(args.kind,environment=environment) if args.profile=="auto" else args.profile
    result=infer(model,tok,task,args.kind,profile,args.max_new_tokens)
    print(result["text"])
    if args.metadata:
        args.metadata.write_text(json.dumps({"environment":environment,**result},indent=2))
    if result["status"]!="complete_unverified":
        print(f"Generation {result['status']}; output is not verified complete.",file=sys.stderr)
        raise SystemExit(2)


if __name__=="__main__":
    main()
