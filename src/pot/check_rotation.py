#!/usr/bin/env python3
"""
Equivalence check for pot/rotation.py.

Stage 1 (cheap): max |logit| difference between the original model and the rotated
model on a fixed prompt, plus the NLL on one wikitext window.  A rotation that is not
equivalence preserving blows up here immediately.

Stage 2 (the deliverable): full WikiText-2 perplexity of the ROTATED, UNQUANTIZED
model.  It must match the bf16 baseline (16.72) to within float noise.

    python3 src/pot/check_rotation.py --rotate random_hadamard
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
for p in (_SRC, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from pot.rotation import rotate_qwen3   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--rotate", default="random_hadamard",
                    choices=["none", "hadamard", "random_hadamard"])
    ap.add_argument("--no-r1", action="store_true")
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--no-r4", action="store_true")
    ap.add_argument("--untie-lm-head", action="store_true",
                    help="explicitly costed option: fuse R1 into embed/lm_head by "
                         "untying them (+0.622 GB bf16 on Qwen3-1.7B). Off by default.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="float32 isolates the rotation algebra from bf16 storage noise")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pot.eval_ppl import eval_wikitext_ppl, get_wikitext2_testenc

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval()
    model.config.use_cache = False

    testenc = get_wikitext2_testenc(tok)
    batch = testenc[:, : args.seqlen].to(dev)

    model.to(dev)
    with torch.no_grad():
        ref = model(batch).logits.float().cpu()
    model.cpu()
    torch.cuda.empty_cache()

    info = rotate_qwen3(model, mode=args.rotate, seed=args.seed, dev=dev,
                        do_r1=not args.no_r1, do_r2=not args.no_r2,
                        do_r4=not args.no_r4, untie_lm_head=args.untie_lm_head)
    model.to(dev)
    with torch.no_grad():
        got = model(batch).logits.float().cpu()

    d = (ref - got).abs()
    rel = d.max().item() / ref.abs().max().item()
    print(f"[stage1] max|dlogit| = {d.max().item():.4f}  "
          f"mean|dlogit| = {d.mean().item():.5f}  "
          f"rel-to-max-logit = {rel:.2e}")
    # agreement of the argmax token, which is what actually matters downstream
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    print(f"[stage1] argmax agreement = {agree * 100:.3f}%")

    if args.stage1_only:
        return 0

    ppl = eval_wikitext_ppl(model, tok, seqlen=args.seqlen, device=dev)
    print(f"[stage2] rotated UNQUANTIZED wikitext2 ppl = {ppl:.4f}   "
          f"(bf16 baseline 16.72; equivalence requires |d| < 0.05)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
