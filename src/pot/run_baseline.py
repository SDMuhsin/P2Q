"""
Baseline PTQ harness: load a HF causal LM, optionally fake-quantize every nn.Linear
inside the decoder layers, and report WikiText-2 perplexity.

Example:
    export CUDA_VISIBLE_DEVICES=0
    python3 src/pot/run_baseline.py --model Qwen/Qwen3-1.7B --method none
    python3 src/pot/run_baseline.py --model Qwen/Qwen3-1.7B --method rtn_int_asym --bits 4 --group 128
    python3 src/pot/run_baseline.py --model Qwen/Qwen3-1.7B --method rtn_int_asym --tasks ppl,arc_easy

Results append to results/pot/baselines.jsonl.

What gets quantized
-------------------
Every `nn.Linear` reachable from `model.model.layers` (the decoder stack): q/k/v/o_proj
and gate/up/down_proj. Explicitly NOT quantized:
  * the embedding (`model.embed_tokens`) -- not an nn.Linear anyway,
  * `lm_head` -- lives outside the layer stack, and for Qwen3-0.6B/1.7B it is TIED to
    the embedding, so quantizing it would silently corrupt the embedding too.
This is the standard PTQ protocol (GPTQ/AWQ/SINQ all skip embeddings + head).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from pot.eval_arc import eval_arc_easy
from pot.eval_ppl import eval_wikitext_ppl
from pot.quantizers import quantize_weight, QUANTIZERS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO_ROOT, "results", "pot", "baselines.jsonl")


def get_decoder_layers(model):
    """Return the ModuleList of decoder blocks for common HF architectures."""
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, (nn.ModuleList, list)):
            return obj
    raise RuntimeError(f"could not locate decoder layers on {type(model).__name__}")


@torch.no_grad()
def quantize_model_(model, method, bits, group, verbose=True):
    """
    In-place fake-quantization (quantize->dequantize) of all decoder nn.Linear weights.
    Returns a stats dict.
    """
    layers = get_decoder_layers(model)
    n_lin, n_params, sq_err, sq_norm = 0, 0, 0.0, 0.0
    skipped = []

    for li, layer in enumerate(layers):
        for name, mod in layer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            W = mod.weight.data
            in_f = W.shape[1]
            g = group if (group > 0 and group <= in_f) else in_f
            if in_f % g != 0:
                skipped.append(f"layers.{li}.{name} (in={in_f} % group={g} != 0)")
                continue
            dev = W.device
            W_hat, meta = quantize_weight(W.to("cuda" if torch.cuda.is_available() else dev),
                                          method, bits=bits, group=g)
            W_hat = W_hat.to(dev)
            sq_err += meta["mse"] * W.numel()
            sq_norm += W.float().pow(2).sum().item()
            mod.weight.data.copy_(W_hat)
            n_lin += 1
            n_params += W.numel()

    stats = {
        "n_linear_quantized": n_lin,
        "n_weights_quantized": n_params,
        "weight_rel_fro_err": (sq_err / sq_norm) ** 0.5 if sq_norm > 0 else 0.0,
        "skipped": skipped,
    }
    if verbose:
        print(f"[quant] {method} b={bits} g={group}: {n_lin} Linear layers, "
              f"{n_params/1e6:.1f}M weights, global rel Fro err = "
              f"{stats['weight_rel_fro_err']:.4f}")
        for s in skipped:
            print(f"[quant] SKIPPED {s}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--method", default="none",
                    choices=["none"] + sorted(QUANTIZERS))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128,
                    help="group size along the input dim; <=0 means per-output-channel")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--tasks", default="ppl",
                    help="comma-separated subset of {ppl, arc_easy}, or 'all'")
    ap.add_argument("--arc-limit", type=int, default=None,
                    help="evaluate only the first N ARC docs (fast iteration)")
    ap.add_argument("--arc-batch-size", type=int, default=16)
    ap.add_argument("--mc-extra", action="store_true",
                    help="also score pot.eval_mc.EXTRA_TASKS (MMLU); excluded from "
                         "panel_macro by construction")
    ap.add_argument("--mc-token-budget", type=int, default=4096,
                    help="token budget for the pot.eval_mc panel batcher. MUST match "
                         "the value every compared artifact used: the packing is part "
                         "of the instrument and changing it perturbs scores (~0.4 pp).")
    ap.add_argument("--mc-dump", default=None,
                    help="directory for per-item panel score dumps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--tag", default="", help="free-form label stored in the JSONL record")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)

    t0 = time.time()
    # Kept deliberately in step with pot/run_pot.py's setup (`use_cache = False`,
    # `use_fast=True`), so a bf16 reference is measured on the same code path as the
    # artifacts it anchors -- a damage fraction is precisely a claim that the two were
    # measured the same way.
    #
    # NOTE: aligning these two did NOT explain the 2026-09-14 discrepancy, in which this
    # runner's bf16 panel disagreed with the SESSION 4 `eval_mc.BF16` constants on 5 of
    # 7 tasks by up to 0.20 pp. Runs before and after the alignment agree EXACTLY, so
    # neither flag was the cause and bf16 evaluation is reproducible on this path. The
    # remaining difference from run_pot is `attn_implementation="sdpa"`, passed here and
    # not there. Do not read the stored BF16 constants as interchangeable with a fresh
    # measurement until that is settled.
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(args.device).eval()
    model.config.use_cache = False
    t_load = time.time() - t0
    print(f"[load] {args.model} in {args.dtype} ({t_load:.1f}s), "
          f"tied_embeddings={getattr(model.config, 'tie_word_embeddings', None)}")

    qstats = {}
    t1 = time.time()
    if args.method != "none":
        qstats = quantize_model_(model, args.method, args.bits, args.group)
    t_quant = time.time() - t1

    ALL_TASKS = ["ppl", "arc_easy", "mc"]
    tasks = ALL_TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        ap.error(f"unknown task(s) {sorted(unknown)}; choose from {ALL_TASKS}")

    metrics, timings, parts = {}, {}, []

    if "ppl" in tasks:
        t2 = time.time()
        ppl = eval_wikitext_ppl(model, tok, seqlen=args.seqlen, device=args.device)
        timings["t_ppl_s"] = round(time.time() - t2, 2)
        metrics["wikitext2_ppl"] = ppl
        parts.append(f"wikitext2_ppl={ppl:.4f}")

    if "arc_easy" in tasks:
        t3 = time.time()
        arc = eval_arc_easy(model, tok, device=args.device, limit=args.arc_limit,
                            batch_size=args.arc_batch_size)
        timings["t_arc_s"] = round(time.time() - t3, 2)
        metrics.update({"arc_easy_acc": arc["acc"],
                        "arc_easy_acc_norm": arc["acc_norm"],
                        "arc_easy_acc_norm_token": arc["acc_norm_token"],
                        "arc_easy_n": arc["n"]})
        parts.append(f"arc_easy_acc={arc['acc']:.4f} "
                     f"acc_norm={arc['acc_norm']:.4f} "
                     f"acc_norm_token={arc['acc_norm_token']:.4f} (n={arc['n']})")

    if "mc" in tasks:
        # The 7-task panel, scored through the same path pot/run_pot.py uses, so a
        # bf16 reference here is directly comparable to a quantized artifact's panel.
        # Needed per model: eval_mc.BF16 holds Qwen3-1.7B only, and a damage fraction
        # against the wrong model's reference is meaningless.
        t4 = time.time()
        from pot.eval_mc import eval_panel
        from pot.eval_mc import EXTRA_TASKS
        mc = eval_panel(model, tok, device=args.device,
                        token_budget=args.mc_token_budget, dump_dir=args.mc_dump,
                        extra_tasks=EXTRA_TASKS if args.mc_extra else ())
        timings["t_mc_s"] = round(time.time() - t4, 2)
        metrics["mc"] = mc
        parts.append(f"panel_macro={mc['panel_macro']:.4f} "
                     f"panel_micro={mc['panel_micro']:.4f} (n={mc['n_items']})")

    t_eval = sum(timings.values())
    label = ("bf16" if args.method == "none"
             else f"{args.method} b{args.bits} g{args.group}")
    print(f"\nRESULT | {args.model} | {label} | seqlen={args.seqlen} | "
          + " | ".join(parts)
          + f" | eval={t_eval:.1f}s quant={t_quant:.1f}s")

    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "method": args.method, "bits": args.bits,
        "group": args.group, "seqlen": args.seqlen, "dtype": args.dtype,
        "tasks": tasks, "tag": args.tag,
        **metrics,
        "t_load_s": round(t_load, 2), "t_quant_s": round(t_quant, 2),
        **timings,
        **{k: v for k, v in qstats.items() if k != "skipped"},
        "n_skipped": len(qstats.get("skipped", [])),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
