#!/usr/bin/env python3
"""
P2Q end-to-end driver: quantize a causal LM layer-by-layer with PoTGPTQ, verify the
shift-free condition on the artifact, account the bits-per-weight honestly, and evaluate.

    python3 src/pot/run_pot.py --codebook strict_pot --group 128 --tasks ppl,arc_easy

Memory plan: one decoder layer on the GPU at a time, everything else on CPU. Calibration
activations are propagated layer by layer (the standard GPTQ driver), so the Hessian each
Linear sees is built from the *already quantized* prefix of the network -- which is what
makes sequential PTQ work.

Every run appends one JSON record to results/pot/pot_runs.jsonl. A run is only recorded
as `success: true` when BOTH
  * the shift-free verifier passes on every layer it was run on, AND
  * the fully accounted bits-per-weight is <= --bpw-budget (default 4.5).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
for p in (_SRC, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from pot.codebooks import get_codebook, fit_add2_codebook, Codebook          # noqa: E402
from pot.gptq_pot import PoTGPTQ                                             # noqa: E402
from pot.alphabets import (default_family, selector_bits_for,               # noqa: E402
                           PoTGPTQFamily)
from pot.scale_search import search_col_scale                                # noqa: E402
from pot.equalize import equalize_columns, qwen3_equalize_plan              # noqa: E402
from pot.code_refine import refine_codes                                     # noqa: E402
from pot.scale_learn import learn_scale_residual                             # noqa: E402
from pot.rotation import rotate_qwen3                                        # noqa: E402
from pot.verify_shiftfree import verify_codebook_artifact, expand_scales    # noqa: E402
from pot.out_correct import output_correction, quantize_bias                 # noqa: E402

RESULTS = os.path.join(_ROOT, "results", "pot", "pot_runs.jsonl")


# --------------------------------------------------------------------------- utils
def log(*a, **k):
    print(*a, **{**k, "flush": True})


def find_linears(module, prefix=""):
    out = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and name:
            out[name] = child
    return out


def parse_group_map(spec):
    """Parse '--group-map default=128,mlp.gate_proj=32,mlp.down_proj=32' into a dict
    {suffix_or_'default': int_group}. Returns {} if spec is falsy (uniform --group)."""
    if not spec:
        return {}
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        key, _, val = tok.partition("=")
        key = key.strip()
        if not key or not val.strip():
            raise ValueError(f"bad --group-map token {tok!r}, expected key=int")
        out[key] = int(val.strip())
    if "default" not in out:
        raise ValueError("--group-map must include a default=<group> entry")
    return out


def _scale_weight_arg(v):
    """argparse type for --scale-weight: one of the named values, or 'powdiag:<float>'."""
    from pot.gptq_pot import PoTGPTQ
    if v in PoTGPTQ.SCALE_WEIGHTS or PoTGPTQ._parse_powdiag(v) is not None:
        return v
    raise argparse.ArgumentTypeError(
        f"bad --scale-weight {v!r}; expected one of {PoTGPTQ.SCALE_WEIGHTS} "
        f"or 'powdiag:<alpha>'")


def resolve_group(name, group_map, fallback):
    """Resolve the group size for a linear `name` (e.g. 'mlp.gate_proj') against a
    parsed --group-map. Longest-suffix match wins over shorter/other suffixes; a
    matching 'default' key is used when nothing else matches; `fallback` (args.group)
    is used only when group_map is empty (i.e. --group-map was not passed at all)."""
    if not group_map:
        return fallback
    best_key, best_len = None, -1
    for key in group_map:
        if key == "default":
            continue
        if name.endswith(key) and len(key) > best_len:
            best_key, best_len = key, len(key)
    if best_key is not None:
        return group_map[best_key]
    return group_map["default"]


def get_decoder_layers(model):
    m = getattr(model, "model", model)
    for attr in ("layers", "decoder"):
        obj = getattr(m, attr, None)
        if obj is not None:
            return obj.layers if hasattr(obj, "layers") else obj
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("cannot locate decoder layers")


# ------------------------------------------------------------------------ bpw
# Per-column g_idx cost for --act-order, measured for Qwen3-1.7B (1,409,286,144
# quantized weights): act-order permutes columns, so group boundaries no longer align
# with contiguous column ranges and each of the model's input columns (summed over all
# seven matrix kinds x 28 layers) needs its own stored group index. account_bpw did NOT
# charge for this term before; any run that passes --act-order must include it (PI
# audit) so the reported total stays honest.
G_IDX_BPW_ACT_ORDER = 0.002075


def account_bpw(n_weights, n_groups_total, n_rows_total, code_bits, group,
                scale_bits, zero_bits, col_scale_bits, col_scale_entries,
                codebook_entries, codebook_bits=16, selector_bits=0, g_idx_bpw=0.0,
                eq_bits=0, eq_entries=0, bias_bits=0, bias_entries=0,
                bias_mats=0):
    """Fully accounted bits per weight. Every term is returned separately so the total
    is auditable; nothing is hidden behind a nominal '4-bit'.

    `selector_bits`: bits per (row, column-group) charged for a per-group ALPHABET
    selector (see alphabets.py / --alphabet-select). It is charged at the same
    granularity as the group scale (n_groups_total), because a selector lives
    alongside the scale it selects the ladder for.
    `eq_bits` x `eq_entries`: storage charged for --equalize's per-input-column
    diagonal. It is ZERO by default and that zero is a CLAIM, not an omission: every
    Qwen3 linear folds its diagonal exactly into an op whose parameters are already
    stored and already uncharged (an fp RMSNorm weight, or the fp16 output-row group
    scales of v_proj / up_proj). See `pot.equalize.qwen3_equalize_plan`. The run record
    also carries the hypothetical cost at 4 and 16 bits per entry so the claim can be
    priced by anyone who rejects it.
    `bias_bits` x `bias_entries`: storage for --out-correct's per-output-row bias, one
    `bias_bits`-wide integer per output row against a per-matrix float32 step. The
    `bias_mats` steps are 32 bits each (196 matrices over 1.409e9 weights, ~4.5e-6 bpw)
    and ARE included below rather than waved away. Unlike --equalize's diagonal there is no
    op to absorb a bias into, so this term is never zero when the option is on.
    `g_idx_bpw`: honest flat addition for --act-order's stored per-column group index
    (see G_IDX_BPW_ACT_ORDER above); 0.0 when act-order is off.
    """
    codes = code_bits
    scale = n_groups_total * n_rows_total * 0.0                       # placeholder
    scale = (scale_bits + zero_bits) * n_groups_total / n_weights
    col = col_scale_bits * col_scale_entries / n_weights
    book = codebook_entries * codebook_bits / n_weights
    sel = selector_bits * n_groups_total / n_weights
    breakdown = {
        "codes": codes,
        "group_scale": scale,
        "zero_point": 0.0 if zero_bits == 0 else zero_bits * n_groups_total / n_weights,
        "col_scale": col,
        "codebook": book,
        "alphabet_selector": sel,
        "g_idx": g_idx_bpw,
        "equalize": eq_bits * eq_entries / n_weights if n_weights else 0.0,
        "out_bias": ((bias_bits * bias_entries + 32 * bias_mats) / n_weights
                     if (n_weights and bias_entries) else 0.0),
    }
    # zero_bits is already inside `scale`; keep the reported terms disjoint
    if zero_bits:
        breakdown["group_scale"] = scale_bits * n_groups_total / n_weights
        breakdown["zero_point"] = zero_bits * n_groups_total / n_weights
    total = sum(breakdown.values())
    breakdown["total"] = total
    return total, breakdown


@torch.no_grad()
def rung_stats(codes, levels, W_hat, W):
    """Occupancy of the two EXTREME rungs of the alphabet, and how far the weights that
    land there had to move in log space.

    This is the quantity that actually discriminates on a log grid, and weight MSE is
    blind to it. A weight clamped UP onto the bottom rung 2^-7 carries unbounded
    RELATIVE error while contributing negligible SQUARED error -- which is exactly how
    `pot4 g=16` reached rel_mse 0.026303 (better than the incumbent's 0.028618) at
    bpw 4.000000 (cheaper) and still collapsed to acc_norm 0.3451 / ppl 1826.

    By the covariance theorem in pot/equalize.py, sliding a column within the octave
    window is the ONLY thing a column scale can do on this alphabet, so this is also the
    direct readout of what --equalize is doing.
    """
    a = levels.abs().to(codes.device)
    amin, amax = float(a.min()), float(a.max())
    mag = a[codes.long()]
    bot = mag <= amin * (1 + 1e-6)
    top = mag >= amax * (1 - 1e-6)

    def dlog(mask):
        n = int(mask.sum())
        if not n:
            return 0.0, 0
        wh = W_hat[mask].abs().clamp_min(1e-30)
        w = W[mask].abs().clamp_min(1e-30)
        return float((torch.log2(wh) - torch.log2(w)).abs().mean()), n

    db, nb = dlog(bot)
    dt, nt = dlog(top)
    return {"n": int(codes.numel()), "n_bottom": nb, "n_top": nt,
            "dlog2_bottom_sum": db * nb, "dlog2_top_sum": dt * nt}


def check_scale_storable(scales_list, scale_bits):
    """Refuse to charge fewer than 16 bits for a scale we did not actually constrain.

    `--scale-bits 8` claims the group scale is stored as an 8-bit power-of-two exponent.
    That claim is only true if the scales really ARE powers of two with an exponent in
    int8 range. Nothing in the sweep enforces that unless `--pot-scale` is set, so
    without this check `--scale-bits 8` is a way to write a smaller number in the bpw
    column without changing the artifact -- exactly the "reported as 4-bit while hiding
    metadata" failure the budget exists to prevent.

    Returns (ok, detail).
    """
    if scale_bits >= 16:
        return True, {"checked": False, "reason": "fp16 scale, nothing to prove"}
    bad_pot = 0
    bad_range = 0
    total = 0
    emin, emax = float("inf"), float("-inf")
    for sc in scales_list:
        sc = sc.detach().float().reshape(-1)
        sc = sc[sc > 0]
        total += sc.numel()
        m, e = torch.frexp(sc)
        bad_pot += int((m != 0.5).sum())
        if e.numel():
            emin = min(emin, int(e.min())); emax = max(emax, int(e.max()))
    span = 2 ** (scale_bits - 1)
    if emin != float("inf"):
        bad_range = int(emin < -span or emax > span - 1)
    ok = (bad_pot == 0 and bad_range == 0)
    return ok, {"checked": True, "scale_bits": scale_bits, "n_scales": total,
                "n_not_power_of_two": bad_pot, "exp_min": None if emin == float("inf") else emin,
                "exp_max": None if emax == float("-inf") else emax,
                "exp_out_of_range": bool(bad_range)}


# ------------------------------------------------------------ calibration capture
@torch.no_grad()
def eval_with_oom_retry(fn, label, need_mb=None, attempts=6, poll_s=60, log=log):
    """Run an evaluation, and on CUDA OOM WAIT for memory and run it again.

    The GPUs on this box are shared, and a neighbour that grows mid-run can OOM an
    evaluation after the expensive part of the job is already done: `SS_g128_mu1` lost
    28 minutes of completed, verified quantization to an OOM in the FIRST eval, with
    every one of its 28 layers already solved.

    Retrying in-process keeps the quantized model and re-runs the SAME computation with
    the same arguments, so the retry is byte-identical to the first attempt. That is the
    only reason this is allowed to exist: the alternative fix -- shrinking a batch to fit
    -- would change the batch packing, and `pot.eval_mc`'s packing is deliberately a
    constant of the code (the evaluator's own batching noise floor is 0.42 pp, larger
    than the effects being ranked). Adapt the SCHEDULE, never the instrument.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            if attempt == attempts:
                raise
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            want = need_mb if need_mb is not None else max(2048, int(0.25 * total / 2**20))
            log(f"  [oom] {label} attempt {attempt}/{attempts}: {free // 2**20} MB free, "
                f"waiting for {want} MB (another tenant grew); the quantized model is "
                f"kept and the eval will be repeated identically")
            waited = 0
            while waited < 1800:
                time.sleep(poll_s)
                waited += poll_s
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
                if free // 2**20 >= want:
                    break
            log(f"  [oom] {label}: retrying after {waited}s ({free // 2**20} MB free)")


def capture_layer0_inputs(model, layers, calib, device, seqlen, batch=1):
    """Run the embedding + everything before layer 0 and catch its inputs/kwargs."""
    cache = {"inps": [], "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod
            # transformers 4.55 reads decoder_layer.attention_type off the module before
            # calling it, so the wrapper has to forward unknown attributes to the layer.

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.__dict__["_modules"]["mod"], name)

        def forward(self, hidden_states, **kw):
            cache["inps"].append(hidden_states.detach().to("cpu"))
            if cache["kwargs"] is None:
                cache["kwargs"] = {k: v for k, v in kw.items()}
            raise ValueError("stop")

    layers[0] = Catcher(layers[0])
    for i in range(0, calib.shape[0], batch):
        try:
            model(calib[i:i + batch].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].mod
    return torch.cat(cache["inps"], 0), cache["kwargs"]


# ------------------------------------------------------------------------- main
def load_calibration_cached(loader, tok, dataset_key, want, seqlen, cache_path):
    """`loader(...)` with an optional on-disk cache of the tokenized calibration tensor.

    The generic-corpus loaders re-tokenize the whole training split on every call. For
    wikitext2 that is seconds; for wikitext103 it is ~11 minutes, which dominates a 7
    minute quantization run and makes many-draw protocols unaffordable. The cache stores
    the tokenized [N, seqlen] tensor keyed by (dataset, seqlen) and serves any request
    for `want <= N` by slicing its head -- which is exactly what the loader itself
    returns, because the loaders chunk contiguously from the head of the split with no
    RNG. So the cache is a pure speedup: same bits, no behaviour change, and it is only
    consulted when --calib-cache is passed.

    A cache built for a smaller N than requested is ignored (and rebuilt), never
    silently padded.
    """
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        if (blob.get("dataset") == dataset_key and blob.get("seqlen") == seqlen
                and blob["ids"].shape[0] >= want):
            log(f"  calib cache hit: {cache_path} "
                f"({tuple(blob['ids'].shape)}, using first {want})")
            return blob["ids"][:want]
        log(f"  calib cache at {cache_path} does not cover this request "
            f"(dataset={blob.get('dataset')}, seqlen={blob.get('seqlen')}, "
            f"n={blob['ids'].shape[0]} < {want}); rebuilding")
    ids = loader(tok, n_samples=want, seq_len=seqlen, dataset_key=dataset_key)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save({"dataset": dataset_key, "seqlen": seqlen, "ids": ids}, cache_path)
        log(f"  calib cache written: {cache_path} {tuple(ids.shape)}")
    return ids


def main():
    ap = argparse.ArgumentParser(description="P2Q / PoT-GPTQ end-to-end driver")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--codebook", default="strict_pot",
                    help="strict_pot | strict_pot_zero | apot_k2_b4 | learned_add2 | "
                         "int4_asym | int4_sym | nf4 | fit_add2")
    ap.add_argument("--codebook-nexp", type=int, default=None,
                    help="DIAGNOSTIC: override the number of octaves in strict_pot / "
                         "strict_pot_zero. Default (unset) keeps the standard 4-bit "
                         "books, strict_pot(8) and strict_pot_zero(7). Any other value "
                         "changes the code width -- strict_pot(16) is 32 levels = 5 "
                         "bits/weight -- which account_bpw charges honestly and which "
                         "will therefore exceed the 4.25 bpw ceiling. Such arms are "
                         "capacity diagnostics, never deliverables.")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--group-map", default="",
                    help="per-matrix group override, e.g. "
                         "'default=64,mlp.gate_proj=128,mlp.down_proj=32'. Resolved by "
                         "longest-suffix match against the linear's dotted name within "
                         "each decoder layer (e.g. 'self_attn.q_proj', 'mlp.down_proj'); "
                         "a key that is not 'default' must be a real suffix match or it "
                         "silently never fires -- check the per-layer log line. The "
                         "'default' entry is required and is used for any linear whose "
                         "name matches no other key. When set, --group is ignored.")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-dataset", default="wikitext2")
    ap.add_argument("--calib-cache", default=None,
                    help="cache the tokenized calibration tensor at this path and reuse "
                         "it on later runs. A pure speedup -- the loaders chunk "
                         "contiguously from the head of the split with no RNG, so the "
                         "cached head is bit-identical to what a fresh load returns. "
                         "Worth it for wikitext103, whose tokenization costs ~11 min per "
                         "run against a ~7 min quantization. Off by default.")
    ap.add_argument("--calib-offset", type=int, default=0,
                    help="skip the first N sequences of the chunked calibration set before "
                         "taking --nsamples, giving a DISJOINT draw from the same corpus. "
                         "This is the only way to vary the calibration data: the loader "
                         "takes contiguous chunks from the head of the tokenized split and "
                         "uses no RNG, so --seed cannot change it (--seed reaches only the "
                         "random Hadamard). Default 0, so existing invocations are unchanged.")
    ap.add_argument("--method", default="gptq", choices=["gptq", "rtn"],
                    help="rtn = no error compensation (H := I), the ablation baseline")
    ap.add_argument("--act-order", action="store_true",
                    help="alias for --order act (used only when --order is left unset)")
    ap.add_argument("--order", default=None,
                    choices=["none", "act", "err", "errasc", "errmag",
                             "logmean", "logmeanasc", "salogmean", "rand"],
                    help="column processing order for the GPTQ sweep. 'act' is the "
                         "classic descending diag(H) heuristic (== --act-order); 'err' "
                         "sorts on diag(H)_j * e_j with e_j the measured PoT error of "
                         "column j from an RTN pre-pass; 'errmag' uses the pre-pass-free "
                         "proxy diag(H)_j * ||W[:,j]||^2; 'errasc' is ascending 'err', a "
                         "falsification control. Any non-identity order costs exactly the "
                         "same stored g_idx as --act-order, so all of these are "
                         "bpw-neutral against each other. Unset means: take it from "
                         "--act-order. 'logmean' sorts on the column's log-magnitude "
                         "LOCATION mean_i log2|W_ij|, which under --method rtn acts only "
                         "through group membership and is aimed at within-group dynamic "
                         "range (hence bottom-rung clamping); 'salogmean' is its "
                         "salience-weighted form log2 h_j + 2 mu_j (the section-21.1 "
                         "per-weight output energy, geometric-mean-reduced to a column "
                         "statistic); 'logmeanasc' is the reversal, which under rtn is a "
                         "NULL change because reversing a permutation leaves every group "
                         "SET identical; 'rand' is a random permutation with the same "
                         "g_idx cost, the falsification control.")
    ap.add_argument("--perm-seed", type=int, default=0,
                    help="seed for --order rand. Fully deterministic; --seed does not "
                         "reach it.")
    ap.add_argument("--rescale-in-loop", dest="rescale", action="store_true", default=True)
    ap.add_argument("--no-rescale-in-loop", dest="rescale", action="store_false")
    ap.add_argument("--col-scale", dest="col_scale", action="store_true", default=False,
                    help="ablation only: buys ~0.9%% relMSE, costs 0.008 bpw")
    ap.add_argument("--no-col-scale", dest="col_scale", action="store_false")
    # --- activation-aware PoT column equalization (src/pot/equalize.py) ---
    ap.add_argument("--equalize", default="off", choices=["off", "int", "real"],
                    help="activation-aware per-input-column equalization, the AWQ/"
                         "SmoothQuant/DFQ insight in its PoT-native form. 'int' "
                         "constrains the diagonal to c_j = 2^t_j (an EXACT relabelling "
                         "on a log alphabet, and still a shift at inference); 'real' "
                         "leaves it unconstrained (strictly more general, and equally "
                         "free wherever the absorption target is an fp vector). "
                         "'off' (default) is bit-identical to before this flag existed. "
                         "The diagonal is TIED so that every group of linears sharing an "
                         "input shares one c and o_proj's is constant within each GQA "
                         "group -- which is what makes it absorbable into an upstream op "
                         "for ZERO stored bits. See qwen3_equalize_plan.")
    ap.add_argument("--eq-hess", dest="eq_hess", action="store_true", default=False,
                    help="weight the equalizer's objective by the real calibration "
                         "salience h = diag((2/n) X^T X). THIS is the AWQ ingredient: "
                         "without it the column lever is just a finer weight-MSE scale "
                         "(which is what search_col_scale already was). Note h enters "
                         "only through the shared group scale -- for fixed r it cancels "
                         "out of the per-column argmin; see equalize.py. OFF by "
                         "default: with it off the whole pipeline is H-free and hence "
                         "exactly reproducible under --method rtn, and the calibration "
                         "Gram has now failed twice as a scale objective on this grid "
                         "(--scale-weight diag/ldl), so activation-awareness is treated "
                         "here as a hypothesis to test, not an ingredient to assume.")
    ap.add_argument("--no-eq-hess", dest="eq_hess", action="store_false")
    ap.add_argument("--eq-iters", type=int, default=4,
                    help="alternations of the (exact) two half-steps. Not a tuned knob: "
                         "each half-step exactly minimizes the same objective, so the "
                         "sequence is monotone; 4 mirrors search_col_scale.")
    ap.add_argument("--eq-charge-bits", type=int, default=0,
                    help="bits charged per stored equalization exponent. Default 0 "
                         "because the absorption analysis shows every Qwen3 linear folds "
                         "its diagonal into an op that is already stored (see "
                         "qwen3_equalize_plan). The hypothetical cost at 4 and 16 bits "
                         "is recorded in the run record regardless, so the claim is "
                         "auditable.")
    ap.add_argument("--pot-scale", action="store_true",
                    help="ablation: constrain the group scale to a power of two")
    ap.add_argument("--rotate", default="none",
                    choices=["none", "hadamard", "random_hadamard"],
                    help="incoherence processing: fuse an orthogonal (Hadamard) rotation "
                         "into the weights before quantizing, W x = (W Q)(Q^T x). The "
                         "stored weights stay strictly power-of-two; the only runtime cost "
                         "is one online Hadamard (add/sub) on the down_proj input. "
                         "See src/pot/rotation.py.")
    ap.add_argument("--rotate-untie-lm-head", action="store_true",
                    help="explicitly costed rotation option: untie lm_head from "
                         "embed_tokens so R1 can be FUSED into both. Costs +0.622 GB in "
                         "bf16 on Qwen3-1.7B (+83%% on top of the whole quantized decoder), "
                         "so it is OFF by default -- the online variant is identical.")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--scale-lo", type=float, default=0.3)
    ap.add_argument("--scale-hi", type=float, default=1.2)
    ap.add_argument("--hess-weighted-scale", action="store_true",
                    help="alias for --scale-weight diag, kept so every previously "
                         "recorded run reproduces from its stored config verbatim.")
    ap.add_argument("--scale-weight", type=_scale_weight_arg, default=None,
                    help="per-column weights for the group scale-search objective. "
                         "'none' (default) is uniform weights and is bit-for-bit the "
                         "historical behaviour; 'diag' weights by diag(H) and is exactly "
                         "what --hess-weighted-scale has always meant; 'ldl' weights by "
                         "1/Hinv[j,j]^2, the weight GPTQ's own objective decomposition "
                         "implies. 'ldl' costs nothing extra over 'diag' and cuts "
                         "tr(E H E^T) by 10.7%% vs 9.0%% on 25 real layers. 'invdiag' "
                         "(1/diag(H)) is a DELIBERATELY WRONG weight kept as a "
                         "falsification control -- it makes the objective worse on "
                         "purpose and must never be used for a deliverable. "
                         "'powdiag:<a>' is diag(H)^a, the family containing all three "
                         "(a=1/0/-1), used to dial the objective delta when mapping the "
                         "response curve; also diagnostic only. Mutually exclusive with "
                         "--hess-weighted-scale.")
    ap.add_argument("--feedback-damp", type=float, default=1.0,
                    help="scalar lambda in [0,1] on the residual GPTQ propagates to later "
                         "columns. 1.0 (default) is stock GPTQ, bit-for-bit identical to "
                         "before this flag existed; 0.0 propagates nothing. Motivated by "
                         "the measured 2x2: the error feedback is worth +1.21 pp to "
                         "int4_sym and -2.21 pp to strict_pot at identical bpw, because a "
                         "log grid's resolution depends on where you move a weight and "
                         "GPTQ's retargeting is grid-blind.")
    ap.add_argument("--feedback-clip", type=float, default=None,
                    help="grid-aware cap kappa: clip each element's GPTQ correction to "
                         "kappa * |w| / 3.5, the local representable step of a strict-PoT "
                         "grid. Never asks a weight to move further than its own grid can "
                         "follow. Default None = no clip (bit-identical to before).")
    ap.add_argument("--exact-scale", action="store_true",
                    help="replace the n-grid/scale-lo/scale-hi grid scale search with "
                         "the exact breakpoint-based search (src/pot/exact_scale.py). "
                         "Default OFF; with it off, behaviour is bit-identical to "
                         "before this flag existed. n-grid/scale-lo/scale-hi are "
                         "ignored when this is set.")
    # --- the format-native log-domain objective (src/pot/log_scale.py) ---
    ap.add_argument("--log-scale", action="store_true",
                    help="fit each group scale by minimizing the FORMAT-NATIVE "
                         "log-domain distortion sum_i h_i (log2|w_hat|-log2|w|)^2 "
                         "instead of squared error on weights, solved EXACTLY by a "
                         "breakpoint argument in t=log2(s) (src/pot/log_scale.py). A "
                         "log alphabet is a uniform lattice in log-magnitude, so this "
                         "is the distortion the format induces, not a reweighting: it "
                         "charges a small weight clamped UP by 3 octaves exactly as "
                         "much as a large one clamped down by 3 -- the failure mode "
                         "squared error is structurally blind to. Supersedes "
                         "--exact-scale (passing both is an error). Default OFF; with "
                         "it off, behaviour is bit-identical to before this flag.")
    ap.add_argument("--log-scale-hess", action="store_true",
                    help="SALIENCE log-domain objective: weight each column of the "
                         "log-domain scale fit by the real calibration Gram diagonal "
                         "h_j = E[x_j^2], i.e. minimize sum_j h_j sum_i "
                         "(log2|w_hat_ij| - log2|w_ij|)^2. Requires --log-scale. "
                         "Derivation (llmdocs/SESSION_2026-09-10_GPTQ_NATIVE.md §21.5): "
                         "the honest diagonal output error is sum_j h_j sum_i w_ij^2 "
                         "eps_ij^2, so the four candidate objectives differ only in the "
                         "factor on eps^2 -- w^2 (weight MSE, the incumbent), h*w^2 "
                         "(honest output error, MEASURED AND CATASTROPHIC: 0.5282/0.5812 "
                         "vs 0.6364), 1 (--log-scale), and h alone (this flag). Under "
                         "--method rtn the quantizer's H is the IDENTITY, so this "
                         "captures diag(H) from the REAL accumulated calibration Gram "
                         "BEFORE that overwrite: the ROUNDING stays pure RTN, only the "
                         "SCALE sees salience. That makes this arm calibration-DEPENDENT "
                         "and therefore draw-dependent -- run >= 3 --calib-offset values "
                         "and judge the mean. Default OFF.")
    ap.add_argument("--log-round", action="store_true",
                    help="assign codes by nearest-in-LOG rather than nearest-in-LINEAR: "
                         "the decision boundary between 2^(e-1) and 2^e moves from the "
                         "arithmetic midpoint 1.5*2^(e-1) to the geometric midpoint "
                         "sqrt(2)*2^(e-1) ~ 1.4142*2^(e-1). Hyperparameter-free, a 5.7% "
                         "shift of every interior boundary. Deliberately a SEPARATE "
                         "switch from --log-scale so the 2x2 can be ablated. Default "
                         "OFF; with it off, behaviour is bit-identical to before.")
    # --- post-GPTQ discrete code-assignment refinement (src/pot/code_refine.py) ---
    ap.add_argument("--refine-codes", action="store_true",
                    help="after GPTQ, run coordinate descent on the CODE ASSIGNMENT: "
                         "for each weight, try moving to an adjacent codebook level and "
                         "accept iff it strictly reduces tr(E H E^T). Scales and the "
                         "codebook are never touched, so this costs zero extra bits and "
                         "the strict-PoT invariant re-verifies unchanged. See "
                         "src/pot/code_refine.py.")
    ap.add_argument("--refine-hessian", default="quantize",
                    choices=["quantize", "calib"],
                    help="which H --refine-codes descends on. 'quantize' (DEFAULT, "
                         "pre-existing behaviour) uses whatever H the quantizer itself "
                         "saw -- which under --method rtn is the IDENTITY, so the "
                         "refinement then descends plain weight MSE and is a provable "
                         "no-op from an RTN start (RTN is already the per-entry weight-MSE "
                         "minimiser given the scale). 'calib' keeps the RTN initial "
                         "assignment but hands the refinement the REAL accumulated "
                         "calibration Hessian (2/n) X^T X, so coordinate descent runs on "
                         "the true layer-output objective tr(E H E^T) from an undamaged "
                         "starting point. Only 'calib' makes --method rtn --refine-codes "
                         "a meaningful experiment; it is off by default so every existing "
                         "invocation is bit-identical.")
    ap.add_argument("--refine-damp", type=float, default=0.0,
                    help="ridge-regularise the refinement's H as GPTQ regularises its "
                         "own working copy: H += x * mean(diag H) * I. DEBUG/DIAGNOSTIC. "
                         "Default 0.0 = the raw H, which is what the module has always "
                         "used. It matters because a near-singular H makes tr(E H E^T) "
                         "reducible by pushing error into H's low-curvature subspace, "
                         "where it costs the objective nothing and ||E|| everything: "
                         "measured at --refine-sweeps 3 the raw-H descent cuts the "
                         "objective ~74%% while DOUBLING ||E||_F^2. GPTQ never descends "
                         "the raw H (it uses --percdamp 0.01), so 0.01 here is the "
                         "faithful 'same objective, non-greedy mechanism' comparison.")
    ap.add_argument("--refine-trace", action="store_true",
                    help="log and record tr(E H E^T) after every refinement sweep "
                         "(optimizer verification, not a knob). Costs one extra matmul "
                         "per sweep per linear; does not change the result.")
    ap.add_argument("--refine-sweeps", type=int, default=1,
                    help="number of full column sweeps for --refine-codes. 0 is a "
                         "no-op (anchor case): output is bit-identical to plain GPTQ.")
    # --- PoT-PTQ step 2: the learned scale residual (src/pot/scale_learn.py) ---
    ap.add_argument("--learn-scale", action="store_true",
                    help="after GPTQ, learn a multiplicative residual Gamma on the group "
                         "scales (S_hat = S*(1+Gamma)) by output reconstruction. Codes and "
                         "weights are never learned, so the PoT invariant is preserved and "
                         "re-verified.")
    ap.add_argument("--learn-iters", type=int, default=200)
    ap.add_argument("--learn-lr", type=float, default=5e-3)
    ap.add_argument("--learn-lambda", type=float, default=0.0,
                    help="(lam/2)||Gamma||_F^2 penalty")
    ap.add_argument("--learn-scope", default="block", choices=["layer", "block"],
                    help="'block' = whole-decoder-block reconstruction (PoT-PTQ's choice); "
                         "'layer' = per-linear, exact via the Gram matrix, no activations")
    ap.add_argument("--learn-retarget", default="none", choices=["gptq", "orig", "none"],
                    help="what the codes are re-assigned against as the scale moves. "
                         "'none' (default) keeps GPTQ's codes -- re-rounding on top of a "
                         "GPTQ artifact breaks the error feedback and measures 2x WORSE; "
                         "see the scale_learn.py docstring. Use 'orig' with --method rtn.")
    ap.add_argument("--learn-batch", type=int, default=1,
                    help="calibration sequences per Adam step (--learn-scope block)")
    ap.add_argument("--learn-col", action="store_true",
                    help="also learn a per-input-column residual (costs 0.0078 bpw)")
    ap.add_argument("--alphabet-select", action="store_true",
                    help="per-group ALPHABET selector (src/pot/alphabets.py): instead "
                         "of spending the whole 0.2500 bpw metadata budget on one fp16 "
                         "scale per group, spend scale + a few selector bits choosing "
                         "among a small family of strict-PoT ladders that differ in "
                         "exponent STRIDE (dynamic range), fit jointly with the scale "
                         "per (row, group). Every ladder stays a single signed power "
                         "of two (C1); the verifier needs no changes. Requires --group "
                         "to leave headroom under --bpw-budget (see the docstring in "
                         "alphabets.py for the arithmetic). Incompatible with "
                         "--exact-scale, --refine-codes, --learn-scale, --col-scale.")
    ap.add_argument("--alphabet-strides", default="1,2,3,4",
                    help="comma list of integer exponent strides for --alphabet-select's "
                         "family, e.g. '1,2,3,4' -> 4 ladders, 2-bit selector.")
    ap.add_argument("--scale-bits", type=int, default=16,
                    help="bits charged per group scale in the bpw accounting. Anything "
                         "below 16 must be BACKED by an actually-constrained scale: the "
                         "run refuses unless the stored scales are verifiably "
                         "representable in that many bits (see check_scale_storable).")
    ap.add_argument("--out-correct", default="none",
                    choices=["none", "gain", "gainmean", "bias", "both"],
                    help="POST-HOC per-output-channel correction (pot/out_correct.py), "
                         "fitted on the calibration activations AFTER the artifact is "
                         "fixed. `gain` folds a per-row factor into the stored fp16 "
                         "per-(row,group) scale -- ZERO extra bits, every code "
                         "bit-identical. `bias` adds a per-output-row constant, which is "
                         "real new storage and is charged (see --out-bias-bits). Neither "
                         "changes any code, so unlike every previous activation-aware "
                         "lever here it cannot redirect the weight error. Default off.")
    ap.add_argument("--out-bias-bits", type=int, default=8,
                    help="bits per output row charged and ACTUALLY APPLIED for "
                         "--out-correct bias/both (the bias vector is quantized to this "
                         "width against a per-matrix step, so the charge is honest). "
                         "8 bits = 0.003255 bpw on Qwen3-1.7B, which fits the "
                         "incumbent's 0.005208 bpw headroom; fp16 (16) does not.")
    ap.add_argument("--scale-offset-log2", type=float, default=0.0,
                    help="multiply every searched per-(row,group) scale by 2**DELTA. "
                         "A systematic shift of the representable window: positive "
                         "trades more top-rung clipping for less bottom-rung clamping. "
                         "bpw-neutral (the scale is still stored fp16), so this is a "
                         "diagnostic of the scale's response surface, not a saving. "
                         "Default 0.0 = inert.")
    ap.add_argument("--scale-round-log2", type=float, default=None,
                    help="snap log2(scale) to a multiple of DELTA octaves. A random "
                         "per-group perturbation of at most DELTA/2 octaves, and "
                         "simultaneously the exact distortion a DELTA-resolution stored "
                         "scale would suffer -- so this sweep prices how many bits a "
                         "per-group scale needs. bpw-neutral. Default off.")
    ap.add_argument("--solve-scale", type=float, default=None,
                    help="EXACTLY re-solve the per-(row,group) scales against the "
                         "calibration Gram with the CODES FROZEN (pot/scale_solve.py). "
                         "The layer output is linear in a row's group-scales, so this is "
                         "a G x G normal equation per row, solved in closed form. Zero "
                         "extra bits -- the scales are already stored, only their VALUES "
                         "change -- and every code stays bit-identical, so C1 is "
                         "untouched. MU is the ridge toward the incumbent MSE-optimal "
                         "scale in units of mean(diag(A)); MU=0 is the unregularised "
                         "output-error optimum, MU=inf a provable no-op. Default off.")
    ap.add_argument("--temp-match", action="store_true",
                    help="after quantization, fit ONE scalar alpha on the final RMSNorm "
                         "gain (an fp vector already stored and uncharged, so ZERO bits) "
                         "by minimising KL(bf16 teacher || alpha-scaled student) over the "
                         "run's own CALIBRATION text -- never test, never a benchmark. "
                         "Logits are exactly linear in that gain, so this is a pure "
                         "temperature match of the artifact to the unquantized model. "
                         "See pot/temp_match.py for the mechanism it targets.")
    ap.add_argument("--temp-windows", type=int, default=16,
                    help="calibration windows used by --temp-match (cost knob).")
    ap.add_argument("--scale-fp16", action="store_true",
                    help="store the per-(row,group) scale as float16, i.e. actually at "
                         "the precision `account_bpw` has always CHARGED for it. The "
                         "pipeline keeps scales in float32, so every bpw figure in this "
                         "project's 226 logged runs -- the incumbent's 4.244792 "
                         "included -- assumes a rounding nobody had applied. Turning "
                         "this on makes the artifact match its own accounting; it also "
                         "serves as the smallest available perturbation probe, at "
                         "~0.0007 octaves per scale, ~30x finer than the finest "
                         "--scale-round-log2 arm.")
    ap.add_argument("--bpw-budget", type=float, default=4.5)
    ap.add_argument("--tasks", default="ppl",
                    help="comma list: ppl,arc_easy,mc,kl. `mc` is the seven-task "
                         "multiple-choice PANEL (pot.eval_mc) used as the ranking "
                         "instrument; it deliberately excludes ARC-Easy, which is held "
                         "out as the test set.")
    ap.add_argument("--mc-extra", action="store_true",
                    help="also score pot.eval_mc.EXTRA_TASKS (MMLU). Reported under "
                         "mc['extra'] and EXCLUDED from panel_macro, which stays the "
                         "seven-task statistic every earlier arm was ranked on.")
    ap.add_argument("--mc-token-budget", type=int, default=4096,
                    help="for --tasks mc: padded-token budget per forward pass (bounds the [tokens x 151936] logit tensor). A pure "
                         "cost knob -- the batch plan is a function of tokenized lengths "
                         "only, so scores are identical at any budget that fits in "
                         "memory (asserted by tests/pot/test_eval_mc.py).")
    ap.add_argument("--mc-dump", default=None,
                    help="for --tasks mc: directory to write per-task, per-item, "
                         "per-choice log-prob tables into (<task>.npz, schema as "
                         "pot.eval_arc.dump_per_item). Purely additive; enables paired "
                         "item-level statistics between artifacts without re-quantizing.")
    ap.add_argument("--kl-windows", type=int, default=None,
                    help="for --tasks kl: evaluate the behavioural-fidelity KL on only "
                         "the first N wikitext-2 test windows (default: the whole split, "
                         "i.e. the same text the ppl eval uses). Purely a cost knob.")
    ap.add_argument("--kl-dump", default=None,
                    help="for --tasks kl: also write the raw per-token KL / teacher "
                         "entropy / margin vectors to this .npz, so conditional readouts "
                         "can be recomputed without re-quantizing. Purely additive.")
    ap.add_argument("--limit", type=int, default=None, help="ARC-Easy item limit")
    ap.add_argument("--arc-batch-size", type=int, default=16)
    ap.add_argument("--arc-dump", default=None,
                    help="also write the raw per-item, per-choice ARC-Easy log-prob "
                         "table to this .npz (see pot.eval_arc.dump_per_item). Purely "
                         "additive: the reported acc/acc_norm/acc_norm_token are "
                         "unchanged. Used for higher-power readouts (margin, choice-NLL, "
                         "Brier) and for item-level pairing across arms without a "
                         "re-quantize.")
    ap.add_argument("--verify-layers", default="0",
                    help="'all' or a comma list of decoder-layer indices to verify")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="debug: only quantize the first N decoder layers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--dry-run", action="store_true", help="skip evaluation")
    args = ap.parse_args()

    # Resolve the two spellings into ONE value that is what gets passed down AND what
    # gets recorded in pot_runs.jsonl. The default stays False (not "none") so a config
    # dumped by this driver still round-trips through the historical boolean.
    if args.scale_weight is not None and args.hess_weighted_scale:
        ap.error("--scale-weight and --hess-weighted-scale are mutually exclusive; "
                 "--hess-weighted-scale is just --scale-weight diag")
    if args.scale_weight is not None:
        args.hess_weighted_scale = {"none": False, "diag": True}.get(
            args.scale_weight, args.scale_weight)
    else:
        args.scale_weight = "diag" if args.hess_weighted_scale else "none"

    torch.manual_seed(args.seed)
    dev = args.device
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {args.model} ...")
    # NOTE: transformers is pinned to 4.55.0 here, which takes `torch_dtype=`, NOT `dtype=`.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # ---- incoherence processing (QuaRot / SpinQuant style), before anything else -----
    # Must happen BEFORE calibration capture so the Hessians, the scale search and the
    # GPTQ sweep all see the rotated coordinates. Equivalence-preserving: verified
    # unquantized with src/pot/check_rotation.py.
    rot_info = None
    if args.rotate != "none":
        rot_info = rotate_qwen3(model, mode=args.rotate, seed=args.seed, dev=dev,
                                untie_lm_head=args.rotate_untie_lm_head)

    sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
    from benchmark_suite import get_calibration_data
    log(f"calibration: {args.nsamples} x {args.seqlen} from {args.calib_dataset}"
        + (f" (offset {args.calib_offset})" if args.calib_offset else ""))
    # Ask for offset + nsamples sequences, then drop the first `offset`. The loader
    # chunks contiguously from the head of the split, so slicing this way yields draws
    # that are disjoint across offsets rather than merely reshuffled.
    want = args.nsamples + args.calib_offset
    calib = load_calibration_cached(get_calibration_data, tok, args.calib_dataset,
                                    want, args.seqlen, args.calib_cache)
    if calib.shape[0] < want:
        raise RuntimeError(
            f"calibration set has only {calib.shape[0]} sequences of {args.seqlen} tokens, "
            f"need {want} for --nsamples {args.nsamples} --calib-offset {args.calib_offset}")
    calib = calib[args.calib_offset: args.calib_offset + args.nsamples]
    assert calib.shape[0] == args.nsamples
    log(f"  calib tensor {tuple(calib.shape)}")

    # ------------------------------------------------------------------ codebook
    if args.log_scale_hess and not args.log_scale:
        raise SystemExit("--log-scale-hess weights the LOG-DOMAIN scale fit; it is "
                         "meaningless without --log-scale.")
    if args.log_scale and args.exact_scale:
        raise SystemExit("--log-scale and --exact-scale are two different OBJECTIVES "
                         "for the same quantity; pass exactly one (--exact-scale is "
                         "the exact squared-error fit, --log-scale the exact "
                         "log-domain fit).")
    if args.codebook_nexp is not None and (args.alphabet_select
                                           or args.codebook == "fit_add2"):
        raise SystemExit("--codebook-nexp is incompatible with --alphabet-select and "
                         "--codebook fit_add2")
    if args.alphabet_select:
        if (args.exact_scale or args.refine_codes or args.learn_scale
                or args.col_scale or args.log_scale):
            raise SystemExit("--alphabet-select is incompatible with --log-scale, "
                             "--exact-scale, "
                             "--refine-codes, --learn-scale, --col-scale")
        if args.equalize != "off":
            raise SystemExit("--alphabet-select is incompatible with --equalize")
        strides = [int(s) for s in args.alphabet_strides.split(",") if s.strip()]
        alpha_family = default_family(strides=tuple(strides))
        alpha_family = [c.to(device=dev, dtype=torch.float32) for c in alpha_family]
        sel_bits = selector_bits_for(alpha_family)
        cb = alpha_family[0]                     # for shift_free dispatch / logging
        code_bits = cb.bits()
        log(f"alphabet family: {[c.name for c in alpha_family]} "
            f"selector_bits={sel_bits}")
    elif args.codebook == "fit_add2":
        cb = fit_codebook_globally(model, args.group, dev)
        cb = cb.to(device=dev, dtype=torch.float32)
        code_bits = cb.bits()
    else:
        cb = get_codebook(args.codebook, nexp=args.codebook_nexp)
        cb = cb.to(device=dev, dtype=torch.float32)
        code_bits = cb.bits()
    log(f"codebook: {cb}  levels={[round(float(x), 6) for x in cb.levels]}")
    if args.log_scale or args.log_round:
        # The log-domain objective is undefined for an alphabet containing 0 (the
        # zero codeword has log-magnitude -inf). That is a real property of
        # int4_sym / nf4 / strict_pot_zero, so it fails loudly here rather than
        # silently degrading -- those books are linear-objective arms only.
        from pot.log_scale import log_magnitudes
        try:
            log_magnitudes(cb.levels)
        except ValueError as exc:
            raise SystemExit(f"--log-scale/--log-round on codebook {cb.name!r}: {exc}")

    group_map = parse_group_map(args.group_map)
    if group_map:
        log(f"group-map: {group_map} (per-linear group resolved by longest-suffix "
            f"match; --group {args.group} is ignored)")

    layers = get_decoder_layers(model)
    n_layers = len(layers) if args.max_layers is None else min(args.max_layers, len(layers))

    # embeddings / rotary need to be on the device for the prefix pass
    mm = getattr(model, "model", model)
    for name in ("embed_tokens", "rotary_emb", "norm"):
        sub = getattr(mm, name, None)
        if sub is not None:
            sub.to(dev)

    log("capturing layer-0 activations ...")
    inps, layer_kwargs = capture_layer0_inputs(model, layers, calib, dev, args.seqlen)
    log(f"  inps {tuple(inps.shape)}  kwargs={sorted(layer_kwargs)}")
    layer_kwargs = {k: (v.to(dev) if torch.is_tensor(v) else v)
                    for k, v in layer_kwargs.items()}

    # ------------------------------------------------- equalization absorption plan
    eq_plan = []
    eq_infos = []
    if args.equalize != "off":
        if args.col_scale:
            raise SystemExit("--equalize and --col-scale both fit a per-input-column "
                             "diagonal; they are mutually exclusive")
        eq_plan = qwen3_equalize_plan(model.config)
        for gs in eq_plan:
            log(f"equalize plan [{gs['key']}] members={gs['members']} "
                f"n_free={gs['n_free']} free={gs['free']} absorb-> {gs['absorb']}")

    # ----------------------------------------------------------------- the sweep
    n_weights = 0
    n_groups_total = 0
    all_scales = []
    n_col_scale_entries = 0
    n_bias_rows = 0
    n_bias_mats = 0
    rung_acc = {"n": 0, "n_bottom": 0, "n_top": 0,
                "dlog2_bottom_sum": 0.0, "dlog2_top_sum": 0.0}
    verify_targets = (set(range(n_layers)) if args.verify_layers == "all"
                      else {int(x) for x in args.verify_layers.split(",") if x != ""})
    verifications = []
    layer_stats = []
    learn_infos = []

    for li in range(n_layers):
        layer = layers[li].to(dev)
        lins = find_linears(layer)
        gptq_cls = PoTGPTQFamily if args.alphabet_select else PoTGPTQ
        gptqs = {n: gptq_cls(m) for n, m in lins.items()}

        handles = []
        def mk(name):
            def hook(mod, inp, out):
                gptqs[name].add_batch(inp[0])
            return hook
        for n, m in lins.items():
            handles.append(m.register_forward_hook(mk(n)))

        outs = torch.empty_like(inps)
        with torch.no_grad():
            for j in range(inps.shape[0]):
                x = inps[j:j + 1].to(dev, torch.bfloat16)
                outs[j:j + 1] = layer(x, **layer_kwargs)[0].to(inps.dtype).cpu()
        for h in handles:
            h.remove()

        # Y_ref for --learn-scope block: the block's output under the ORIGINAL weights,
        # which is exactly what the forward pass above just produced.
        Yref = outs.clone() if (args.learn_scale and args.learn_scope == "block") else None

        # ------------- activation-aware PoT column equalization (equalize.py) -------
        # Runs BEFORE the rtn identity-Hessian overwrite below, because the salience
        # h = diag((2/n) X^T X) is a property of the calibration ACTIVATIONS and is
        # wanted even when the ROUNDING uses no error feedback.
        eq_cs, eq_h = {}, {}
        for gs in eq_plan:
            members = [m for m in gs["members"] if m in gptqs]
            if not members:
                continue
            mats = [gptqs[m].W for m in members]
            grps = [resolve_group(m, group_map, args.group) for m in members]
            hd = None
            if args.eq_hess:
                hd = sum(torch.diag(gptqs[m].H) for m in members) / len(members)
            tie = None if gs["tie"] is None else gs["tie"].to(dev)
            c, info = equalize_columns(mats, cb, grps, hdiag=hd, mode=args.equalize,
                                       iters=args.eq_iters, tie=tie)
            for m in members:
                eq_cs[m] = c
                eq_h[m] = torch.ones_like(c) if hd is None else hd
            eq_infos.append({"layer": li, "key": gs["key"], "members": members,
                             "n_free": gs["n_free"], **info})
            if li in (0, n_layers - 1):
                log(f"  equalize[{gs['key']}] obj {info['obj_before']:.6g} -> "
                    f"{info['obj_after']:.6g} ({100 * info['obj_rel']:+.2f}%) "
                    f"log2c in [{info['log2c_min']:.3g}, {info['log2c_max']:.3g}]")

        layer_state = {}
        for name, g in gptqs.items():
            H_calib = None
            perm_h = None
            if args.method == "rtn":
                if args.refine_codes and args.refine_hessian == "calib":
                    # keep a reference to the REAL calibration Hessian before the
                    # quantizer's H is replaced by I; the initial code assignment is
                    # still pure RTN, only the refinement sees this H.
                    H_calib = g.H
                if args.log_scale_hess:
                    # The SALIENCE log-domain objective (§21.5). --method rtn replaces H
                    # with I, so no value of --hess-weighted-scale can express "weight
                    # the log-domain scale fit by real activation salience"; capture the
                    # true calibration Gram here, before the overwrite. Rounding stays
                    # pure RTN; only the shared group scale sees h_j.
                    H_calib = g.H
                if args.out_correct != "none" or args.solve_scale is not None:
                    # --out-correct / --solve-scale fit against the calibration
                    # activations. Without this capture it would see `H = I` and return
                    # the WEIGHT-space gain, which exact-scale stationarity pins at
                    # exactly 1 (SESSION 3.7) -- i.e. silently nothing. The artifact
                    # itself is still pure RTN; only the post-hoc correction sees H.
                    H_calib = g.H
                if args.order == "salogmean":
                    # --method rtn replaces H with I, which would make the salience term
                    # of the 'salogmean' key constant. The permutation is chosen from the
                    # REAL calibration diag(H), captured here; the ROUNDING is still pure
                    # RTN, only which columns share a group depends on calibration.
                    perm_h = torch.diag(g.H).clone()
                g.H = torch.eye(g.columns, device=g.dev)
                g.nsamples = 1
            grp = resolve_group(name, group_map, args.group)
            cs = None
            if args.alphabet_select:
                out = g.quantize(alpha_family, group=grp,
                                 percdamp=(0.0 if args.method == "rtn" else args.percdamp),
                                 blocksize=args.blocksize, act_order=args.act_order,
                                 order=args.order, rescale_in_loop=args.rescale,
                                 perm_hess=perm_h, perm_seed=args.perm_seed,
                                 pot_scale=args.pot_scale, n_grid=args.n_grid,
                                 lo=args.scale_lo, hi=args.scale_hi,
                                 hess_weighted_scale=args.hess_weighted_scale,
                                 feedback_damp=args.feedback_damp,
                                 feedback_clip=args.feedback_clip)
            else:
                if args.col_scale:
                    _, cs, _ = search_col_scale(g.W, cb, group=grp,
                                                hess_diag=torch.diag(g.H))
                sc_hess = None
                if name in eq_cs:
                    cs = eq_cs[name]
                    # W-space consistency: the sweep fits r against V = W/c, and an
                    # error e in V is an error e*c in W, so the per-column weight must
                    # carry the c^2 factor (see equalize.py).
                    sc_hess = eq_h[name] * cs ** 2
                if args.log_scale_hess:
                    # h_j ALONE: no w^2 factor (that is the whole point -- w^2 is what
                    # makes squared error blind to bottom-rung clamping), and no c^2
                    # factor either, because the log distortion log2|w_hat/w| is
                    # invariant to a per-column rescale, unlike the W-space MSE that
                    # needs equalize.py's c^2.
                    Hc = g.H if H_calib is None else H_calib
                    sc_hess = torch.diag(Hc).clone().float()
                out = g.quantize(cb, group=grp,
                                 percdamp=(0.0 if args.method == "rtn" else args.percdamp),
                                 blocksize=args.blocksize, act_order=args.act_order,
                                 order=args.order, rescale_in_loop=args.rescale,
                                 perm_hess=perm_h, perm_seed=args.perm_seed,
                                 col_scale=cs,
                                 pot_scale=args.pot_scale, n_grid=args.n_grid,
                                 lo=args.scale_lo, hi=args.scale_hi,
                                 hess_weighted_scale=args.hess_weighted_scale,
                                 exact_scale=args.exact_scale,
                                 scale_offset_log2=args.scale_offset_log2,
                                 scale_round_log2=args.scale_round_log2,
                                 scale_fp16=args.scale_fp16,
                                 log_scale=args.log_scale,
                                 log_round=args.log_round,
                                 feedback_damp=args.feedback_damp,
                                 feedback_clip=args.feedback_clip,
                                 scale_hess=sc_hess,
                                 return_targets=(args.learn_scale
                                                 and args.learn_retarget == "gptq"))

            if args.refine_codes:
                H_ref = g.H if H_calib is None else H_calib
                if args.refine_damp:
                    H_ref = H_ref.clone()
                    _d = args.refine_damp * torch.mean(torch.diag(H_ref))
                    H_ref[range(g.columns), range(g.columns)] += _d
                rinfo = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"],
                                     out["col_scale"], cb, H_ref,
                                     sweeps=args.refine_sweeps,
                                     trace=args.refine_trace)
                out["codes"] = rinfo["codes"]
                out["W_hat"] = rinfo["W_hat"]
                err = out["W_hat"] - g.W
                out["rel_mse"] = float((err ** 2).sum() / (g.W ** 2).sum().clamp_min(1e-30))
                out["rel_fro_err"] = float(err.norm() / g.W.norm().clamp_min(1e-30))
                out["proxy_err"] = rinfo["obj_after"]
                out["refine_n_moves"] = rinfo["n_moves"]
                out["refine_obj_before"] = rinfo["obj_before"]
                out["refine_obj_after"] = rinfo["obj_after"]
                out["refine_err2_before"] = rinfo["err2_before"]
                out["refine_err2_after"] = rinfo["err2_after"]
                out["refine_h_norm"] = rinfo["h_norm"]
                if args.refine_trace:
                    out["refine_obj_trace"] = rinfo.get("obj_trace")
                    out["refine_err2_trace"] = rinfo.get("err2_trace")
                    out["refine_moves_trace"] = rinfo.get("moves_trace")
                log(f"  refine-codes[{name}] H={args.refine_hessian} "
                    f"sweeps={args.refine_sweeps} moves={rinfo['n_moves']} "
                    f"obj {rinfo['obj_before']:.6g} -> {rinfo['obj_after']:.6g}"
                    + f" err2 {rinfo['err2_before']:.6g} -> {rinfo['err2_after']:.6g}"
                    + f" ratio {rinfo['obj_before'] / max(rinfo['err2_before'], 1e-30):.6g}"
                      f" -> {rinfo['obj_after'] / max(rinfo['err2_after'], 1e-30):.6g}"
                    + (f" objtrace={['%.6g' % v for v in rinfo['obj_trace']]}"
                       f" err2trace={['%.6g' % v for v in rinfo['err2_trace']]}"
                       if args.refine_trace else ""))

            if not args.alphabet_select:
                rs = rung_stats(out["codes"], cb.levels, out["W_hat"], g.W)
                for _k in rung_acc:
                    rung_acc[_k] += rs[_k]
                out["rung"] = rs

            if args.solve_scale is not None:
                from pot.scale_solve import solve_group_scales, solve_diagnostics
                gi = out["g_idx"]
                if gi is not None:
                    # The solver assumes each group is a CONTIGUOUS block of columns.
                    # `g_idx` is stored unconditionally; the trivial `arange // group`
                    # IS contiguous, so accept it and reject only a real permutation.
                    triv = (torch.arange(g.columns, device=gi.device) // grp)
                    if not torch.equal(gi.long(), triv.long()):
                        raise SystemExit(
                            "--solve-scale assumes contiguous column groups; it is "
                            "incompatible with a non-identity --order (g_idx is a real "
                            "permutation). Run with --order none.")
                Hs = (H_calib if H_calib is not None else g.H)
                lev = cb.levels.to(out["codes"].device)
                cvals = lev[out["codes"].long()]
                s_new = solve_group_scales(g.W, cvals, Hs, grp,
                                           mu=args.solve_scale, s0=out["scales"])
                ssd = solve_diagnostics(g.W, cvals, Hs, grp, out["scales"], s_new)
                out["scales"] = s_new
                out["W_hat"] = cvals * s_new.repeat_interleave(grp, dim=1)
                _e = out["W_hat"] - g.W
                out["rel_mse"] = float((_e ** 2).sum() / (g.W ** 2).sum().clamp_min(1e-30))
                out["rel_fro_err"] = float(_e.norm() / g.W.norm().clamp_min(1e-30))
                out["solve_scale"] = ssd
                if li in (0, n_layers - 1):
                    log(f"  solve-scale[{name}] mu={args.solve_scale} output err "
                        f"{100 * ssd['out_err_rel_change']:+.2f}%  "
                        f"mean|dlog2 s|={ssd['mean_abs_dlog2_scale']:.4f} oct")

            if args.out_correct != "none":
                # `g.H` may already have been replaced by the identity (--method rtn);
                # `xsum`/`ntokens` are accumulated separately and always hold the real
                # calibration statistics. The stored H is the running average
                # (2/n) X^T X, so the raw second moment is H * n/2 -- out_correct needs
                # H and N on the SAME footing to balance its bias term against its
                # quadratic term.
                Hs = (H_calib if H_calib is not None else g.H)
                Hsum = Hs * (g.ntokens / 2.0)
                xbar = (g.xsum / max(g.ntokens, 1)).to(torch.float32)
                a, bvec, ocd = output_correction(g.W, out["W_hat"], Hsum, xbar,
                                                 g.ntokens, mode=args.out_correct)
                if args.out_correct in ("gain", "gainmean", "both"):
                    # Fold into the stored scale and REBUILD W_hat from (codes, scale):
                    # that is the artifact the hardware computes, and rescaling the old
                    # W_hat instead would differ in the last fp32 bits.
                    out["scales"] = out["scales"] * a[:, None]
                    S = expand_scales(out["scales"], g.columns, group=grp,
                                      col_scale=out["col_scale"], g_idx=out["g_idx"])
                    out["W_hat"] = cb.levels.to(S.device)[out["codes"].long()] * S
                if args.out_correct in ("bias", "both"):
                    bq, bstep = quantize_bias(bvec, args.out_bias_bits)
                    lins[name].bias = torch.nn.Parameter(
                        bq.to(lins[name].weight.dtype))
                    n_bias_rows += bq.numel()
                    n_bias_mats += 1
                    ocd["bias_step"] = bstep
                    ocd["bias_bits"] = args.out_bias_bits
                err = out["W_hat"] - g.W
                out["rel_mse"] = float((err ** 2).sum() / (g.W ** 2).sum().clamp_min(1e-30))
                out["rel_fro_err"] = float(err.norm() / g.W.norm().clamp_min(1e-30))
                out["out_correct"] = ocd
                if li in (0, n_layers - 1):
                    log(f"  out-correct[{name}] {args.out_correct} removed "
                        f"{100 * ocd['frac_output_mse_removed']:.2f}% of layer output MSE "
                        f"(a in [{ocd['a_min']:.4f}, {ocd['a_max']:.4f}])")

            lins[name].weight.data = out["W_hat"].to(lins[name].weight.dtype)

            if args.scale_bits < 16:
                all_scales.append(out["scales"].detach().cpu())
            n_weights += out["W_hat"].numel()
            n_groups_total += out["W_hat"].shape[0] * out["n_groups"]
            if cs is not None and name not in eq_cs:
                n_col_scale_entries += cs.numel()
            if args.learn_scale:
                out["W_orig"] = g.W.clone()
                if args.learn_scope == "layer":
                    out["H"] = g.H.clone()
            layer_state[name] = out
            g.free()

        # ---------------- PoT-PTQ step 2: learned scale residual ----------------
        if args.learn_scale:
            info = learn_scale_residual(
                layer, X=inps, quant_state=layer_state, codebook=cb,
                scope=args.learn_scope, Y_ref=Yref, forward_kwargs=layer_kwargs,
                iters=args.learn_iters, lr=args.learn_lr, lam=args.learn_lambda,
                batch=args.learn_batch, retarget=args.learn_retarget,
                learn_col=args.learn_col, device=dev, verify=True, verbose=False)
            if args.learn_col:
                n_col_scale_entries += sum(o["W_hat"].shape[1]
                                           for o in layer_state.values())
            learn_infos.append({
                "layer": li,
                "block_err_before": info.get("block_err_before"),
                "block_err_after": info.get("block_err_after"),
                "loss_first": info.get("loss_first"), "loss_last": info.get("loss_last"),
                "verify_ok": info.get("verify_ok"),
                "linears": {k: {kk: vv for kk, vv in v.items() if kk != "loss_trace"}
                            for k, v in info["linears"].items()},
            })
            log(f"  learn-scale[{args.learn_scope}/{args.learn_retarget}] "
                f"loss {info.get('loss_first')} -> {info.get('loss_last')}  "
                f"verify={info.get('verify_ok')}")
            del Yref

        for name, out in layer_state.items():
            if li in verify_targets:
                ok, diag = verify_codebook_artifact(
                    out["W_hat"], out["scales"], cb, group=out["group"],
                    g_idx=out["g_idx"], col_scale=out["col_scale"])
                verifications.append({"layer": li, "linear": name, "ok": ok,
                                      "n_violations": diag.get("n_violations"),
                                      "kind": diag.get("kind"),
                                      "examples": diag.get("examples", [])[:2]})
                if ok is False:
                    log(f"  !! VERIFY FAILED layer {li} {name}: {diag['examples'][:2]}")
            st = {"layer": li, "linear": name,
                  "group": out["group"],
                  "rel_mse": out["rel_mse"],
                  "proxy_err": out["proxy_err"]}
            # format-native diagnostics (never affect the artifact; see log_scale.py)
            for k in ("clamp_lo_frac", "clamp_lo_relerr", "clamp_lo_octaves",
                      "clamp_hi_frac", "clamp_hi_relerr", "clamp_hi_octaves",
                      "bot_rung_frac", "top_rung_frac", "mean_abs_log2_err",
                      "log_mse", "mean_rel_err", "n_zero_weights",
                      "clamp_diag_error"):
                if k in out:
                    st[k] = out[k]
            if out.get("rung") is not None:
                st["rung"] = out["rung"]
            for k in ("refine_n_moves", "refine_obj_before", "refine_obj_after",
                      "refine_err2_before", "refine_err2_after", "refine_h_norm",
                      "refine_obj_trace", "refine_err2_trace", "refine_moves_trace"):
                if k in out:
                    st[k] = out[k]
            layer_stats.append(st)
        del layer_state
        torch.cuda.empty_cache()

        # re-run the layer with the QUANTIZED weights so the next layer's Hessian is
        # built on the real, error-carrying activations (sequential PTQ).
        with torch.no_grad():
            for j in range(inps.shape[0]):
                x = inps[j:j + 1].to(dev, torch.bfloat16)
                outs[j:j + 1] = layer(x, **layer_kwargs)[0].to(inps.dtype).cpu()
        inps = outs
        layers[li] = layer.cpu()
        torch.cuda.empty_cache()
        mrel = sum(s["rel_mse"] for s in layer_stats if s["layer"] == li) / len(lins)
        log(f"layer {li:2d}/{n_layers}  mean rel_mse={mrel:.5f}  "
            f"[{time.time() - t0:.0f}s]")

    # -------------------------------------------------------------------- bpw
    zero_bits = 0 if cb.shift_free != "none" or "sym" in cb.name else 0
    sel_bits_charged = selector_bits_for(alpha_family) if args.alphabet_select else 0
    codebook_entries = (sum(len(c) for c in alpha_family) if args.alphabet_select
                        else len(cb))
    # any non-identity column order needs the same stored per-column group index that
    # --act-order needs, so it is charged identically -- the orderings are bpw-neutral
    # against each other by construction.
    eff_order = args.order if args.order is not None else ("act" if args.act_order
                                                           else "none")
    eq_entries = sum(gs["n_free"] for gs in eq_plan) * n_layers
    g_idx_bpw = G_IDX_BPW_ACT_ORDER if eff_order != "none" else 0.0
    bpw, breakdown = account_bpw(
        n_weights=n_weights, n_groups_total=n_groups_total, n_rows_total=0,
        code_bits=code_bits, group=args.group, scale_bits=args.scale_bits,
        zero_bits=zero_bits, col_scale_bits=16,
        col_scale_entries=n_col_scale_entries, codebook_entries=codebook_entries,
        selector_bits=sel_bits_charged, g_idx_bpw=g_idx_bpw,
        eq_bits=args.eq_charge_bits, eq_entries=eq_entries,
        bias_bits=(args.out_bias_bits if args.out_correct in ("bias", "both") else 0),
        bias_entries=n_bias_rows, bias_mats=n_bias_mats)
    log(f"accounted bpw = {bpw:.6f}  {breakdown}")
    if eq_entries:
        log(f"equalize: {eq_entries} stored exponents would cost "
            f"{4 * eq_entries / n_weights:.6f} bpw at 4 bits, "
            f"{16 * eq_entries / n_weights:.6f} bpw at 16 bits; "
            f"charged {breakdown['equalize']:.6f} (absorbed)")

    rung = None
    if rung_acc["n"]:
        n = rung_acc["n"]
        rung = {
            "n_weights": n,
            "frac_bottom_rung": rung_acc["n_bottom"] / n,
            "frac_top_rung": rung_acc["n_top"] / n,
            "mean_abs_dlog2_bottom": (rung_acc["dlog2_bottom_sum"]
                                      / max(1, rung_acc["n_bottom"])),
            "mean_abs_dlog2_top": (rung_acc["dlog2_top_sum"]
                                   / max(1, rung_acc["n_top"])),
        }
        log(f"rung occupancy: bottom(2^-7) {100 * rung['frac_bottom_rung']:.4f}% "
            f"(mean |dlog2|w|| = {rung['mean_abs_dlog2_bottom']:.4f})  "
            f"top(2^0) {100 * rung['frac_top_rung']:.4f}% "
            f"(mean |dlog2|w|| = {rung['mean_abs_dlog2_top']:.4f})")

    scale_ok, scale_detail = check_scale_storable(all_scales, args.scale_bits)
    if not scale_ok:
        log(f"  !! SCALE STORAGE CHECK FAILED: --scale-bits {args.scale_bits} charges for a "
            f"constrained scale that the artifact does not have: {scale_detail}")

    verify_ok = all(v["ok"] is True for v in verifications) if verifications else None
    if cb.shift_free == "none":
        verify_ok = None
    bpw_ok = bpw <= args.bpw_budget

    # ------------------------------------------------------------------- eval
    temp_info = None
    if args.temp_match and not args.dry_run:
        from pot.temp_match import fit_temperature, apply_temperature
        from transformers import AutoModelForCausalLM as _AMC
        model.to(dev)
        log("temp-match: loading bf16 teacher ...")
        _teacher = _AMC.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        _teacher.eval(); _teacher.config.use_cache = False; _teacher.to(dev)
        _alpha, temp_info = fit_temperature(model, _teacher, calib, device=dev,
                                            n_windows=args.temp_windows)
        del _teacher
        torch.cuda.empty_cache()
        apply_temperature(model, _alpha)
        log(f"temp-match applied: alpha={_alpha:.5f} {temp_info}")
    ppl = arc = kl = mc = None
    if not args.dry_run:
        model.to(dev)
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        if "ppl" in tasks:
            from pot.eval_ppl import eval_wikitext_ppl
            ppl = eval_with_oom_retry(
                lambda: eval_wikitext_ppl(model, tok, seqlen=args.seqlen, device=dev),
                "ppl")
            log(f"wikitext2 ppl = {ppl:.4f}")
        if "arc_easy" in tasks:
            from pot.eval_arc import eval_arc_easy
            arc = eval_with_oom_retry(
                lambda: eval_arc_easy(model, tok, device=dev, limit=args.limit,
                                      batch_size=args.arc_batch_size,
                                      dump_path=args.arc_dump),
                "arc_easy")
            log(f"ARC-Easy = {arc}")
        if "mc" in tasks:
            from pot.eval_mc import eval_panel
            from pot.eval_mc import EXTRA_TASKS
            mc = eval_with_oom_retry(
                lambda: eval_panel(model, tok, device=dev,
                                   token_budget=args.mc_token_budget,
                                   dump_dir=args.mc_dump,
                                   extra_tasks=EXTRA_TASKS if args.mc_extra else ()),
                "mc")
            log(f"PANEL macro = {mc['panel_macro']:.4f}  micro = {mc['panel_micro']:.4f}"
                f"  ({mc['n_items']} items)")
        if "kl" in tasks:
            # Task-free behavioural fidelity against the unquantized model. Needs a
            # second copy of the network, so it is loaded here and dropped immediately;
            # the quantized `model` is never touched.
            from pot.eval_kl import eval_kl_fidelity
            from transformers import AutoModelForCausalLM as _AMC
            log("loading bf16 teacher for KL fidelity ...")
            teacher = _AMC.from_pretrained(args.model, torch_dtype=torch.bfloat16)
            teacher.eval()
            teacher.config.use_cache = False
            teacher.to(dev)
            kl = eval_kl_fidelity(model, teacher, tok, seqlen=args.seqlen, device=dev,
                                  n_windows=args.kl_windows, dump_path=args.kl_dump)
            del teacher
            torch.cuda.empty_cache()
            log(f"KL fidelity = {kl}")

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "config": {**vars(args), "codebook_levels": [float(x) for x in cb.levels],
                   "codebook_shift_free": cb.shift_free,
                   "code_bits": code_bits, "n_layers_quantized": n_layers},
        "rotation": rot_info,
        "bpw": bpw,
        "bpw_breakdown": breakdown,
        "bpw_budget": args.bpw_budget,
        "bpw_ok": bpw_ok,
        "scale_storage_ok": scale_ok,
        "scale_storage_detail": scale_detail,
        "n_weights_quantized": n_weights,
        "rung": rung,
        "equalize": ({"mode": args.equalize, "hess": args.eq_hess,
                      "iters": args.eq_iters,
                      "plan": [{k: v for k, v in gs.items() if k != "tie"}
                               for gs in eq_plan],
                      "entries_total": eq_entries,
                      "charged_bpw": breakdown.get("equalize", 0.0),
                      "hypothetical_bpw_4bit": 4 * eq_entries / n_weights,
                      "hypothetical_bpw_16bit": 16 * eq_entries / n_weights,
                      "per_group": eq_infos} if eq_plan else None),
        "verify": {"ok": verify_ok, "n_checked": len(verifications),
                   "detail": verifications},
        "ppl": ppl,
        "arc": arc,
        "mc": mc,
        "temp_match": temp_info,
        "kl": kl,
        "learn_scale": (learn_infos if args.learn_scale else None),
        "mean_layer_rel_mse": (sum(s["rel_mse"] for s in layer_stats) / len(layer_stats)
                               if layer_stats else None),
        "clamp_diag": ({k: sum(s.get(k, 0.0) for s in layer_stats) / len(layer_stats)
                        for k in ("clamp_lo_frac", "clamp_lo_relerr",
                                  "clamp_lo_octaves", "clamp_hi_frac",
                                  "clamp_hi_relerr", "clamp_hi_octaves",
                                  "bot_rung_frac", "top_rung_frac",
                                  "mean_abs_log2_err", "log_mse", "mean_rel_err")}
                       if layer_stats and "clamp_lo_frac" in layer_stats[0] else None),
        "layer_stats": layer_stats,
        "runtime_s": time.time() - t0,
        "success": bool(bpw_ok and scale_ok and (verify_ok is not False)),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")

    log("=" * 72)
    if verify_ok is False:
        log("RUN FAILED: shift-free verification did not pass. Results NOT valid.")
    if not bpw_ok:
        log(f"RUN FAILED: bpw {bpw:.4f} exceeds budget {args.bpw_budget}.")
    if not scale_ok:
        log(f"RUN FAILED: the bpw above charges {args.scale_bits} bits per group scale, but "
            f"the stored scales are not representable in that many bits. Either add "
            f"--pot-scale or account the scale at 16 bits.")
    log(f"codebook={args.codebook} g={args.group} method={args.method} "
        f"rescale={args.rescale} col_scale={args.col_scale} "
        f"learn_scale={args.learn_scale}"
        + (f" ({args.learn_scope}/{args.learn_retarget} it={args.learn_iters} "
           f"lr={args.learn_lr} lam={args.learn_lambda})" if args.learn_scale else ""))
    log(f"bpw={bpw:.4f} verify={verify_ok} ppl={ppl} arc={arc} "
        f"kl_fwd={(kl or {}).get('kl_fwd')} success={rec['success']}")
    return 0 if rec["success"] else 1


@torch.no_grad()
def fit_codebook_globally(model, group, dev, n_samples=400000, max_mats=40):
    """Fit the two-shift codebook ONCE, model-wide, on pooled group-normalized weights.

    It must be global and frozen: 16 fp16 entries amortized over the model is 0.00006 bpw,
    but a per-GROUP codebook would cost 256 bits per 128 weights = 2.0 bpw and blow the
    budget outright. See llmdocs/POT_METHOD.md, "Bits-per-weight budget".
    """
    samp, n = [], 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or "lm_head" in name:
            continue
        W = m.weight.data.float().reshape(-1, group)
        W = W / W.abs().amax(1, keepdim=True).clamp_min(1e-12)
        idx = torch.randperm(W.shape[0])[:2000]
        samp.append(W[idx].reshape(-1))
        n += 1
        if n >= max_mats:
            break
    x = torch.cat(samp).to(dev)
    x = x[torch.randperm(x.numel(), device=dev)[:n_samples]]
    cb = fit_add2_codebook(x, n=16, name="fit_add2")
    log(f"fitted global codebook from {x.numel()} samples: "
        f"{[round(float(v), 8) for v in cb.levels]}")
    return cb


if __name__ == "__main__":
    sys.exit(main())
