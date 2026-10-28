#!/usr/bin/env python3
"""
MECHANISM probe for the "rotation inverts on a log alphabet" hypothesis.

For every quantized decoder linear, split each output row into groups of ``--group``
consecutive input channels (exactly how the quantizer groups weights) and look at the
within-group distribution of the LOG-MAGNITUDE

    z = log2(|w| / max|w|)        in (-inf, 0]

A strict-PoT 4-bit alphabet is {+-2^-r, r = 0..7}: eight octaves of RELATIVE
resolution.  Under absmax normalization the rung a weight lands on is
``r = clamp(round(-z), 0, 7)``; everything with ``z < -7.5`` is CLAMPED onto the
bottom rung and loses its magnitude entirely.

Reported per group, then averaged:
  sd_z          standard deviation of z  (how spread over octaves the group is)
  kl_unif       KL( rung occupancy || uniform over the 8 octaves ), nats.  0 = the
                alphabet's octaves are used evenly, which is what a log grid wants.
  frac_clamp    fraction of weights with z < -7.5 (clamped at the bottom rung)
  frac_top      fraction of weights on rung 0 (top rung -- where 27.3% of the
                squared error lives)
  max_over_rms  max|w| / rms(w), the quantity incoherence processing minimizes
  err_pot       relative squared error of nearest-rung strict-PoT rounding at
                scale = max|w|  (sum (w-q)^2 / sum w^2), no scale search
  err_int4      same for a symmetric 15-level uniform grid at scale = max|w|

Run with --rotate none / hadamard / random_hadamard to compare.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from pot.rotation import rotate_qwen3  # noqa: E402

SUFFIXES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

POT_LEVELS = torch.tensor([2.0 ** (-r) for r in range(8)], dtype=torch.float32)
INT4_LEVELS = torch.tensor([i / 7.0 for i in range(-7, 8)], dtype=torch.float32)


def _rel_err(w, levels):
    """w: (G, g) rows already divided by their own max|.|.  Nearest-level rounding."""
    q = levels[(w.unsqueeze(-1) - levels.view(1, 1, -1)).abs().argmin(-1)]
    return ((w - q) ** 2).sum().item() / max(((w ** 2).sum().item()), 1e-30)


def _rel_err_pot(w):
    """strict_pot is +-2^-r: sign is free, so round |w| in log space to a rung."""
    a = w.abs()
    r = torch.log2(a.clamp_min(1e-45)).neg().round().clamp(0, 7)
    q = torch.sign(w) * (2.0 ** (-r))
    return ((w - q) ** 2).sum().item() / max(((w ** 2).sum().item()), 1e-30)


def stats_for_matrix(W, group, chunk_rows=512):
    """W: (out, in) float32 on cpu. Returns dict of averages over groups."""
    out, inn = W.shape
    if inn % group:
        return None
    acc = dict(sd_z=0.0, kl_unif=0.0, frac_clamp=0.0, frac_top=0.0,
               max_over_rms=0.0, err_pot=0.0, err_int4=0.0, n_groups=0)
    hist_tot = torch.zeros(8, dtype=torch.float64)
    for i0 in range(0, out, chunk_rows):
        X = W[i0:i0 + chunk_rows].reshape(-1, group).float()          # (G, g)
        mx = X.abs().amax(1, keepdim=True).clamp_min(1e-30)
        Xn = X / mx
        a = Xn.abs().clamp_min(1e-45)
        z = torch.log2(a)                                             # (G, g) <= 0
        acc["sd_z"] += z.std(dim=1).sum().item()
        r = (-z).round().clamp(0, 7).long()
        acc["frac_clamp"] += (z < -7.5).float().mean(1).sum().item()
        acc["frac_top"] += (r == 0).float().mean(1).sum().item()
        # per-group rung histogram -> KL to uniform-over-8-octaves
        G = X.shape[0]
        h = torch.zeros(G, 8)
        h.scatter_add_(1, r, torch.ones_like(r, dtype=torch.float32))
        p = h / group
        kl = (p * (p.clamp_min(1e-12) * 8.0).log()).sum(1)            # KL(p||unif) nats
        acc["kl_unif"] += kl.sum().item()
        hist_tot += h.sum(0).double()
        rms = (X ** 2).mean(1).sqrt().clamp_min(1e-30)
        acc["max_over_rms"] += (mx.squeeze(1) / rms).sum().item()
        acc["err_pot"] += _rel_err_pot(Xn) * G
        acc["err_int4"] += _rel_err(Xn, INT4_LEVELS) * G
        acc["n_groups"] += G
    n = acc.pop("n_groups")
    outd = {k: v / n for k, v in acc.items()}
    outd["n_groups"] = n
    outd["rung_hist"] = (hist_tot / hist_tot.sum()).tolist()
    return outd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--rotate", default="none",
                    choices=["none", "hadamard", "random_hadamard"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--perm", default="none",
                    choices=["none", "sorted", "interleave"],
                    help="input-channel permutation applied AFTER any rotation. A fixed "
                         "permutation of the residual stream is orthogonal and fuses "
                         "into the previous layer's output weights, so it is exactly "
                         "absorbable and costs nothing at inference and nothing in bpw "
                         "(unlike --act-order, it is the SAME permutation for every "
                         "layer, so no per-column g_idx is stored). 'sorted' orders "
                         "input channels by descending column RMS, making each group of "
                         "64 magnitude-HOMOGENEOUS (narrow log spread). 'interleave' "
                         "snakes largest/smallest alternately, making each group "
                         "magnitude-HETEROGENEOUS (wide log spread). These push the "
                         "within-group log-magnitude distribution in OPPOSITE "
                         "directions and are the anti-Hadamard probe.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 device_map=args.dev)
    model.eval()
    if args.rotate != "none":
        rotate_qwen3(model, mode=args.rotate, seed=args.seed, dev=args.dev,
                     verbose=False)

    per_suffix = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            suf = next((s for s in SUFFIXES if name.endswith(s)), None)
            if suf is None:
                continue
            W = mod.weight.detach().cpu().float()
            if args.perm != "none":
                rms = (W ** 2).mean(0).sqrt()
                order = torch.argsort(rms, descending=True)
                if args.perm == "interleave":
                    n = order.numel()
                    hi, lo = order[: n // 2], order[n // 2:].flip(0)
                    order = torch.stack([hi, lo], dim=1).reshape(-1)
                W = W[:, order]
            st = stats_for_matrix(W, args.group)
            if st is None:
                continue
            per_suffix.setdefault(suf, []).append(st)

    keys = ["sd_z", "kl_unif", "frac_clamp", "frac_top", "max_over_rms",
            "err_pot", "err_int4"]
    agg = {}
    for suf, lst in per_suffix.items():
        tot = sum(s["n_groups"] for s in lst)
        agg[suf] = {k: sum(s[k] * s["n_groups"] for s in lst) / tot for k in keys}
        agg[suf]["n_groups"] = tot
        rh = torch.zeros(8, dtype=torch.float64)
        for s in lst:
            rh += torch.tensor(s["rung_hist"], dtype=torch.float64) * s["n_groups"]
        agg[suf]["rung_hist"] = (rh / rh.sum()).tolist()
    tot = sum(v["n_groups"] for v in agg.values())
    overall = {k: sum(v[k] * v["n_groups"] for v in agg.values()) / tot for k in keys}
    rh = torch.zeros(8, dtype=torch.float64)
    for v in agg.values():
        rh += torch.tensor(v["rung_hist"], dtype=torch.float64) * v["n_groups"]
    overall["rung_hist"] = (rh / rh.sum()).tolist()
    overall["n_groups"] = tot

    res = {"model": args.model, "rotate": args.rotate, "perm": args.perm,
           "seed": args.seed,
           "group": args.group, "overall": overall, "per_suffix": agg}
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
