"""
Is diag(H) an adequate stand-in for H on a LOG grid? -- the cheap diagnostic.

THE QUESTION
------------
Output error is  dy = E x,  so  E[||dy||^2] = tr(E H E^T)  with H = E[x x^T].
EVERY activation-aware method in this project uses only the DIAGONAL of H, which is
exact iff the cross terms vanish:

    tr(E H E^T) = sum_j H_jj ||E[:,j]||^2  +  sum_{j != k} H_jk <E[:,j], E[:,k]>
                  \________ the diagonal ________/   \_______ what is dropped _______/

So the whole question is one number per layer,

    rho = tr(E H E^T) / tr(E diag(H) E^T),

with rho == 1 exactly when the dropped term is zero.

WHY IT SHOULD DIFFER BETWEEN FORMATS
------------------------------------
On a UNIFORM grid the rounding residual of column j is ~ i.i.d. uniform on
[-Delta/2, Delta/2] and essentially independent of the weight, so `<E[:,j], E[:,k]>`
is a sum of independent zero-mean products and the cross terms very nearly cancel.

On a LOG grid they should not. Two structural reasons, both from the format:
  (1) all `g` columns of a group SHARE one scale `s`, and the relative error of a
      weight is a DETERMINISTIC function of frac(log2|w/s|) -- so within the unit
      that shares a parameter the errors are a deterministic pattern, not noise;
  (2) every CLAMPED weight is displaced to the same +-2^-(n-1) s, sign-preserving.
      That is a systematic, nearly rank-structured perturbation, and it is exactly
      the population where the log-grid error is largest.

PREDICTION (pre-registered in scripts/pot_log_domain.sh, addendum 2): strict_pot's
|rho - 1| is substantially LARGER than int4_sym's, at matched grouping -- and larger
still when restricted to the clamped population. If it is not, the derivation is dead
and the activation-aware question closes.

THE CONTROL THAT MAKES IT INTERPRETABLE
---------------------------------------
`rho` can move for two reasons: because the error entries are CORRELATED across
columns, or merely because their magnitudes happen to sit on channels with big H_jk.
So every rho is reported alongside `rho_signflip`: the same error tensor with each
entry's SIGN randomized (magnitudes, and hence the entire diagonal term, preserved
exactly). Sign-randomizing destroys any cross-column correlation while changing
nothing else, so `rho_signflip` is what rho WOULD be under the independence
assumption. The quantity that carries the claim is therefore

    excess = |rho - 1|  -  |rho_signflip - 1| .

REUSE
-----
The quantizer definitions `q_pot` / `q_int4` and the full-Gram accumulation are taken
verbatim from the rotation lane's `src/pot/full_hess.py` so the numbers are directly
comparable to its measured context (tr(EHE)/tr(EE) = 0.569 strict_pot vs 0.836
int4_sym: RTN error is naturally ANTI-aligned with curvature, about twice as strongly
on the log grid). They are copied rather than imported because `full_hess.py` runs
argparse at module scope and cannot be imported without executing.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, "/workspace/CoBALT/src")
sys.path.insert(0, "/workspace/CoBALT")
sys.path.insert(0, "/workspace/CoBALT/benchmarks")

SUF = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def q_pot(Xg):                                   # verbatim from full_hess.py
    s = Xg.abs().amax(1, keepdim=True).clamp_min(1e-30)
    Xn = Xg / s
    r = torch.log2(Xn.abs().clamp_min(1e-45)).neg().round().clamp(0, 7)
    return torch.sign(Xn) * (2.0 ** (-r)) * s


def q_int4(Xg):                                  # verbatim from full_hess.py
    s = Xg.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return (Xg / s * 7).round().clamp(-7, 7) / 7 * s


def bottom_rung_mask_pot(Xg):
    """Weights that landed on the BOTTOM rung 2^-7 of the strict-PoT ladder."""
    s = Xg.abs().amax(1, keepdim=True).clamp_min(1e-30)
    Xn = Xg / s
    r = torch.log2(Xn.abs().clamp_min(1e-45)).neg().round().clamp(0, 7)
    return r >= 7


def bottom_rung_mask_int4(Xg):
    """INT4's analogous under-range population: weights rounding to the smallest
    NONZERO code +-1/7 or to the zero code. Both are 'the bottom of the ladder'."""
    s = Xg.abs().amax(1, keepdim=True).clamp_min(1e-30)
    k = (Xg / s * 7).round().clamp(-7, 7).abs()
    return k <= 1


def block_mask(n, g, device):
    """[n, n] boolean: True inside the g x g diagonal blocks (the unit that shares a
    scale), False outside."""
    idx = torch.arange(n, device=device) // g
    return idx[:, None] == idx[None, :]


def ratios(E, H, Hd, gen, n_flip=8):
    """rho, and the sign-flip null distribution, for one error tensor E [out, in].

    The null is averaged over `n_flip` independent sign draws and its sd reported:
    a single draw has run-to-run scatter comparable to the effect being measured,
    so one draw is not a control.
    """
    diag = float((Hd * (E * E).sum(0)).sum())
    if diag <= 0:
        return None
    full = float((E @ H * E).sum())
    fl = []
    for _ in range(n_flip):
        sgn = (torch.randint(0, 2, E.shape, device=E.device, generator=gen,
                             dtype=torch.int8).float() * 2 - 1)
        Er = E * sgn
        diag_r = float((Hd * (Er * Er).sum(0)).sum())      # identical by construction
        assert abs(diag_r - diag) <= 1e-3 * abs(diag) + 1e-12
        fl.append(float((Er @ H * Er).sum()) / diag)
    m = sum(fl) / len(fl)
    sd = (sum((x - m) ** 2 for x in fl) / max(1, len(fl) - 1)) ** 0.5
    return {"rho": full / diag, "rho_signflip": m, "rho_signflip_sd": sd,
            "tr_diag": diag, "tr_full": full}


@torch.no_grad()
def block_scale_probe(W, H, group, n_rows=256, n_grid=1601, span=2.0, seed=0):
    """Would a BLOCK-Hessian group objective change the artifact at all?

    The derived method replaces the group-scale objective `sum_j h_j e_j^2` with
    `e_g^T H_gg e_g`, H_gg the g x g diagonal block -- the honest output error
    restricted to the unit that actually shares the parameter. (The exact
    breakpoint argument SURVIVES this change: on an interval where the assignment
    is fixed, e_g = c*s - w is affine in s, so e^T H e = s^2 c^T H c - 2 s c^T H w
    + w^T H w is still an exact quadratic, and the breakpoints depend only on the
    assignment map, which does not move. So no bracketed search is needed.)

    Before building that, measure whether it can matter: for real (row, group)
    pairs, find argmin of the BLOCK objective and of the DIAGONAL objective on the
    same dense scale sweep, and report how far apart they land and how much
    objective the block optimum actually buys over the diagonal optimum.
    """
    dev = W.device
    g = group
    O, I = W.shape
    G = I // g
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rows = torch.randint(0, O, (n_rows,), device=dev, generator=gen)
    grps = torch.randint(0, G, (n_rows,), device=dev, generator=gen)

    lv = torch.tensor([2.0 ** -i for i in range(8)], device=dev, dtype=torch.float64)
    lv = torch.cat([-lv.flip(0), lv])                                  # strict_pot8

    d_rel, gap, moved = [], [], 0
    d_bu, d_du = [], []
    mult = torch.exp2(torch.linspace(-span, span, n_grid, device=dev,
                                     dtype=torch.float64))
    for r, gi in zip(rows.tolist(), grps.tolist()):
        w = W[r, gi * g:(gi + 1) * g].double()                         # [g]
        Hb = H[gi * g:(gi + 1) * g, gi * g:(gi + 1) * g].double()      # [g, g]
        hd = torch.diag(Hb)
        base = w.abs().max().clamp_min(1e-30)
        s = (base * mult).view(-1, 1)                                  # [n, 1]
        q = lv[torch.bucketize((w.view(1, -1) / s).contiguous(),
                               ((lv[1:] + lv[:-1]) * 0.5).contiguous())] * s
        e = q - w.view(1, -1)                                          # [n, g]
        ob = ((e @ Hb) * e).sum(1)                                     # block objective
        od = ((e * e) * hd.view(1, -1)).sum(1)                         # diagonal objective
        ou = (e * e).sum(1)                                            # plain weight MSE
        ib, idx, iu = int(ob.argmin()), int(od.argmin()), int(ou.argmin())
        if ib != idx:
            moved += 1
        d_rel.append(abs(float(torch.log2(mult[ib] / mult[idx]))))
        d_bu.append(abs(float(torch.log2(mult[ib] / mult[iu]))))
        d_du.append(abs(float(torch.log2(mult[idx] / mult[iu]))))
        # what the block optimum buys, measured in the BLOCK objective
        gap.append(float((ob[idx] - ob[ib]) / ob[idx].clamp_min(1e-300)))
    n = len(gap)
    return {"n_probed": n, "frac_scale_moved": moved / n,
            "dlog2_block_vs_unweighted": sum(d_bu) / n,
            "dlog2_diagH_vs_unweighted": sum(d_du) / n,
            "mean_abs_dlog2_scale": sum(d_rel) / n,
            "max_abs_dlog2_scale": max(d_rel),
            "mean_block_obj_gain": sum(gap) / n, "max_block_obj_gain": max(gap)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-probe", action="store_true",
                    help="also probe whether a block-Hessian group objective would "
                         "move the scale at all (see block_scale_probe)")
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--layers", default="0,9,18,27")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = {int(x) for x in a.layers.split(",")}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark_suite import get_calibration_data

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                                 torch_dtype=torch.bfloat16).to(dev)
    model.eval(); model.config.use_cache = False
    calib = get_calibration_data(tok, a.nsamples, a.seqlen, "wikitext2")

    Hs, hooks = {}, []
    def mk(n):
        def h(m, i, o):
            x = i[0].detach().reshape(-1, i[0].shape[-1]).float()
            Hs[n] = Hs.get(n, torch.zeros(x.shape[1], x.shape[1], device=x.device)) + x.T @ x
        return h
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and any(n.endswith(s) for s in SUF):
            if int(n.split("layers.")[1].split(".")[0]) in want:
                hooks.append(m.register_forward_hook(mk(n)))
    with torch.no_grad():
        for i in range(calib.shape[0]):
            model(calib[i:i + 1].to(dev))
    for h in hooks:
        h.remove()

    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    rows = []
    with torch.no_grad():
        for n, m in model.named_modules():
            if n not in Hs:
                continue
            W = m.weight.detach().float()
            H = Hs[n]
            Hd = torch.diag(H).clamp_min(0)
            bmask = block_mask(H.shape[0], a.group, H.device)
            sh = W.shape
            Wg = W.reshape(-1, a.group)
            row = {"layer": n, "suffix": next(s for s in SUF if n.endswith(s)),
                   "in_features": sh[1], "out_features": sh[0]}
            for kind, fn, bm in (("pot", q_pot, bottom_rung_mask_pot),
                                 ("int4", q_int4, bottom_rung_mask_int4)):
                E = (fn(Wg) - Wg).reshape(sh)
                r = ratios(E, H, Hd, gen)
                row[f"rho_{kind}"] = r["rho"]
                row[f"rhoflip_{kind}"] = r["rho_signflip"]
                row[f"rhoflip_sd_{kind}"] = r["rho_signflip_sd"]
                # ---- WHERE does the cross term live? Within the unit that shares a
                # scale (a g x g diagonal block of H -- capturable by a block-Hessian
                # group objective), or between groups (not capturable by any per-group
                # method at all)? This is what decides whether the derived method is
                # even reachable.
                rb = ratios(E, H * bmask, Hd, gen)
                row[f"rho_blk_{kind}"] = rb["rho"]
                row[f"rhoflip_blk_{kind}"] = rb["rho_signflip"]
                row[f"rhoflip_blk_sd_{kind}"] = rb["rho_signflip_sd"]
                # ---- restricted to the bottom-rung / under-range population ----
                mask = bm(Wg).reshape(sh)
                Ec = E * mask
                row[f"botfrac_{kind}"] = float(mask.float().mean())
                rc = ratios(Ec, H, Hd, gen)
                row[f"rho_bot_{kind}"] = None if rc is None else rc["rho"]
                row[f"rhoflip_bot_{kind}"] = None if rc is None else rc["rho_signflip"]
                # ---- and to the complement, as an internal control ----
                Eo = E * (~mask)
                ro = ratios(Eo, H, Hd, gen)
                row[f"rho_rest_{kind}"] = None if ro is None else ro["rho"]
                row[f"rhoflip_rest_{kind}"] = None if ro is None else ro["rho_signflip"]
            if a.block_probe:
                row["block_probe"] = block_scale_probe(W, H, a.group)
            rows.append(row)
            del H, Hs[n]
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"group": a.group, "nsamples": a.nsamples, "rows": rows},
              open(a.out, "w"), indent=1)

    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return sum(v) / len(v) if v else float("nan")
    print(f"\n{'metric':22} {'strict_pot':>12} {'int4_sym':>12}")
    print("-" * 48)
    for lab, kp, ki in (("rho (all)", "rho_pot", "rho_int4"),
                        ("rho signflip (all)", "rhoflip_pot", "rhoflip_int4"),
                        ("rho (bottom rung)", "rho_bot_pot", "rho_bot_int4"),
                        ("rho signflip (bot)", "rhoflip_bot_pot", "rhoflip_bot_int4"),
                        ("rho (rest)", "rho_rest_pot", "rho_rest_int4"),
                        ("rho signflip (rest)", "rhoflip_rest_pot", "rhoflip_rest_int4"),
                        ("rho within-group blk", "rho_blk_pot", "rho_blk_int4"),
                        ("rho blk signflip", "rhoflip_blk_pot", "rhoflip_blk_int4"),
                        ("  (blk null sd)", "rhoflip_blk_sd_pot", "rhoflip_blk_sd_int4"),
                        ("  (all null sd)", "rhoflip_sd_pot", "rhoflip_sd_int4"),
                        ("bottom-rung frac", "botfrac_pot", "botfrac_int4")):
        print(f"{lab:22} {mean(kp):12.6f} {mean(ki):12.6f}")
    # Paired per-linear z against each linear's OWN sign-flip null: the pooled means
    # above hide that the null has real scatter, so the claim is made per linear and
    # then aggregated, never on one pooled number.
    print()
    for scope, suf in (("whole matrix", ""), ("within-group block", "_blk")):
        for k in ("pot", "int4"):
            z = [(r[f"rho{suf}_{k}"] - r[f"rhoflip{suf}_{k}"])
                 / max(r[f"rhoflip{suf}_sd_{k}"], 1e-12) for r in rows]
            n_pos = sum(1 for v in z if v > 2)
            print(f"paired z [{scope:18} {k:4}]: mean {sum(z)/len(z):+7.2f}  "
                  f"median {sorted(z)[len(z)//2]:+7.2f}  "
                  f"{n_pos}/{len(z)} linears with z > 2")
    if a.block_probe:
        bp = [r["block_probe"] for r in rows]
        print("\nBLOCK-HESSIAN GROUP OBJECTIVE -- would it move the artifact?")
        for k in ("frac_scale_moved", "mean_abs_dlog2_scale", "max_abs_dlog2_scale",
                  "dlog2_block_vs_unweighted", "dlog2_diagH_vs_unweighted",
                  "mean_block_obj_gain", "max_block_obj_gain"):
            print(f"  {k:24} {sum(b[k] for b in bp) / len(bp):.6g}")
    print()
    for lab, kp, ki in (("EXCESS |rho-1| all", "rho_pot", "rho_int4"),
                        ("EXCESS |rho-1| bot", "rho_bot_pot", "rho_bot_int4"),
                        ("EXCESS |rho-1| blk", "rho_blk_pot", "rho_blk_int4")):
        fp = kp.replace("rho_", "rhoflip_"); fi = ki.replace("rho_", "rhoflip_")
        ep = abs(mean(kp) - 1) - abs(mean(fp) - 1)
        ei = abs(mean(ki) - 1) - abs(mean(fi) - 1)
        print(f"{lab:22} {ep:12.6f} {ei:12.6f}")
    # the split that decides whether a per-group method can reach the cross terms
    for k in ("pot", "int4"):
        tot = mean(f"rho_{k}") - 1.0
        wit = mean(f"rho_blk_{k}") - 1.0
        print(f"cross-term split [{k:4}]: total {tot:+.6f} = within-group {wit:+.6f} "
              f"+ between-group {tot - wit:+.6f}"
              + (f"  ({100 * wit / tot:.1f}% reachable per-group)" if abs(tot) > 1e-9
                 else ""))


if __name__ == "__main__":
    main()
