"""
Exact least-squares re-solve of the per-(row, group) scales, with the CODES FROZEN.

The argument
------------
Every intervention that has failed on this format MOVES WEIGHTS TO NEW CODEWORDS:

    GPTQ (per-element retargeting)      ARC-E -7.91 pp, panel -1.49 pp   (2026-09-12)
    refine_codes (discrete descent)     ARC-E -6.51 pp, t = -4.84
    rotation (changes everything)       ARC-E -32.45 pp (PoT), -18.06 (int4)
    h-weighted scale search             ARC-E -8.5 to -11 pp

The one H-fitted intervention that leaves every code bit-identical and only RESCALES is
the one that does not fail -- `--out-correct gain` gave ARC-E +3.24 pp at zero extra
bits, the largest gain of the session. So the working hypothesis is not "RTN is optimal"
but:

    **The codes RTN picks are right. What is left to optimise is the SCALE.**

`--out-correct gain` explores only a ONE-dimensional slice of that: it multiplies a whole
row by a single `a_r`, i.e. it constrains `s_{r,g} = a_r * s_{r,g}^MSE`. The scales are
per (row, group) -- 32 free numbers per row at g=64 -- and they are ALREADY STORED, so
re-choosing their VALUES costs exactly zero extra bits.

The problem is exactly solvable
-------------------------------
Freeze the codes. For output row `r`, the layer's output is

    y_r(x) = sum_g  s_{r,g} * (c_{r,g} . x_g)

which is **linear in the 32-vector s_r**. Minimising the calibration-weighted output
error `E_x[(y_r(x) - w_r . x)^2]` is therefore an ordinary least squares in 32 unknowns:

    A s = b,     A_gh = c_{r,g}^T H_{g,h} c_{r,h},     b_g = c_{r,g}^T (H w_r)_g

with `H = sum_t x_t x_t^T` the calibration Gram. `A` is a Gram matrix of the group
partial-sums, hence PSD, and `G x G` is 32 x 32 -- a direct Cholesky solve per row.

This is "GPTQ for the scales instead of the codes": it optimises the part of the artifact
that is free and currently chosen myopically (each group's scale is set by its OWN local
MSE, blind to every other group in the row), and it leaves untouched the part that the
evidence says is already right.

Regularisation
--------------
22M scales fitted against a Gram estimated from 128 x 2048 tokens will overfit, and
overfitting the calibration Hessian is this project's most reliable failure mode. The
solve is therefore ridged TOWARD the incumbent MSE-optimal scale `s0` rather than toward
zero:

    (A + lam I) s = b + lam * s0,      lam = mu * mean(diag(A))

so `mu -> 0` is the unregularised output-error optimum and `mu -> inf` returns the
incumbent exactly. `mu` is the single hyperparameter and `mu = inf` is a provable no-op,
which makes the default inert and gives the sweep a known endpoint.
"""

from __future__ import annotations

import torch


def _plan_row_chunk(row_chunk, in_dim, G, dev, elem=8, slack=3.0, frac=0.35):
    """Largest row block whose float64 temporaries fit in a fraction of free VRAM.

    The dominant allocation in the solve is `cH` (shape [R, G, G, gs], i.e. R*in_dim*G
    elements) plus a comparable temporary inside the two einsums. At the historical
    fixed `row_chunk=512` that is 2.4 GB for Qwen3-1.7B `down_proj` (in=6144, G=96) and
    17 GB for gemma-2b `down_proj` (in=16384, G=256) -- the first OOMed every arm of the
    2026-09-12 --solve-scale validation on a shared GPU, the second cannot fit at all.
    The maths is row-separable, so shrinking R is exact: it changes nothing but peak
    memory. `frac` leaves headroom for the other tenants on a shared device.
    """
    per_row = max(1, int(in_dim * G * elem * slack))
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        budget = max(int(free * frac), 64 << 20)
    else:
        budget = 2 << 30
    return int(max(1, min(row_chunk, budget // per_row)))


@torch.no_grad()
def solve_group_scales(W, codes_val, H, group, mu=1.0, s0=None, eps=1e-12,
                       row_chunk=512):
    """Re-solve per-(row, group) scales with codes frozen.

    Args:
        W:         [out, in] original weights (float32).
        codes_val: [out, in] the CODEWORD VALUE assigned to each weight, i.e.
                   `levels[codes]` -- NOT the integer index, and not `W_hat`.
        H:         [in, in] calibration Gram (any positive multiple).
        group:     columns per scale group; `in` must be divisible by it.
        mu:        ridge strength toward `s0`, in units of mean(diag(A)).
        s0:        [out, G] incumbent scales to ridge toward. Required.
    Returns:
        [out, G] the re-solved scales.
    """
    out_dim, in_dim = W.shape
    assert in_dim % group == 0, (in_dim, group)
    G = in_dim // group
    dev = W.device
    Wd = W.double()
    Cd = codes_val.double()
    Hd = H.to(dev, torch.float64)
    assert s0 is not None, "solve_group_scales needs the incumbent scales to ridge toward"
    S0 = s0.double()

    outS = torch.empty(out_dim, G, device=dev, dtype=torch.float64)
    Hb = Hd.view(G, group, G, group)
    eyeG = torch.eye(G, device=dev, dtype=torch.float64)

    def _block(r0, r1):
        c = Cd[r0:r1].reshape(r1 - r0, G, group)                 # [R, G, gs]
        w = Wd[r0:r1]                                            # [R, in]

        # A[R, G, G] = sum_{i in g, j in h} c[r,g,i] H[g,h][i,j] c[r,h,j]
        # Build by blocks of H; H is [in, in] so H.view(G, gs, G, gs).
        # cH[R, G(g), G(h), gs(j)] = sum_i c[r,g,i] * Hb[g,i,h,j]
        cH = torch.einsum("rgi,gihj->rghj", c, Hb)
        A = torch.einsum("rghj,rhj->rgh", cH, c)                 # [R, G, G]
        del cH
        # b[R, G] = sum_{i in g} c[r,g,i] * (H w_r)[i]
        Hw = w @ Hd                                              # [R, in]
        b = torch.einsum("rgi,rgi->rg", c, Hw.view(r1 - r0, G, group))
        del Hw

        d = A.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(eps)  # [R]
        lam = (mu * d).view(-1, 1, 1)
        Areg = A + lam * eyeG
        breg = b + (mu * d).view(-1, 1) * S0[r0:r1]
        # Cholesky with a jitter fallback: A is PSD but can be singular when a group's
        # codes are all zero (possible under strict_pot_zero).
        try:
            L = torch.linalg.cholesky(Areg)
            return torch.cholesky_solve(breg.unsqueeze(-1), L).squeeze(-1)
        except Exception:
            return torch.linalg.lstsq(Areg, breg.unsqueeze(-1)).solution.squeeze(-1)

    # Row blocks are independent, so R is a pure memory/speed knob: plan it from the
    # free VRAM, then halve on any OOM the plan did not anticipate (a shared GPU can
    # lose memory to another tenant between the plan and the allocation). Results are
    # bit-identical for every R -- each row's normal equation is solved on its own.
    chunk = _plan_row_chunk(row_chunk, in_dim, G, dev)
    r0 = 0
    while r0 < out_dim:
        r1 = min(r0 + chunk, out_dim)
        try:
            outS[r0:r1] = _block(r0, r1)
        except torch.cuda.OutOfMemoryError:
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            continue
        r0 = r1
    return outS.float()


@torch.no_grad()
def solve_diagnostics(W, codes_val, H, group, s_old, s_new):
    """Output-error reduction achieved, and how far the scales moved."""
    Wd, Cd, Hd = W.double(), codes_val.double(), H.double()
    G = s_old.shape[1]
    # Chunked over output rows for the same reason the solve is: the full residual
    # `e` is [out, in] float64 and `e @ H` is another of the same, which is 4.3 GB on
    # gemma-2b's down_proj. The sum over rows is exact regardless of the block size.
    _rows = _plan_row_chunk(512, W.shape[1], 1, W.device, elem=8, slack=4.0)

    def err(s):
        sd = s.double().repeat_interleave(group, dim=1)
        tot = 0.0
        for r0 in range(0, Wd.shape[0], _rows):
            e = Cd[r0:r0 + _rows] * sd[r0:r0 + _rows] - Wd[r0:r0 + _rows]
            tot += float(torch.einsum("oi,ij,oj->", e, Hd, e))
        return tot

    e0, e1 = err(s_old), err(s_new)
    dl = (torch.log2(s_new.double().clamp_min(1e-30))
          - torch.log2(s_old.double().clamp_min(1e-30)))
    return {
        "out_err_before": e0, "out_err_after": e1,
        "out_err_rel_change": (e1 - e0) / max(abs(e0), 1e-30),
        "mean_abs_dlog2_scale": float(dl.abs().mean()),
        "max_abs_dlog2_scale": float(dl.abs().max()),
        "frac_scales_moved_gt_0.01oct": float((dl.abs() > 0.01).float().mean()),
    }
