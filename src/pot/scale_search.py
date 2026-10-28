"""
Scale fitting for a fixed scalar codebook.

Why a grid search and not a closed form
---------------------------------------
For a uniform codebook the per-group scale has a near-closed form (absmax, or a short
MSE line-search) because the assignment `round(w/s)` is piecewise-constant in s with
*small, local* jumps. On a log grid the levels are a factor of two apart, so a change in
s of a few percent can move a weight a whole exponent, and the loss-vs-scale curve is
non-smooth with many local minima. PoT-PTQ (arXiv:2507.11959) sweeps 200 multipliers for
exactly this reason; gradient or bisection methods are unreliable here. We do the same,
fully vectorized over groups.

All searches here minimize
        sum_j  h_j * (W_hat[.,j] - W[.,j])^2
with optional per-column weights `h_j`. Setting `h = diag(H)` from the calibration Gram
makes the objective the (diagonal approximation of the) OUTPUT error rather than plain
weight MSE, which is the quantity GPTQ itself minimizes.
"""

from __future__ import annotations

import math

import torch

from .codebooks import Codebook, nearest_codeword

__all__ = ["search_group_scale", "search_col_scale", "pot_round", "group_quant_error"]


def _levels(C):
    return C.levels if isinstance(C, Codebook) else C


def pot_round(s: torch.Tensor) -> torch.Tensor:
    """Snap a scale to the nearest power of two: 2^round(log2 s).

    For the "fully-shift dequant" ablation: with a PoT scale, dequantization itself is
    an exponent add, so nothing in the pipeline multiplies. RAPQ (IJCAI'22) shows naive
    per-layer PoT scales are catastrophic without a global search, so this is an
    ablation, never a default.
    """
    out = torch.zeros_like(s)
    pos = s > 0
    out[pos] = torch.exp2(torch.round(torch.log2(s[pos])))
    return out


@torch.no_grad()
def group_quant_error(Wg: torch.Tensor, C: torch.Tensor, s: torch.Tensor,
                      h: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted squared error of quantizing `Wg` [B, g] with per-group scale `s` [B, 1]."""
    q = C[nearest_codeword(Wg / s, C)] * s
    e = (q - Wg) ** 2
    if h is not None:
        e = e * h
    return e.sum(-1)


@torch.no_grad()
def search_group_scale(W_group: torch.Tensor, C, n_grid: int = 200,
                       lo: float = 0.3, hi: float = 1.2,
                       hess_diag: torch.Tensor | None = None,
                       pot_scale: bool = False,
                       max_elems: int = 2 ** 24):
    """Per-group multiplicative grid search for the scale minimizing group MSE.

    Args:
        W_group:   [B, g] weights, one group per row (B = out_features * n_groups).
        C:         Codebook or sorted 1-D level tensor.
        n_grid:    number of multipliers (PoT-PTQ uses 200).
        lo, hi:    multiplier range, relative to the absmax scale `absmax / max|C|`.
        hess_diag: optional [g] or [B, g] per-column weights for the error.
        pot_scale: constrain the returned scale to a power of two.
        max_elems: scratch-memory budget; the grid is evaluated in row chunks.

    Returns:
        (scale [B, 1], err [B]) -- the best scale and its weighted squared error.

    The grid ALWAYS contains the multiplier 1.0, so the result is never worse than the
    absmax initialization. (With pot_scale=True that guarantee is against the *rounded*
    absmax scale, which is the correct reference for that ablation.)
    """
    C = _levels(C).to(device=W_group.device, dtype=torch.float32)
    W = W_group.float()
    B, g = W.shape
    cmax = float(C.abs().max())

    base = (W.abs().amax(1, keepdim=True) / cmax).clamp_min(1e-12)       # [B, 1]
    mult = torch.linspace(lo, hi, n_grid, device=W.device, dtype=torch.float32)
    mult = torch.cat([mult, torch.ones(1, device=W.device)])            # guarantee m=1
    mult, _ = torch.sort(mult)
    n_grid = mult.numel()

    h = hess_diag
    if h is not None:
        h = h.float().to(W.device)
        if h.dim() == 1:
            h = h.view(1, 1, g)
        else:
            h = h.view(B, 1, g)

    rows = max(1, min(B, max_elems // max(1, n_grid * g)))
    best_s = torch.empty(B, 1, device=W.device, dtype=torch.float32)
    best_e = torch.empty(B, device=W.device, dtype=torch.float32)
    for i in range(0, B, rows):
        w = W[i:i + rows]                                                # [c, g]
        s = base[i:i + rows] * mult[None, :]                             # [c, n]
        if pot_scale:
            s = pot_round(s)
        s3 = s[:, :, None]                                               # [c, n, 1]
        q = C[nearest_codeword(w[:, None, :] / s3, C)] * s3
        e = (q - w[:, None, :]) ** 2
        if h is not None:
            e = e * (h if h.shape[0] == 1 else h[i:i + rows])
        e = e.sum(-1)                                                    # [c, n]
        k = e.argmin(1)
        best_s[i:i + rows] = s.gather(1, k[:, None])
        best_e[i:i + rows] = e.gather(1, k[:, None]).squeeze(1)
    return best_s, best_e


@torch.no_grad()
def search_col_scale(W: torch.Tensor, C, r: torch.Tensor | None = None,
                     group: int = 128, n_grid: int = 33, span: float = 1.0,
                     iters: int = 4, pot_scale: bool = False,
                     hess_diag: torch.Tensor | None = None):
    """Per-input-column multiplicative scale `c`, fitted by alternating exact grid search
    against the per-(row, group) scale `r`.

    W_hat[i, j] = level * r[i, g(j)] * c[j].  `c` is folded into the activation at GEMM
    time (x' = c * x, N multiplies for a whole matmul), so it costs no per-weight work.

    Both subproblems are separable -- given `c`, each (row, group) scale is independent;
    given `r`, each column scale is independent -- so each half-step is an EXACT grid
    minimization, and the alternation is monotone in the objective. The offset grid is
    multiplicative, spanning 2^-span .. 2^+span, and halves each iteration.

    STATUS: this is an ablation, not a default. Measured gain on strict PoT is 0.9%
    relMSE (see llmdocs/POT_METHOD.md, "the column scale is a dead end"): a 4-bit log grid
    is already at its intrinsic lattice-coarseness floor, so sliding it barely helps.

    Args:
        W:  [O, I] weight.  r: optional [O, I//group] init (absmax if None).
        hess_diag: optional [I] per-column error weights.
    Returns:
        (r [O, G], c [I], rel_err float)
    """
    C = _levels(C).to(device=W.device, dtype=torch.float32)
    W = W.float()
    O, I = W.shape
    assert I % group == 0, f"in_features {I} not divisible by group {group}"
    G = I // group
    cmax = float(C.abs().max())

    if r is None:
        r = (W.reshape(O, G, group).abs().amax(-1) / cmax).clamp_min(1e-12)
    r = r.float().clone()
    c = torch.ones(I, device=W.device, dtype=torch.float32)
    h = None if hess_diag is None else hess_diag.float().to(W.device).view(1, I)

    def _full(r_, c_):
        rr = r_.repeat_interleave(group, dim=1)
        if pot_scale:
            rr = pot_round(rr)
        return rr * c_[None, :]

    def _err(s):
        e = (C[nearest_codeword(W / s, C)] * s - W) ** 2
        return e if h is None else e * h

    sp = span
    for _ in range(iters):
        offs = torch.exp2(torch.linspace(-sp, sp, n_grid, device=W.device))
        # --- row-group half-step (exact given c) ---
        E = torch.stack([_err(_full(r * o, c)).reshape(O, G, group).sum(-1)
                         for o in offs], -1)                              # [O, G, n]
        r = r * offs[E.argmin(-1)]
        # --- column half-step (exact given r) ---
        E = torch.stack([_err(_full(r, c * o)).sum(0) for o in offs], -1)  # [I, n]
        c = c * offs[E.argmin(-1)]
        sp *= 0.5

    s = _full(r, c)
    rel = float(_err(s).sum() / (W ** 2).sum().clamp_min(1e-30))
    return r, c, rel
