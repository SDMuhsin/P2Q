"""
Exact per-group scale search for a fixed scalar codebook.

Why the grid search in `scale_search.py` is an approximation of a finite problem
-------------------------------------------------------------------------------
`search_group_scale` samples `n_grid` multipliers of the scale on `[lo, hi]` and takes
the best. But the objective

        E(s) = sum_j h_j * (C[k(w_j, s)] * s - w_j)^2,        k(w_j, s) = nearest_codeword(w_j / s, C)

is **piecewise quadratic in s**: for a fixed group of weights `w_1..w_g` and a fixed
sorted codebook `C = [c_0 < c_1 < ... < c_{L-1}]`, the assignment `k(w_j, s)` of every
weight to a codeword is *locally constant* in `s` and only changes at finitely many
values of `s` -- the "breakpoints". Between two consecutive breakpoints the assignment
of every weight in the group is fixed, so `E(s)` restricted to that interval is an
honest quadratic in `s` (not merely locally, everywhere on the interval), and can be
minimized in closed form.

Breakpoint derivation
----------------------
`nearest_codeword(x, C)` snaps `x` to the codeword nearest it, i.e. it is a step
function of `x` that changes exactly at the midpoints of adjacent codewords,
`m_k = (c_k + c_{k+1}) / 2` for `k = 0 .. L-2` (this is literally how
`codebooks.nearest_codeword` is implemented: `bucketize(x, midpoints)`).

For one weight `w_j` (fixed) and the scale `s` (variable, `s > 0`), the *argument*
`w_j / s` sweeps monotonically as `s` moves, so `w_j / s` crosses the midpoint `m_k`
exactly once, at

        s_{j,k} = w_j / m_k                      (only a valid, positive breakpoint
                                                   when w_j and m_k have the same sign;
                                                   m_k == 0 crossings happen only "at
                                                   infinity" and need no explicit
                                                   breakpoint -- see below)

So a group of `g` weights against an `L`-level codebook has at most `g * (L - 1)`
breakpoints (some invalid/duplicate). Sort the ~`g*(L-1)` finite positive breakpoints
`b_1 <= b_2 <= ... <= b_N` for the whole group; on each of the `N + 1` open intervals
`(0, b_1), (b_1, b_2), ..., (b_N, inf)` every weight's assignment `k(w_j, s)` is
constant, hence so is the codeword `C[k(w_j,s)] =: c_j` used for every weight.

On such an interval the objective is exactly

        E(s) = A*s^2 - 2*B*s + K,      A = sum_j h_j * c_j^2,
                                        B = sum_j h_j * c_j * w_j,
                                        K = sum_j h_j * w_j^2   (constant, same on every
                                                                 interval, does not affect
                                                                 the argmin)

a textbook quadratic in `s`, minimized (ignoring the interval's endpoints) at
`s* = B / A`. The exact per-interval optimum is `s* ` clipped into the interval's
closed bounds (clipping can be optimal because a quadratic opening upward is monotone
outside its vertex). Evaluate `E` at that clipped point on every interval and take the
global minimum over all `N + 1` intervals: that is the EXACT minimizer of `E(s)` over
all `s > 0`, not a sample of it. No `--n-grid`, `--scale-lo`, `--scale-hi` needed.

This generalizes immediately to the Hessian-weighted objective used by
`hess_weighted_scale` in `gptq_pot.py` -- `h_j` there is `diag(H)_j`, a fixed per-column
weight, and none of the algebra above assumed `h_j == 1`.

Vectorization
-------------
For a `[B, g]` batch of groups (`B` = rows * n_groups), the breakpoint set has a fixed
size `g * (L-1)` for every row (invalid entries are masked to `+inf` so they sort to the
tail and are harmless -- they just produce a redundant, already-covered interval).
Building the `B x (g*(L-1)+1)` interval-representative array and calling
`nearest_codeword` once on it gives every interval's codeword assignment in one shot;
the closed-form quadratic is then pure elementwise arithmetic. For `g=64`, `L=16` that
is 961 candidates per row, i.e. about 5x the 200-point grid -- still trivial.

`pot_scale=True` (the "fully-shift dequant" ablation, scale itself constrained to a
power of two) is a genuinely different, discrete problem: `s` no longer ranges over a
continuum, so the breakpoint argument above does not apply. We instead directly evaluate
every power-of-two scale in a generous window around the absmax anchor and take the true
minimum over that window -- exact *within the window*, not proven exact over all integer
exponents, but the window is wide enough that anything outside it is a large clipping or
underflow error (see `_POT_HALF_WINDOW`). Since `pot_scale` is an ablation (never the
project's actual candidate), this asymmetry is noted, not hidden.
"""

from __future__ import annotations

import torch

from .codebooks import Codebook, nearest_codeword

__all__ = ["exact_group_scale", "exact_group_error"]

_POT_HALF_WINDOW = 12  # +-12 exponents around the absmax anchor; see module docstring.


def _levels(C):
    return C.levels if isinstance(C, Codebook) else C


def _prep_h(h, B, g, device):
    if h is None:
        return torch.ones(1, g, dtype=torch.float64, device=device)
    h = h.double().to(device)
    if h.dim() == 1:
        return h.view(1, g)
    return h.view(B, g)


@torch.no_grad()
def exact_group_error(W_group: torch.Tensor, C, s: torch.Tensor,
                      h: torch.Tensor | None = None) -> torch.Tensor:
    """Same contract as `scale_search.group_quant_error`, kept here so tests/dominance
    checks can compare exact-search output against the shared error definition without
    importing the grid module's internals."""
    C = _levels(C).to(W_group.device, torch.float64)
    W = W_group.double()
    hh = _prep_h(h, W.shape[0], W.shape[1], W.device)
    q = C[nearest_codeword(W / s.double(), C)] * s.double()
    return ((q - W) ** 2 * hh).sum(-1)


@torch.no_grad()
def _exact_pot_window(W: torch.Tensor, C: torch.Tensor, hh: torch.Tensor,
                      half_window: int = _POT_HALF_WINDOW):
    """Discrete exact-within-window search for pot_scale=True. W, C float64."""
    B, g = W.shape
    cmax = float(C.abs().max())
    absmax = W.abs().amax(1, keepdim=True).clamp_min(1e-300)          # [B,1]
    k0 = torch.round(torch.log2(absmax / cmax))
    offs = torch.arange(-half_window, half_window + 1, device=W.device,
                        dtype=torch.float64)
    s = torch.exp2(k0 + offs.view(1, -1))                             # [B, nc]
    ratio = W.unsqueeze(1) / s.unsqueeze(-1)                          # [B, nc, g]
    idx = nearest_codeword(ratio, C)
    Cv = C[idx]                                                       # [B, nc, g]
    e = ((Cv * s.unsqueeze(-1) - W.unsqueeze(1)) ** 2 * hh.unsqueeze(1)).sum(-1)  # [B,nc]
    k = e.argmin(1)
    best_s = s.gather(1, k[:, None])
    best_e = e.gather(1, k[:, None]).squeeze(1)
    return best_s, best_e


@torch.no_grad()
def exact_group_scale(W_group: torch.Tensor, C, hess_diag: torch.Tensor | None = None,
                      pot_scale: bool = False, max_elems: int = 2 ** 24,
                      eps: float = 1e-12):
    """Exact per-group scale minimizing (Hessian-weighted) MSE against codebook `C`.

    Signature mirrors `scale_search.search_group_scale` minus the grid hyperparameters
    (`n_grid`, `lo`, `hi`), which this function has no use for.

    Args:
        W_group:   [B, g] weights, one group per row.
        C:         Codebook or sorted 1-D level tensor.
        hess_diag: optional [g] or [B, g] per-column weights (the `h_j` above).
        pot_scale: constrain the returned scale to a power of two (see module doc --
                   this path is a wide discrete window search, not the breakpoint proof).
        max_elems: scratch-memory budget; rows are processed in chunks.

    Returns:
        (scale [B, 1], err [B]) in float32, matching `search_group_scale`'s dtype.
    """
    C = _levels(C).to(device=W_group.device, dtype=torch.float64)
    W = W_group.double()
    B, g = W.shape
    L = C.numel()

    hh = _prep_h(hess_diag, B, g, W.device)

    if pot_scale:
        rows = max(1, min(B, max_elems // max(1, g * (2 * _POT_HALF_WINDOW + 1))))
        best_s = torch.empty(B, 1, device=W.device, dtype=torch.float64)
        best_e = torch.empty(B, device=W.device, dtype=torch.float64)
        for i in range(0, B, rows):
            hb = hh if hh.shape[0] == 1 else hh[i:i + rows]
            s, e = _exact_pot_window(W[i:i + rows], C, hb)
            best_s[i:i + rows] = s
            best_e[i:i + rows] = e
        return best_s.float(), best_e.float()

    mids = (C[1:] + C[:-1]) * 0.5                                     # [L-1]
    nmid = L - 1
    if nmid == 0:
        # degenerate 1-level codebook: any positive scale ties; fall back to absmax.
        cmax = float(C.abs().max()) or 1.0
        s = (W.abs().amax(1, keepdim=True) / cmax).clamp_min(eps)
        e = exact_group_error(W, C, s, hh).float()
        return s.float(), e

    # The dominant scratch tensor is [rows, nmid*g + 1, g] (`ratio` / `Cv`), NOT
    # [rows, nmid+1]: the chunk heuristic used to divide by `g * (nmid + 1)` and so
    # under-counted the footprint by a factor of ~g, making the real peak ~8 GB per call
    # for g=64, L=16 at the default budget instead of the ~134 MB the module docstring
    # describes. Chunking is EXACT -- rows are independent and the returned scale/err are
    # bit-identical for any chunk size (asserted in tests/pot/test_equalize.py) -- so this
    # changes no result, only peak memory.
    rows_per_chunk = max(1, min(B, max_elems // max(1, (g * nmid + 1) * g)))
    best_s = torch.empty(B, 1, device=W.device, dtype=torch.float64)
    best_e = torch.empty(B, device=W.device, dtype=torch.float64)

    for i0 in range(0, B, rows_per_chunk):
        w = W[i0:i0 + rows_per_chunk]                                  # [b, g]
        hb = hh if hh.shape[0] == 1 else hh[i0:i0 + rows_per_chunk]
        b = w.shape[0]

        wj = w.unsqueeze(-1)                                            # [b, g, 1]
        m = mids.view(1, 1, nmid)
        same_sign = (wj * m) > 0
        cand = torch.where(same_sign, wj / m, torch.full_like(wj / m.clamp_min(1.0),
                                                              float("inf")))
        cand = cand.reshape(b, g * nmid)
        cand, _ = torch.sort(cand, dim=1)                               # [b, N], N=g*nmid

        first = cand[:, :1]
        last = cand[:, -1:]
        rep0 = torch.where(torch.isfinite(first), first * 0.5, torch.ones_like(first))
        interior = 0.5 * (cand[:, :-1] + cand[:, 1:]) if cand.shape[1] > 1 else cand[:, :0]
        repN = torch.where(torch.isfinite(last), last * 2.0, torch.ones_like(last))
        reps = torch.cat([rep0, interior, repN], dim=1)                 # [b, N+1]
        lo = torch.cat([torch.zeros_like(first), cand], dim=1)          # [b, N+1]
        hi = torch.cat([cand, torch.full_like(last, float("inf"))], dim=1)

        bad = ~torch.isfinite(reps) | (reps <= 0)
        reps_safe = torch.where(bad, torch.ones_like(reps), reps)

        ratio = w.unsqueeze(1) / reps_safe.unsqueeze(-1)                # [b, nr, g]
        idx = nearest_codeword(ratio, C)
        Cv = C[idx]                                                     # [b, nr, g]

        A = (hb.unsqueeze(1) * Cv * Cv).sum(-1)                         # [b, nr]
        Bc = (hb.unsqueeze(1) * Cv * w.unsqueeze(1)).sum(-1)            # [b, nr]
        K = (hb * w * w).sum(-1, keepdim=True)                          # [b, 1]

        safe_A = A.clamp_min(1e-300)
        s_star = torch.where(A > 1e-300, Bc / safe_A, lo.clamp_min(eps))
        lo_c = lo.clamp_min(eps)
        s_clip = torch.clamp(s_star, min=lo_c, max=hi)
        s_clip = torch.where(bad, torch.full_like(s_clip, float("nan")), s_clip)

        err = A * s_clip ** 2 - 2 * Bc * s_clip + K                     # [b, nr]
        err = torch.where(torch.isnan(err), torch.full_like(err, float("inf")), err)

        k = err.argmin(1)
        best_s[i0:i0 + rows_per_chunk] = s_clip.gather(1, k[:, None])
        best_e[i0:i0 + rows_per_chunk] = err.gather(1, k[:, None]).squeeze(1)

    return best_s.float(), best_e.float()
