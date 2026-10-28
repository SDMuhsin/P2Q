"""
The format-native distortion measure for a LOGARITHMIC alphabet, and its exact
per-group scale search.

Why squared error on weights is the wrong objective for a log grid
------------------------------------------------------------------
Every objective this project has optimized is a squared error on weights: GPTQ's
`tr(E H E^T)`, the Hessian-weighted scale search, `scale_search.py`'s grid and
`exact_scale.py`'s breakpoint minimization. All of them are blind to the single
failure mode a logarithmic alphabet has.

The decisive measurement: a 4-octave strict-PoT alphabet at group 16 costs bpw
4.000000 (CHEAPER than the 8-octave incumbent) and reaches rel_mse 0.026303
(BETTER than the incumbent's 0.028618) -- and collapses ARC-E by 29 pp, to
acc 0.3548 / acc_norm 0.3451, barely above the 0.25 chance floor. Verify passed.
Lower weight error, lower bpw, dead model.

The mechanism is clamping at the BOTTOM of the range. A 4-octave alphabet
represents nothing below `2^-3 * absmax`; at g=16 the group absmax is ~2 sigma,
so the bottom rung sits near 0.25 sigma. Clamping a weight there does not make a
small weight slightly wrong -- **it makes a small weight large**. Its RELATIVE
error is unbounded (a weight at 0.01 sigma becomes 0.25 sigma, a factor of 25)
while its SQUARED error contribution is negligible (0.24 sigma)^2 against an
absmax of 2 sigma. Squared error literally cannot see it. That is why rel_mse
improved while the model died.

The distortion the format actually induces
-------------------------------------------
A log alphabet quantizes in the log domain. Write `u = log2|w|`. Storing
`w_hat = sign(w) * 2^e * s` is exactly *rounding `u` to the lattice of available
log-magnitudes, offset by `t = log2 s`*, plus a sign (never lost: strict_pot has
no zero, so `sign(w_hat) == sign(w)` always). The natural distortion is therefore
RELATIVE:

        d(w, w_hat) = (log2|w_hat| - log2|w|)^2

and it charges a small weight clamped UP by three octaves exactly as much as a
large weight clamped DOWN by three octaves. That symmetry is the whole point;
MSE charges the first ~0 and the second everything.

This is not a reweighting of MSE. It is the induced metric of the format: the
alphabet is a uniform lattice in `u`, so `u`-space is where "round to nearest"
and "squared error" are the right primitives.

--------------------------------------------------------------------------------
THE EXACT LOG-DOMAIN SCALE SEARCH
--------------------------------------------------------------------------------
Let the alphabet's positive magnitudes be `m_1 < ... < m_p` with log-magnitudes
`a_k = log2 m_k` (for strict_pot(n): `a = 0, -1, ..., -(n-1)`, i.e. the integers).
Let `u_i = log2|w_i|` for the `g` weights of one group, `t = log2 s`.

Rounding in the log domain assigns weight `i` to the magnitude index

        k(i, t) = argmin_k ( (u_i - t) - a_k )^2

and the distortion is

        D(t) = sum_i h_i * ( (u_i - t) - a_{k(i,t)} )^2 .

For strict_pot, `a_k` are the consecutive integers `-(n-1) .. 0`, and the argmin
is literally `clip(round(u_i - t), -(n-1), 0)` -- an integer round, then a clip.
The clipped terms are NOT a separate case: they are the `k` at the ends of the
lattice winning the argmin, which is what happens when `u_i - t` falls outside
`[a_1, a_p]`. This is the structural reason the log objective is *easier* than the
squared-error one, not harder.

**Breakpoints.** `k(i, t)` is piecewise constant in `t` and changes exactly when
`u_i - t` crosses a LOG-MIDPOINT of two adjacent lattice points,

        t_{i,k} = u_i - (a_k + a_{k+1}) / 2,      k = 1 .. p-1 .

(For strict_pot these are `u_i - m` for the interior half-integers
`m in {-(n-1)+0.5, ..., -0.5}`: exactly `p - 1 = n - 1` of them per weight.)
There are no other breakpoints. In particular there is NO extra breakpoint at the
clip thresholds `a_1` and `a_p`: outside the lattice the argmin is already pinned
to the end index and stays pinned, so the assignment does not change there. This
is an EXACT treatment of clipping, with nothing approximated.

Sort the `g * (p-1)` breakpoints of the whole group. On each of the `g*(p-1) + 1`
open intervals between consecutive breakpoints every `k(i, t)` is constant, hence
so is every `a_{k(i,t)} =: alpha_i`, and on that interval

        D(t) = A t^2 - 2 B t + K,     A = sum_i h_i          (the SAME on every
                                      B = sum_i h_i (u_i - alpha_i)    interval),
                                      K = sum_i h_i (u_i - alpha_i)^2

a textbook upward quadratic minimized at the weighted mean

        t* = ( sum_i h_i (u_i - alpha_i) ) / ( sum_i h_i ) ,

clipped into the interval's closed bounds (a quadratic is monotone outside its
vertex, so the interval optimum is the clipped vertex). Evaluate `D` at that point
on every interval and take the global minimum: that is the EXACT minimizer of
`D(t)` over all real `t`. No grid, no gradient, no hyperparameter. `s = 2^t`.

**It is better behaved than the squared-error objective, provably.**
1. `D` is CONTINUOUS everywhere: at a breakpoint the two competing lattice points
   are equidistant in `u`, so the residual flips sign but `(.)^2` is unchanged.
2. Every piece has the SAME curvature `A = sum_i h_i > 0`, so `D` is a lower
   envelope of identical parabolas -- there is no ill-conditioning between pieces
   and no piece can be flat.
3. `D(t) -> +inf` as `t -> +-inf` (all weights clip, linearly in `t`), so the
   global minimum exists and is attained in a bounded region.
The grid search in `scale_search.py` exists because the SQUARED-error loss-vs-scale
curve on a log grid "is non-smooth with many local minima" (its docstring).
`count_local_minima` below measures both curves on the same data over the same
dense sampling. MEASURED (64 random N(0,1) groups, 2001 samples over +-3 octaves,
strict_pot8): mean strict local minima linear / log = 11.7 / 14.9 at g=16,
19.6 / 16.8 at g=64, 23.3 / 11.9 at g=128; on REAL Qwen3-1.7B weights
(layers.7.mlp.down_proj, 128 groups, 1201 samples) 16.0 / 14.3 at g=64 and
19.8 / 9.7 at g=128. So the log curve is NOT uniformly
smoother -- it is comparable at small g and roughly half as multi-modal at g=128.
Neither curve is unimodal; both need an exact method, which is why both exist.
The log objective's real advantages are (1)-(3) above, not unimodality, and this
is reported rather than asserted.

**Assignment: log-nearest is not linear-nearest.** Between adjacent codewords
`2^(e-1)` and `2^e`, linear-nearest puts the decision boundary at the ARITHMETIC
midpoint `1.5 * 2^(e-1)`; log-nearest puts it at the GEOMETRIC midpoint
`sqrt(2) * 2^(e-1) ~ 1.4142 * 2^(e-1)`. So log-nearest rounds UP over a band that
linear-nearest rounds down -- a hyperparameter-free 5.7% shift of every interior
boundary. `log_nearest_codeword` implements it as a `bucketize` against the
geometric midpoints, the exact analogue of `codebooks.nearest_codeword`.

**Zeros and sign.** strict_pot has no zero code, so the sign is always preserved
and `d` is well defined for every NONZERO weight. `log2|w| = -inf` for an exactly
zero weight; such a weight has infinite log distortion at every `t`, so it carries
no information about the optimal `t` and is excluded from the fit (its `h_i` is
set to 0) rather than being allowed to poison it with a NaN. `count_degenerate`
reports how many there are; on Qwen3-1.7B the answer is 0.

**`pot_scale=True`** (scale itself a power of two) constrains `t` to the integers.
The breakpoint argument gives the continuous optimum; the integer optimum is then
found by evaluating `D` on a wide integer window around it. Exact within the
window, as in `exact_scale.py`, and likewise an ablation only.
"""

from __future__ import annotations

import torch

from .codebooks import Codebook, nearest_codeword

__all__ = [
    "log_magnitudes", "log_boundaries", "log_nearest_codeword",
    "log_group_error", "log_group_scale", "clamp_diagnostics",
    "count_degenerate", "count_local_minima",
]

_POT_HALF_WINDOW = 12      # +-12 integer exponents, matching exact_scale.py
_TINY = 1e-30


def _levels(C):
    return C.levels if isinstance(C, Codebook) else C


# --------------------------------------------------------------------- alphabet
def log_magnitudes(C) -> torch.Tensor:
    """Sorted positive magnitudes of a zero-free, sign-symmetric alphabet.

    Raises if the alphabet contains 0 or is not symmetric: the log distortion is
    undefined for a zero codeword (int4_sym, nf4, strict_pot_zero), which is a
    real property of those formats and not something to paper over.
    """
    C = _levels(C).double().reshape(-1)
    if bool((C == 0).any()):
        raise ValueError(
            "log-domain distortion is undefined for an alphabet containing 0 "
            "(int4_sym / nf4 / strict_pot_zero); use the linear objective there.")
    pos = torch.sort(C[C > 0]).values
    neg = torch.sort(-C[C < 0]).values
    if pos.numel() != neg.numel() or not torch.allclose(pos, neg, rtol=1e-9, atol=0):
        raise ValueError("log-domain distortion needs a sign-symmetric alphabet; "
                         f"got positive magnitudes {pos.tolist()} vs {neg.tolist()}")
    return pos


def log_boundaries(C) -> torch.Tensor:
    """Decision boundaries of `log_nearest_codeword`, in the SAME order/shape as the
    `(C[1:]+C[:-1])/2` midpoints `codebooks.nearest_codeword` buckets on.

    Between two adjacent same-signed levels the boundary is the GEOMETRIC mean
    `sign * sqrt(|c_k c_{k+1}|)`; for a sign-crossing or zero-touching pair it falls
    back to the arithmetic mean (for a zero-free symmetric alphabet the single
    sign-crossing pair is `(-m_1, +m_1)` whose arithmetic mean is 0, which is also
    the correct log-domain answer: magnitude carries no information about sign).
    """
    C = _levels(C).double().reshape(-1)
    lo, hi = C[:-1], C[1:]
    same = (lo * hi) > 0
    geo = torch.sign(lo) * torch.sqrt((lo * hi).abs())
    ari = (lo + hi) * 0.5
    return torch.where(same, geo, ari)


def log_nearest_codeword(x: torch.Tensor, C: torch.Tensor):
    """Nearest codeword of `x` IN THE LOG DOMAIN: drop-in for `nearest_codeword`.

    Identical contract (sorted `C`, returns integer indices via `bucketize`), only
    the boundaries move from arithmetic to geometric midpoints.
    """
    b = log_boundaries(C).to(x.device, x.dtype)
    return torch.bucketize(x.contiguous(), b)


def count_degenerate(W: torch.Tensor) -> dict:
    """Weights for which `log2|w|` is not finite (exact zeros, and any inf/nan)."""
    a = W.detach().abs()
    return {"n": int(W.numel()),
            "n_zero": int((a == 0).sum()),
            "n_nonfinite": int((~torch.isfinite(W.detach())).sum()),
            "min_abs": float(a[a > 0].min()) if bool((a > 0).any()) else 0.0}


# ------------------------------------------------------------------- distortion
def _prep_h(h, B, g, device):
    if h is None:
        return torch.ones(1, g, dtype=torch.float64, device=device)
    h = h.double().to(device)
    return h.view(1, g) if h.dim() == 1 else h.view(B, g)


@torch.no_grad()
def log_group_error(W_group: torch.Tensor, C, s: torch.Tensor,
                    h: torch.Tensor | None = None,
                    log_round: bool = True) -> torch.Tensor:
    """`sum_i h_i (log2|w_hat_i| - log2|w_i|)^2` per group. Mirrors
    `exact_scale.exact_group_error`'s contract exactly (returns [B]).

    `log_round` selects which assignment rule produced `w_hat` (log-nearest, the
    rule this objective is stationary for, or linear-nearest). Zero weights are
    excluded (they have infinite log distortion under every scale).
    """
    Cl = _levels(C).to(W_group.device, torch.float64)
    W = W_group.double()
    hh = _prep_h(h, W.shape[0], W.shape[1], W.device).clone()
    sd = s.double()
    nc = log_nearest_codeword if log_round else nearest_codeword
    q = Cl[nc(W / sd, Cl)] * sd
    ok = (W != 0) & (q != 0)
    d = torch.where(ok, torch.log2(q.abs().clamp_min(_TINY))
                    - torch.log2(W.abs().clamp_min(_TINY)),
                    torch.zeros_like(W))
    hh = hh * ok.double()
    return (d * d * hh).sum(-1)


# --------------------------------------------------------------- the exact search
@torch.no_grad()
def log_group_scale(W_group: torch.Tensor, C, hess_diag: torch.Tensor | None = None,
                    pot_scale: bool = False, max_elems: int = 2 ** 24,
                    eps: float = 1e-12):
    """EXACT per-group scale minimizing the log-domain distortion `D(t)` above.

    Signature mirrors `exact_scale.exact_group_scale` (and therefore
    `scale_search.search_group_scale` minus the grid hyperparameters).

    Args:
        W_group:   [B, g] weights, one group per row.
        C:         Codebook or sorted 1-D level tensor. Must be zero-free and
                   sign-symmetric (see `log_magnitudes`).
        hess_diag: optional [g] or [B, g] per-column weights `h_i`.
        pot_scale: constrain `s` to a power of two (integer `t`); wide-window exact.
        max_elems: scratch-memory budget; rows are processed in chunks.

    Returns:
        (scale [B, 1], err [B]) in float32, `err` being the LOG distortion -- not
        comparable in units to the squared-error searches' `err`.
    """
    Cl = _levels(C).to(device=W_group.device, dtype=torch.float64)
    mags = log_magnitudes(Cl).to(W_group.device)            # [p]
    a = torch.log2(mags)                                    # [p] lattice in log space
    p = a.numel()

    W = W_group.double()
    B, g = W.shape
    hh = _prep_h(hess_diag, B, g, W.device)

    finite = (W != 0) & torch.isfinite(W)
    u = torch.where(finite, torch.log2(W.abs().clamp_min(_TINY)),
                    torch.zeros_like(W))                    # [B, g]
    hh = (hh * finite.double())                             # zero-weight rows drop out
    # a group that is entirely zero has no information; fall back to absmax.
    alive = hh.sum(-1, keepdim=True) > 0                    # [B, 1]

    if p == 1:                                              # degenerate 1-magnitude book
        s = (W.abs().amax(1, keepdim=True) / float(mags[0])).clamp_min(eps)
        return s.float(), log_group_error(W, Cl, s, hess_diag).float()

    mid = (a[:-1] + a[1:]) * 0.5                            # [p-1] log-midpoints
    nb = p - 1

    # Budget the ACTUAL scratch tensor, [b, nr, g] with nr = g*nb + 1, so `max_elems`
    # really is the element count of the largest allocation (exact_scale.py's budget
    # under-counts by a factor of g; here it does not).
    rows_per_chunk = max(1, min(B, max_elems // max(1, g * (g * nb + 1))))
    best_t = torch.empty(B, 1, device=W.device, dtype=torch.float64)
    best_e = torch.empty(B, device=W.device, dtype=torch.float64)

    for i0 in range(0, B, rows_per_chunk):
        uu = u[i0:i0 + rows_per_chunk]                       # [b, g]
        hb = hh if hh.shape[0] == 1 else hh[i0:i0 + rows_per_chunk]
        if hb.shape[0] == 1:
            hb = hb.expand(uu.shape[0], g)
        fin = finite[i0:i0 + rows_per_chunk]
        b = uu.shape[0]

        # breakpoints t_{i,k} = u_i - mid_k; dead weights contribute none (+inf).
        cand = uu.unsqueeze(-1) - mid.view(1, 1, nb)         # [b, g, nb]
        cand = torch.where(fin.unsqueeze(-1), cand,
                           torch.full_like(cand, float("inf")))
        cand = cand.reshape(b, g * nb)
        cand, _ = torch.sort(cand, dim=1)                    # [b, N]

        first, last = cand[:, :1], cand[:, -1:]
        rep0 = torch.where(torch.isfinite(first), first - 1.0, torch.zeros_like(first))
        interior = (0.5 * (cand[:, :-1] + cand[:, 1:]) if cand.shape[1] > 1
                    else cand[:, :0])
        repN = torch.where(torch.isfinite(last), last + 1.0, torch.zeros_like(last))
        reps = torch.cat([rep0, interior, repN], dim=1)      # [b, N+1]
        lo = torch.cat([torch.full_like(first, float("-inf")), cand], dim=1)
        hi = torch.cat([cand, torch.full_like(last, float("inf"))], dim=1)

        bad = ~torch.isfinite(reps)
        reps_safe = torch.where(bad, torch.zeros_like(reps), reps)

        # Assignment on each interval: nearest lattice point to (u_i - t).
        # Written to keep at most two [b, nr, g] tensors alive at once (`r` is
        # updated in place), matching exact_scale.py's memory profile -- these are
        # the largest scratch tensors in the whole pipeline.
        r = uu.unsqueeze(1) - reps_safe.unsqueeze(-1)        # [b, nr, g] = u_i - t
        k = torch.bucketize(r.contiguous(), mid.contiguous())
        r = r.contiguous()
        r -= a[k]                                            # residual (u_i - t) - alpha_i
        del k
        r += reps_safe.unsqueeze(-1)                         # -> (u_i - alpha_i)
        A = hb.sum(-1, keepdim=True)                         # [b, 1] (constant in nr)
        Bc = torch.einsum("brg,bg->br", r, hb)
        r.mul_(r)
        K = torch.einsum("brg,bg->br", r, hb)
        del r

        t_star = Bc / A.clamp_min(_TINY)
        t_clip = torch.clamp(t_star, min=lo, max=hi)
        t_clip = torch.where(bad, torch.zeros_like(t_clip), t_clip)

        err = A * t_clip ** 2 - 2 * Bc * t_clip + K
        err = torch.where(bad, torch.full_like(err, float("inf")), err)
        err = torch.where(torch.isnan(err), torch.full_like(err, float("inf")), err)

        j = err.argmin(1)
        best_t[i0:i0 + rows_per_chunk] = t_clip.gather(1, j[:, None])
        best_e[i0:i0 + rows_per_chunk] = err.gather(1, j[:, None]).squeeze(1)

    if pot_scale:
        offs = torch.arange(-_POT_HALF_WINDOW, _POT_HALF_WINDOW + 1,
                            device=W.device, dtype=torch.float64)
        cands = torch.round(best_t) + offs.view(1, -1)        # [B, nc]
        s_c = torch.exp2(cands)
        e = torch.stack([log_group_error(W, Cl, s_c[:, j:j + 1], hess_diag)
                         for j in range(s_c.shape[1])], dim=1)   # [B, nc]
        j = e.argmin(1)
        best_t = cands.gather(1, j[:, None])
        best_e = e.gather(1, j[:, None]).squeeze(1)

    s = torch.exp2(best_t)
    # fully-degenerate groups (all weights exactly zero): absmax fallback, no NaN.
    fallback = (W.abs().amax(1, keepdim=True) / float(mags[-1])).clamp_min(eps)
    s = torch.where(alive, s, fallback)
    return s.float(), best_e.float()


# ------------------------------------------------------------------ diagnostics
@torch.no_grad()
def clamp_diagnostics(W: torch.Tensor, W_hat: torch.Tensor, C,
                      codes: torch.Tensor | None = None) -> dict:
    """The diagnostic the whole thesis rests on.

    A weight sitting on the alphabet's SMALLEST magnitude with |w_hat| > |w| was
    clamped UP -- the failure mode squared error cannot see (its squared error is
    tiny, its RELATIVE error unbounded). One on the LARGEST magnitude with
    |w_hat| < |w| was clamped down, the failure mode squared error CAN see.

    `codes` (the stored indices into `C`) makes the rung test exact and
    scale-free; without it the rung is inferred from |w_hat| / (row max |w_hat|),
    which is only correct for one group per row.

    Returns fractions and MEAN RELATIVE errors for both ends, plus the log
    distortion itself, alongside the usual rel_mse so the trade is visible.
    """
    C = _levels(C).to(W.device, torch.float64)
    W = W.detach().double()
    Q = W_hat.detach().double()
    n = W.numel()
    aw, aq = W.abs(), Q.abs()
    mag = C.abs()
    mmin, mmax = float(mag.min()), float(mag.max())
    nz = aw > 0

    ratio = torch.where(nz & (aq > 0), aq / aw.clamp_min(_TINY), torch.ones_like(aw))
    l2 = torch.log2(ratio.clamp_min(_TINY))

    if codes is not None:
        lev = mag[codes.long()]
        on_bot = lev <= mmin * (1 + 1e-9)
        on_top = lev >= mmax * (1 - 1e-9)
    else:
        rowmax = aq.reshape(aq.shape[0], -1).amax(-1, keepdim=True).clamp_min(_TINY)
        rel_rung = (aq.reshape(aq.shape[0], -1) / rowmax).reshape(aq.shape)
        on_bot = rel_rung <= (mmin / mmax) * (1 + 1e-6)
        on_top = rel_rung >= 1.0 - 1e-9

    clamp_lo = on_bot & (aq > aw) & nz
    clamp_hi = on_top & (aq < aw) & nz
    relerr = (ratio - 1.0).abs()

    def _m(mask):
        k = int(mask.sum())
        return (k / n, float(relerr[mask].mean()) if k else 0.0,
                float(l2[mask].abs().mean()) if k else 0.0)

    f_lo, r_lo, o_lo = _m(clamp_lo)
    f_hi, r_hi, o_hi = _m(clamp_hi)
    return {
        "clamp_lo_frac": f_lo, "clamp_lo_relerr": r_lo, "clamp_lo_octaves": o_lo,
        "clamp_hi_frac": f_hi, "clamp_hi_relerr": r_hi, "clamp_hi_octaves": o_hi,
        "bot_rung_frac": float(on_bot.double().mean()),
        "top_rung_frac": float(on_top.double().mean()),
        "mean_abs_log2_err": float(l2[nz].abs().mean()) if bool(nz.any()) else 0.0,
        "log_mse": float((l2[nz] ** 2).mean()) if bool(nz.any()) else 0.0,
        "mean_rel_err": float(relerr[nz].mean()) if bool(nz.any()) else 0.0,
        "n_zero_weights": int((~nz).sum()),
    }


# ------------------------------------------------- curve-shape evidence (reporting)
@torch.no_grad()
def count_local_minima(W_group: torch.Tensor, C, n: int = 4001,
                       span: float = 3.0) -> dict:
    """Count strict local minima of the linear (MSE) and log objectives vs `log2 s`,
    on the same weights and the same dense sampling grid.

    This is the measurable version of `scale_search.py`'s claim that the
    squared-error loss-vs-scale curve on a log grid "is non-smooth with many local
    minima". Reported, not used by the pipeline.
    """
    from .exact_scale import exact_group_error
    Cl = _levels(C).to(W_group.device, torch.float64)
    W = W_group.double()
    base = torch.log2((W.abs().amax(1, keepdim=True)
                       / float(Cl.abs().max())).clamp_min(_TINY))
    ts = torch.linspace(-span, span, n, device=W.device, dtype=torch.float64)
    lin, lg = [], []
    for dt in ts:
        s = torch.exp2(base + dt)
        lin.append(exact_group_error(W, Cl, s))
        lg.append(log_group_error(W, Cl, s))
    L = torch.stack(lin, 1)                                  # [B, n]
    G = torch.stack(lg, 1)

    def _cnt(M):
        c = ((M[:, 1:-1] < M[:, :-2]) & (M[:, 1:-1] < M[:, 2:])).sum(1)
        return float(c.double().mean())
    return {"n_samples": n, "span_octaves": span,
            "linear_mse_local_minima": _cnt(L), "log_local_minima": _cnt(G)}
