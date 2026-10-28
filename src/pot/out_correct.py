"""
Post-hoc per-output-channel gain and bias.

What it is
----------
After the artifact is fixed -- codes chosen, scales chosen, nothing about the
quantization revisited -- ask for each output row r of a linear:

    minimise over (a_r, b_r):   E_x [ ( a_r * (W_hat_r . x) + b_r  -  W_r . x )^2 ]

`a_r` folds into the stored per-(row, group) fp16 scale (`s -> a_r * s`), so it costs
**zero extra bits** and leaves every code bit-identical: `W_hat = codes * scale` still
holds exactly, so C1's shift-free verification is untouched. `b_r` is genuinely new
storage and is charged.

Why this is not the sixth failed activation-aware lever
-------------------------------------------------------
Five calibration-driven interventions have failed on this format (GPTQ feedback,
`--scale-weight diag`/`ldl`, h-weighted equalization, the salience-log objective), and
SESSION 22.2 gives them one mechanism: strict-PoT's RTN error is naturally ANTI-aligned
with the Hessian (`tr(EHE)/tr(EE) = 0.569` against int4's 0.836), and every one of those
methods works by REDIRECTING the error -- GPTQ inflates ||E|| by 43% to buy alignment --
which destroys the alignment it was exploiting.

This does not redirect anything. No code moves; the per-weight error pattern is exactly
the one the exact-scale MSE search chose. `a` and `b` STRICTLY REDUCE the output error
of a fixed artifact -- the objective above is a two-parameter least squares whose optimum
is never worse than (a, b) = (1, 0). Whether a strict reduction in layer output error
converts into accuracy is precisely the question the project's flat-basin result
(SESSION 4b) says is open, and it has never been asked of a correction that leaves the
codes alone.

SESSION 3.7 measured these corrections at 22.9% and 14.2% of layer output MSE on plain
RTN, then dismissed them because under GPTQ's feedback they were worth 0.083% and 0.113%.
The incumbent is now RTN. Nobody re-measured them.

CORRECTION TO SESSION 3.7's ALGEBRA, recorded because the same slip is easy to repeat.
3.7 proves `gain = 1 - rel_mse` and concludes the least-squares gain is "1 exactly". That
is a WEIGHT-SPACE identity: exact-scale stationarity gives <W_hat, W_hat> = <W_hat, W>
per group, so the UNWEIGHTED gain is 1. The gain that matters at inference is the
H-weighted one, `(u H v^T)/(u H u^T)`, and no such identity constrains it. Measured on
Qwen3-1.7B it is 1.001-1.023 and removes 0.5-27% of layer output MSE. A per-row gain is
therefore NOT redundant with the per-group scales, contrary to what the weight-space
identity suggests.

The algebra
-----------
For one row, with `u = W_hat_r`, `v = W_r`, `H = sum_t x_t x_t^T`, `xbar = mean_t x_t`
and `N` samples, the joint optimum is

    Sigma = H - N * xbar xbar^T            (the centred scatter)
    a*    = (u Sigma v^T) / (u Sigma u^T)
    b*    = v . xbar - a* * (u . xbar)

and the objective at (1, 0) -- the uncorrected artifact -- is `(u-v) H (u-v)^T`. Gain-only
(`b = 0`) is `a = (u H v^T)/(u H u^T)`; bias-only (`a = 1`) is `b = (v - u) . xbar`.

All three need only `H` and `xbar`, never the activations themselves.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def output_correction(W, W_hat, H, xbar, nsamples, mode="both", eps=1e-30):
    """Per-output-row gain/bias for a fixed artifact.

    Args:
        W, W_hat: [out, in] original and quantized weights (float32).
        H:        [in, in] calibration Gram, `sum_t x_t x_t^T`.
        xbar:     [in] calibration mean activation, `mean_t x_t`.
        nsamples: number of rows that went into `H` / `xbar`.
        mode:     "gain" | "gainmean" | "bias" | "both" | "none".
                  "gainmean" is a single scalar for the entire matrix (see below).

    Returns:
        (a [out], b [out], diag) -- `a` is all-ones and `b` all-zeros for the parts the
        mode switches off, so the caller applies both unconditionally.
    """
    out_dim = W.shape[0]
    dev, dt = W.device, torch.float64
    W64, Wh64 = W.to(dt), W_hat.to(dt)
    H64, xb = H.to(dev, dt), xbar.to(dev, dt)
    N = float(nsamples)

    e = Wh64 - W64
    base = torch.einsum("oi,ij,oj->o", e, H64, e)                 # [out] uncorrected

    a = torch.ones(out_dim, device=dev, dtype=dt)
    b = torch.zeros(out_dim, device=dev, dtype=dt)
    n_fallback = 0

    if mode == "gainmean":
        # ONE scalar for the whole matrix: the granularity H* actually calls for.
        #
        # The per-ROW gain has `out_dim` free parameters fitted from the calibration
        # Gram, and SESSION 22.2's mechanism for five prior activation-aware failures is
        # that fitting many parameters against H REDIRECTS the weight error and destroys
        # the anti-alignment strict PoT gets for free. A single scalar per matrix cannot
        # redirect anything -- it can only rescale -- so it isolates the SYSTEMATIC
        # output-scale drift (each linear's H-weighted gain measured at 1.001-1.025 while
        # its weight-space gain is exactly 1) from the per-channel detail.
        #
        # Together with --temp-match (one scalar for the whole MODEL) and `gain` (one per
        # ROW), this is the middle rung of a granularity ladder. H* predicts the coarse
        # rungs help and the finest hurts -- the opposite of what expressiveness alone
        # would predict, which is what makes the ladder a test rather than a sweep.
        num = float((Wh64 @ H64 * W64).sum())
        den = float((Wh64 @ H64 * Wh64).sum())
        a = a * (num / den if den > 0 else 1.0)

    if mode in ("gain", "both"):
        if mode == "both":
            # centred scatter, applied without materialising Sigma = H - N xbar xbar^T
            uH = Wh64 @ H64
            num = (uH * W64).sum(1) - N * (Wh64 @ xb) * (W64 @ xb)
            den = (uH * Wh64).sum(1) - N * (Wh64 @ xb) ** 2
        else:
            uH = Wh64 @ H64
            num = (uH * W64).sum(1)
            den = (uH * Wh64).sum(1)
        # `Sigma` is PSD, so `den = u Sigma u^T >= 0` in exact arithmetic; a value at or
        # below zero means catastrophic cancellation (few calibration tokens, or a row
        # whose activation energy is concentrated in the mean). Fall back to a_r = 1
        # there -- that is the uncorrected artifact, never worse than the incumbent.
        # Guarding on the UNCENTRED energy rather than an absolute epsilon keeps the
        # test scale-free across matrices whose activations differ by orders of magnitude.
        ref = (Wh64 @ H64 * Wh64).sum(1).clamp_min(0)
        ok = den > 1e-10 * ref.clamp_min(eps)
        a = torch.where(ok, num / torch.where(ok, den, torch.ones_like(den)),
                        torch.ones_like(num))
        n_fallback = int((~ok).sum())

    if mode in ("bias", "both"):
        b = (W64 @ xb) - a * (Wh64 @ xb)

    # Realised residual at the chosen (a, b), in the same units as `base`:
    #   sum_t ( (a u - v) . x_t + b )^2
    #     = (a u - v) H (a u - v)^T  +  2 b sum_t (a u - v) . x_t  +  N b^2
    # and sum_t x_t = N * xbar, which is where the factor N in the cross term comes from.
    e2 = a[:, None] * Wh64 - W64
    res = torch.einsum("oi,ij,oj->o", e2, H64, e2) + 2.0 * N * b * (e2 @ xb) + N * b ** 2

    tot_base, tot_res = float(base.sum()), float(res.sum())
    diag = {
        "mode": mode,
        "frac_output_mse_removed": 1.0 - tot_res / max(tot_base, eps),
        "a_mean": float(a.mean()), "a_min": float(a.min()), "a_max": float(a.max()),
        "b_absmean": float(b.abs().mean()), "b_absmax": float(b.abs().max()),
        "n_gain_fallback": n_fallback, "n_rows": out_dim,
    }
    return a.float(), b.float(), diag


@torch.no_grad()
def quantize_bias(b, bits):
    """Round a per-row bias vector to `bits` of mantissa against a shared exponent.

    The stored form is a per-MATRIX float32 step `d` plus one `bits`-wide signed integer
    per row, so the charge is `bits` per output row (the single step is 32 bits per
    matrix, ~1e-7 bpw, and is folded into the report as negligible -- stated, not hidden).
    `bits >= 32` returns the vector unchanged, for the "charge fp32, store exactly" arm.
    """
    if bits is None or bits >= 32:
        return b, 0.0
    lo = -(2 ** (bits - 1))
    hi = 2 ** (bits - 1) - 1
    amax = float(b.abs().max())
    if amax == 0.0:
        return torch.zeros_like(b), 0.0
    d = amax / hi
    q = torch.clamp(torch.round(b / d), lo, hi)
    return (q * d).to(b.dtype), d
