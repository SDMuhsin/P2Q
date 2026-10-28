"""
Temperature-match the quantized model to its bf16 teacher with ONE scalar, zero bits.

The mechanism this targets (SESSION_2026-09-11_INSTRUMENT.md, "H*")
--------------------------------------------------------------------
Four deterministic strict-PoT artifacts whose weight-space diagnostics agree to five
significant figures are rank-identical on three things: ARC-Easy `acc_norm`, the mean
per-token log-probability of the answer they pick (-4.72 ... -3.89), and BoolQ's
yes/no margin (2.26 ... 6.34, bf16 = 3.44). Six other benchmarks put them within one
standard error of each other. That is the signature of a shift in OUTPUT SHARPNESS --
an effective temperature -- not of capability. Its source is measurable: every
quantized linear's H-weighted output gain is 1.001-1.025 although its weight-space
gain is exactly 1, so each layer's output is ~1-2% off in scale, RMSNorm renormalises
the residual, and 196 such errors compose into a net temperature drift whose SIGN
depends on grouping details. A lottery over temperature is invisible to every
weight-space statistic, which is why three sessions of weight-space levers could not
see it.

The correction
--------------
Qwen3's logits are `lm_head(g * h / rms(h))` with `g` the final RMSNorm gain, an fp
vector that is already stored and already uncharged. Logits are exactly linear in `g`,
so scaling `g` by a scalar `alpha` is exactly a temperature change, costs zero bits,
touches no weight, and is legal under C1/C2 by construction (nothing on the
weight x activation path changes).

`alpha` is fitted by minimising the teacher-forward KL

        KL( softmax(z_T) || softmax(alpha * z_S) )

averaged over CALIBRATION text (the wikitext2 chunks the run was calibrated on, never
the test split and never a benchmark), by golden-section search on a bracket. KL was
tested in SESSION 13.6 as a criterion for RANKING artifacts and failed; here it is the
objective of a one-parameter fit of an already-fixed artifact, a different use. The
choice of KL over "match mean NLL" is deliberate: matching NLL does not match
sharpness, and sharpness is the variable.

If H* is right, this single scalar should move an artifact's BoolQ margin and
per-token log-prob toward bf16's, and move ARC-Easy `acc_norm` in the direction the
sharper artifacts sit -- i.e. it should reproduce, and thereby dissolve, the lottery.
If it is wrong, alpha will be ~1 or the metrics will not follow it, and that is a clean
kill of the mechanism at the cost of one scalar search.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def _teacher_logits(teacher, calib, device, n_windows):
    outs = []
    for i in range(min(n_windows, calib.shape[0])):
        outs.append(teacher(calib[i:i + 1].to(device)).logits[0].float().cpu())
    return outs


@torch.no_grad()
def _student_logits(student, calib, device, n_windows):
    outs = []
    for i in range(min(n_windows, calib.shape[0])):
        outs.append(student(calib[i:i + 1].to(device)).logits[0].float().cpu())
    return outs


@torch.no_grad()
def _kl_at(alpha, t_logits, s_logits, device, chunk=512):
    tot, n = 0.0, 0
    for zt, zs in zip(t_logits, s_logits):
        for a in range(0, zt.shape[0], chunk):
            lt = F.log_softmax(zt[a:a + chunk].to(device), -1)
            ls = F.log_softmax(alpha * zs[a:a + chunk].to(device), -1)
            tot += float((lt.exp() * (lt - ls)).sum())
            n += lt.shape[0]
    return tot / max(n, 1)


@torch.no_grad()
def fit_temperature(student, teacher, calib, device="cuda", n_windows=16,
                    lo=0.80, hi=1.25, tol=1e-3, verbose=True):
    """Golden-section minimiser of KL(teacher || alpha-scaled student) over `alpha`.

    Returns (alpha, info). Both models must already be on `device` in eval mode.
    `calib` is the [N, seqlen] token tensor the run was calibrated on.
    """
    t_logits = _teacher_logits(teacher, calib, device, n_windows)
    s_logits = _student_logits(student, calib, device, n_windows)

    kl1 = _kl_at(1.0, t_logits, s_logits, device)
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _kl_at(c, t_logits, s_logits, device), _kl_at(d, t_logits, s_logits, device)
    it = 0
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _kl_at(c, t_logits, s_logits, device)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _kl_at(d, t_logits, s_logits, device)
        it += 1
    alpha = (a + b) / 2
    kla = _kl_at(alpha, t_logits, s_logits, device)
    info = {"alpha": alpha, "kl_before": kl1, "kl_after": kla,
            "kl_rel_change": (kla - kl1) / max(kl1, 1e-12),
            "n_windows": min(n_windows, calib.shape[0]), "iters": it,
            "bracket": [lo, hi], "at_bracket_edge": bool(alpha - lo < 2 * tol
                                                          or hi - alpha < 2 * tol)}
    if verbose:
        print(f"temp-match: alpha={alpha:.5f}  KL {kl1:.6f} -> {kla:.6f} "
              f"({100 * info['kl_rel_change']:+.2f}%) over {info['n_windows']} windows")
    return alpha, info


@torch.no_grad()
def apply_temperature(model, alpha):
    """Fold `alpha` into the final RMSNorm gain. Exactly a logit scale; zero bits."""
    mm = getattr(model, "model", model)
    mm.norm.weight.data = (mm.norm.weight.data.float() * float(alpha)).to(mm.norm.weight.dtype)
