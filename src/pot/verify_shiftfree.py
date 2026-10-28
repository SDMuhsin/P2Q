"""
Machine verification of the immutable condition.

The claim P2Q makes is not "the code was written to only emit powers of two"; it is that
the ARTIFACT -- the dequantized weight matrix that is actually evaluated -- factors as

    W_hat[i, j] = code * scale[i, j],   code in A

with A = {0} u {+-2^e}                     (strict)   -> 1 exponent add per product
     A = {0} u {+-2^a} u {+-(2^a +- 2^b)}  (add2)     -> 2 exponent adds + 1 add

This module proves that, elementwise, on whatever tensors it is handed.

How the proof is made exact
---------------------------
The naive check `frexp(W_hat / scale).mantissa == 0.5` is exact ONLY when the code is
itself a power of two, because then `code * scale` is exact in binary floating point and
dividing it back by `scale` returns `code` bit-for-bit. For 'add2' codes `code * scale`
rounds, and the recovered ratio misses `code` by up to one ulp -- so a pure ratio test
would either false-alarm or need a tolerance, and a tolerance is exactly the rubber stamp
we are trying to avoid (an epsilon big enough to absorb the ulp is also big enough to
absorb a genuinely non-dyadic weight near a codeword).

So we do the stronger, tolerance-free thing: we EXHIBIT the witness. For each entry we

  1. compute the ratio in float64 and snap it to the nearest member of A -- a closed,
     finite, exactly-representable set, so the snap is exact and the snapped value is
     provably in A by construction;
  2. recompute `code * scale` in the artifact's own dtype and require BIT EQUALITY with
     the stored W_hat.

Passing means we have produced, for every entry, an explicit member of A and a scale
whose product IS the stored number -- not approximately, exactly. There is no tolerance
anywhere. Corrupting a single weight by any amount that changes its float bits fails.
"""

from __future__ import annotations

import torch

from .codebooks import shift_alphabet

__all__ = ["verify_strict_pot", "verify_add2", "verify_codebook_artifact",
           "expand_scales"]


def expand_scales(scales: torch.Tensor, n_cols: int, group: int | None = None,
                  col_scale: torch.Tensor | None = None,
                  g_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Build the full [O, I] effective scale from the stored parameters.

    `scales` is [O, G] (per row-group) or already [O, I]. `g_idx` [I] maps each column to
    its group (needed under act_order); without it columns are assigned in blocks of
    `group`. `col_scale` [I] is the optional per-input-column factor, applied as the
    OUTER multiplication -- the same association order the quantizer used, which is what
    makes the bit-equality test below meaningful.
    """
    S = scales
    if S.shape[1] != n_cols:
        if g_idx is not None:
            S = S[:, g_idx]
        else:
            assert group is not None and S.shape[1] * group == n_cols, (
                f"cannot expand scales {tuple(S.shape)} to {n_cols} columns")
            S = S.repeat_interleave(group, dim=1)
    if col_scale is not None:
        S = S * col_scale.to(S.dtype).view(1, -1)
    return S


# ------------------------------------------------------------------ snap functions
def _snap_pot(r: torch.Tensor) -> torch.Tensor:
    """Nearest +-2^e (in float64, exact) to each entry of r; 0 maps to 0.

    'Nearest' on a log grid is nearest in log space: e = round(log2|r|). That is the
    assignment the quantizer makes, and any r whose true code is a power of two lands on
    it exactly.
    """
    out = torch.zeros_like(r)
    nz = r != 0
    a = r[nz].abs()
    e = torch.round(torch.log2(a))
    e = e.clamp(-1000, 1000)
    out[nz] = torch.sign(r[nz]) * torch.exp2(e)
    return out


class _Add2Snapper:
    """Nearest member of the two-shift alphabet, by bisection on a sorted table."""

    def __init__(self, device, lo=-40, hi=8, cap=1e6):
        self.A = shift_alphabet(lo, hi, cap=cap, dtype=torch.float64, device=device)
        self.b = (self.A[1:] + self.A[:-1]) * 0.5

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return self.A[torch.bucketize(r.contiguous(), self.b)]


# ------------------------------------------------------------------------- core
@torch.no_grad()
def _verify(W_hat: torch.Tensor, scales: torch.Tensor, snap, kind: str,
            group: int | None = None, col_scale=None, g_idx=None,
            max_examples: int = 8, chunk_rows: int = 512):
    O, I = W_hat.shape
    S = expand_scales(scales.to(W_hat.device), I, group, col_scale, g_idx)
    assert S.shape == W_hat.shape, f"scale shape {tuple(S.shape)} != {tuple(W_hat.shape)}"
    dt = W_hat.dtype

    n_bad = 0
    n_zero_scale = 0
    examples = []
    emin, emax = float("inf"), float("-inf")

    for i in range(0, O, chunk_rows):
        w = W_hat[i:i + chunk_rows]
        s = S[i:i + chunk_rows].to(dt)
        w64, s64 = w.double(), s.double()

        bad_s = (s64 == 0) | ~torch.isfinite(s64)
        if bool(bad_s.any()):
            # a zero/nonfinite scale is only admissible if the weight is exactly zero
            viol = bad_s & (w64 != 0)
            n_zero_scale += int(bad_s.sum())
        else:
            viol = torch.zeros_like(w, dtype=torch.bool)

        safe_s = torch.where(bad_s, torch.ones_like(s64), s64)
        r = w64 / safe_s
        code = snap(r)
        # the witness test: does this exact code, times this exact scale, reproduce the
        # stored float bit-for-bit in the artifact's own dtype?
        recon = code.to(dt) * s
        viol = viol | (recon != w) | ~torch.isfinite(w)

        nz = code != 0
        if bool(nz.any()):
            lg = torch.log2(code[nz].abs())
            emin = min(emin, float(lg.min()))
            emax = max(emax, float(lg.max()))

        nb = int(viol.sum())
        if nb:
            n_bad += nb
            if len(examples) < max_examples:
                idx = viol.nonzero()[: max_examples - len(examples)]
                for rr, cc in idx.tolist():
                    examples.append({
                        "row": i + rr, "col": cc,
                        "w_hat": float(w[rr, cc]), "scale": float(s[rr, cc]),
                        "ratio": float(r[rr, cc]), "nearest_code": float(code[rr, cc]),
                        "reconstructed": float(recon[rr, cc]),
                    })

    total = O * I
    ok = (n_bad == 0)
    diag = {
        "kind": kind, "ok": ok, "n_weights": total, "n_violations": n_bad,
        "frac_violations": n_bad / total if total else 0.0,
        "n_zero_or_nonfinite_scales": n_zero_scale,
        "code_log2_min": None if emin == float("inf") else emin,
        "code_log2_max": None if emax == float("-inf") else emax,
        "examples": examples,
        "dtype": str(dt),
    }
    return ok, diag


@torch.no_grad()
def verify_strict_pot(W_hat, scales, group=None, col_scale=None, g_idx=None, **kw):
    """Assert every W_hat[i,j] is exactly (0 or +-2^e) * scale[i,j]. -> (ok, diagnostics)"""
    return _verify(W_hat, scales, _snap_pot, "strict_pot", group, col_scale, g_idx, **kw)


@torch.no_grad()
def verify_add2(W_hat, scales, group=None, col_scale=None, g_idx=None, **kw):
    """Assert every W_hat[i,j] is exactly (0 or +-2^a +- 2^b) * scale[i,j]."""
    snap = _Add2Snapper(W_hat.device)
    return _verify(W_hat, scales, snap, "add2", group, col_scale, g_idx, **kw)


@torch.no_grad()
def verify_codebook_artifact(W_hat, scales, codebook, group=None, col_scale=None,
                             g_idx=None, **kw):
    """Dispatch on `codebook.shift_free`. Returns (ok, diag); a 'none' codebook is not
    verifiable and returns ok=None with kind='none' so callers cannot mistake a
    non-shift-free run for a passing one."""
    sf = getattr(codebook, "shift_free", "none")
    if sf == "strict_pot":
        return verify_strict_pot(W_hat, scales, group, col_scale, g_idx, **kw)
    if sf == "add2":
        return verify_add2(W_hat, scales, group, col_scale, g_idx, **kw)
    return None, {"kind": "none", "ok": None,
                  "note": "codebook is not shift-free; nothing to verify"}
