"""
Per-group ALPHABET selection: spend the fixed metadata budget on choosing, per group,
which strict-PoT exponent ladder to use, instead of spending it all on group-size
granularity.

Motivation (see llmdocs/CHARTER.md and the CPU dynamic-range study this module was
built from): the incumbent alphabet `strict_pot` = {+-2^0, ..., +-2^-7} has a FIXED
dynamic range of 2^7 = 128x. A per-group scale can slide that window but not change its
WIDTH. Measuring per-group dynamic range (log2(max|w|/min|w|) within a group) on real
Qwen3-1.7B weights shows real spread: at g=64 the population has mean 8.00 octaves,
std 1.86, p10=5.94, p50=7.70, p90=10.43, p99=13.83, max up to ~31 (a handful of
outlier-bearing groups). The population is NOT homogeneous -- a fixed 7-octave window
is already narrower than the median group needs, and badly undershoots the ~1% of
groups that need 14+ octaves. That is the gap this module exploits.

Every alphabet here is `strict_pot`-shift-free: each level is a SINGLE signed power of
two (C1). The family varies only the exponent STRIDE (spacing between consecutive
levels), which changes the ladder's total span without ever summing two shifts or
introducing a non-dyadic factor:

    stride s, n magnitudes -> levels = {2^0, 2^-s, 2^-2s, ..., 2^-(n-1)s}
    span (top/bottom ratio) = 2^((n-1)*s) octaves

stride=1 is exactly today's `codebooks.strict_pot(nexp)` (bit-identical anchor case).
Larger strides trade per-step resolution for range, which is exactly the axis real
groups vary on (an outlier-bearing group would rather have a coarse but wide ladder
than a fine one that crushes its outlier's neighbours to the floor).

The verifier (`verify_shiftfree.py`) needs NO changes to check an artifact built from
this module: it only checks that W_hat / scale snaps to a single power of two and that
`code * scale` reproduces the stored float bit-for-bit. It does not care which of the
family's ladders supplied that power of two, so a selected-alphabet artifact verifies
exactly like a single-codebook one.
"""

from __future__ import annotations

import math

import torch

from .codebooks import Codebook, STRICT_POT, nearest_codeword
from .gptq_pot import PoTGPTQ
from .scale_search import search_group_scale

__all__ = [
    "strict_pot_stride", "default_family", "selector_bits_for",
    "select_group_alphabet_scale", "PoTGPTQFamily",
]


def strict_pot_stride(stride: int = 1, nexp: int = 8, name: str | None = None) -> Codebook:
    """Strict-PoT ladder with a configurable exponent stride.

    magnitudes = {2^0, 2^-stride, 2^-2*stride, ..., 2^-(nexp-1)*stride}; signed, so
    2*nexp levels total (nexp=8 -> 16 levels -> 4-bit codes, matching today's format).
    Every level is exactly +-2^e for an integer e -- a single signed power of two,
    never a sum of two shifts and never scaled by a non-dyadic constant. `stride=1`
    reproduces `codebooks.strict_pot(nexp)` exactly (same sorted level set), which is
    what the bit-identical anchor test in tests/pot/test_alphabets.py checks.

    Dynamic range spanned by this ladder alone: 2^((nexp-1)*stride).
    """
    assert stride >= 1 and isinstance(stride, int), "stride must be a positive integer " \
        "so every exponent -(i*stride) stays an integer, i.e. every level stays a " \
        "single power of two (C1)"
    mags = [2.0 ** (-(i * stride)) for i in range(nexp)]
    vals = sorted(set(float(x) + 0.0 for x in ([-m for m in mags] + mags)))
    nm = name or f"strict_pot_s{stride}n{nexp}"
    return Codebook(torch.tensor(vals, dtype=torch.float32), nm, STRICT_POT,
                    {"n_exp": nexp, "stride": stride,
                     "span_octaves": (nexp - 1) * stride})


def default_family(strides=(1, 2, 3, 4), nexp: int = 8) -> list[Codebook]:
    """The alphabet family used by `--alphabet-select`.

    Chosen from the CPU dynamic-range study (see module docstring): stride=1 (span 7
    octaves) covers the bulk of the population up to ~p25; stride=2 (span 14) covers
    the median-to-p95 band; stride=3 (span 21) and stride=4 (span 28) reach the p99 and
    the rare high-outlier groups (observed max ~31 octaves) respectively. 4 members ->
    a 2-bit selector with no wasted code.
    """
    fam = [strict_pot_stride(s, nexp) for s in strides]
    n = len(fam[0])
    assert all(len(c) == n for c in fam), "family members must share code width (C1 bits)"
    assert all(c.shift_free == STRICT_POT for c in fam)
    return fam


def selector_bits_for(family) -> int:
    """Bits needed to index one family member. ceil(log2(K)), minimum 1."""
    return max(1, math.ceil(math.log2(len(family))))


@torch.no_grad()
def select_group_alphabet_scale(Wsub: torch.Tensor, family, hess_diag=None, **kw):
    """Jointly pick, PER ROW, the (alphabet, scale) minimizing weighted group MSE.

    Runs the existing exact/grid scale search once per family member (K calls, K
    small) and keeps whichever gives the lower weighted squared error for that row.
    This is the natural generalization of `search_group_scale` to a discrete extra
    choice: the scale search inside each candidate alphabet is already exact/near-
    exact, so the outer argmin over K candidates is an exact joint minimization over
    the finite (alphabet, scale-grid-point) product for each row.

    Args:
        Wsub: [rows, g] weights, one group per row.
        family: list[Codebook], all `strict_pot`.
    Returns:
        (scale [rows, 1], alpha_idx [rows] long, err [rows]).
    """
    all_s, all_e = [], []
    for cb in family:
        s, e = search_group_scale(Wsub, cb, hess_diag=hess_diag, **kw)
        all_s.append(s.squeeze(1))
        all_e.append(e)
    S = torch.stack(all_s, dim=1)          # [rows, K]
    E = torch.stack(all_e, dim=1)          # [rows, K]
    k = E.argmin(1)
    rows = torch.arange(Wsub.shape[0], device=Wsub.device)
    scale = S[rows, k].unsqueeze(1)
    err = E[rows, k]
    return scale, k, err


class PoTGPTQFamily(PoTGPTQ):
    """PoTGPTQ generalized to a per-group ALPHABET selector alongside the scale.

    This is a deliberate near-duplicate of `PoTGPTQ.quantize()` rather than a patch to
    the shared hot loop in gptq_pot.py: it keeps the alphabet-selection feature from
    colliding with concurrent edits to that file, and keeps the well-tested
    single-codebook path completely unchanged (the `--alphabet-select` anchor test
    checks the two paths agree bit-for-bit when the family has one member).

    A "group" here is a (output row, column-block) unit -- the same granularity the
    plain scale already uses (`scales` is `[out_features, n_col_groups]`). The
    selector therefore costs `selector_bits / group` bpw, exactly like the scale
    costs `scale_bits / group` bpw; see `account_bpw`'s new `alphabet_selector` term.

    Does NOT support `--exact-scale`, `--refine-codes`, `--learn-scale`, or
    `--col-scale`: those act on a single fixed codebook and were out of scope here.
    """

    @torch.no_grad()
    def quantize(self, family, group: int = 128, percdamp: float = 0.01,
                blocksize: int = 128, act_order: bool = False,
                rescale_in_loop: bool = True, pot_scale: bool = False,
                n_grid: int = 200, lo: float = 0.3, hi: float = 1.2,
                hess_weighted_scale: bool = False, verbose: bool = False):
        n_levels = len(family[0])
        assert all(len(c) == n_levels for c in family), \
            "family members must share code width"
        assert all(c.shift_free == STRICT_POT for c in family), \
            "alphabet-select family must be strict_pot (C1)"
        levels_stack = torch.stack(
            [c.levels.to(self.dev, torch.float32) for c in family], 0)   # [K, n_levels]
        K = levels_stack.shape[0]

        W = self.W.clone()
        rows, cols = W.shape
        if group is None or group <= 0 or group > cols:
            group = cols
        assert cols % group == 0, f"in_features {cols} not divisible by group {group}"
        if blocksize % group and group % blocksize:
            raise ValueError("blocksize must be a multiple of group (or vice versa)")
        blocksize = max(blocksize, group)
        blocksize -= blocksize % group
        n_groups = cols // group

        H = self.H.clone()
        assert self.nsamples > 0, "call add_batch() before quantize()"

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Hdiag0 = torch.diag(H).clone()
        perm = None
        if act_order:
            perm = torch.argsort(Hdiag0, descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        damp = percdamp * torch.mean(torch.diag(H))
        H[range(cols), range(cols)] += damp
        L = torch.linalg.cholesky(H)
        Hi = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hi, upper=True)

        hdiag_for_scale = torch.diag(H) if hess_weighted_scale else None

        Q = torch.zeros_like(W)
        codes = torch.zeros(rows, cols, dtype=torch.uint8, device=self.dev)
        scales = torch.zeros(rows, n_groups, device=self.dev, dtype=torch.float32)
        alpha_sel = torch.zeros(rows, n_groups, device=self.dev, dtype=torch.uint8)
        Losses = torch.zeros_like(W)

        sk = dict(n_grid=n_grid, lo=lo, hi=hi, pot_scale=pot_scale)

        if not rescale_in_loop:
            for gi in range(n_groups):
                sl = slice(gi * group, (gi + 1) * group)
                hd = None if hdiag_for_scale is None else hdiag_for_scale[sl]
                s, k, _ = select_group_alphabet_scale(W[:, sl], family, hess_diag=hd, **sk)
                scales[:, gi] = s.squeeze(1)
                alpha_sel[:, gi] = k.to(torch.uint8)

        cur_s = None
        cur_k = None
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            cnt = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            E1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(cnt):
                col = i1 + i
                gi = col // group
                if col % group == 0:
                    if rescale_in_loop:
                        sub = W1[:, i:i + group]
                        hd = (None if hdiag_for_scale is None
                              else hdiag_for_scale[col:col + group])
                        cur_s, cur_k, _ = select_group_alphabet_scale(
                            sub, family, hess_diag=hd, **sk)
                        scales[:, gi] = cur_s.squeeze(1)
                        alpha_sel[:, gi] = cur_k.to(torch.uint8)
                    else:
                        cur_s = scales[:, gi:gi + 1]
                        cur_k = alpha_sel[:, gi].long()

                w = W1[:, i]
                d = Hinv1[i, i]
                s = cur_s.squeeze(1)

                idx = torch.zeros(rows, dtype=torch.long, device=self.dev)
                q = torch.zeros(rows, device=self.dev)
                for ai in range(K):
                    m = cur_k == ai
                    if not bool(m.any()):
                        continue
                    Ca = levels_stack[ai]
                    idxa = nearest_codeword(w[m] / s[m], Ca)
                    idx[m] = idxa
                    q[m] = Ca[idxa] * s[m]

                Q1[:, i] = q
                codes[:, col] = idx.to(torch.uint8)
                Losses[:, col] = (w - q) ** 2 / d ** 2

                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                E1[:, i] = err

            Q[:, i1:i2] = Q1
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
            if verbose:
                print(f"  cols {i1}-{i2}  loss={Losses[:, i1:i2].sum().item():.4g}",
                      flush=True)

        if perm is not None:
            inv = torch.argsort(perm)
            Q = Q[:, inv]
            codes = codes[:, inv]
            g_idx = (torch.arange(cols, device=self.dev) // group)[inv]
        else:
            g_idx = torch.arange(cols, device=self.dev) // group

        S = scales[:, g_idx]
        A = alpha_sel[:, g_idx].long()
        W_hat = levels_stack[A, codes.long()] * S
        assert torch.equal(W_hat, Q), "dequant from stored params disagrees with the sweep"

        Wref = self.W
        err = (W_hat - Wref)
        out = {
            "W_hat": W_hat,
            "codes": codes,
            "scales": scales,
            "alpha_sel": alpha_sel,
            "g_idx": g_idx,
            "col_scale": None,
            "group": group,
            "n_groups": n_groups,
            "act_order": act_order,
            "rescale_in_loop": rescale_in_loop,
            "pot_scale": pot_scale,
            "alphabet_family": [c.name for c in family],
            "selector_bits": selector_bits_for(family),
            "gptq_loss": float(Losses.sum()),
            "rel_fro_err": float(err.norm() / Wref.norm().clamp_min(1e-30)),
            "rel_mse": float((err ** 2).sum() / (Wref ** 2).sum().clamp_min(1e-30)),
            "proxy_err": float(torch.einsum("oi,ij,oj->", err, self.H, err)),
            "nsamples": self.nsamples,
        }
        return out
