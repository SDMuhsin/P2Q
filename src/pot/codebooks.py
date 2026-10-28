"""
Scalar 4-bit codebooks for P2Q, and the machinery to fit new ones.

A codebook is a sorted 1-D tensor of `levels` (the *normalized* codewords) plus a
metadata dict. The dequantized weight is always

    W_hat[i, j] = levels[code[i, j]] * scale[i, j]

so the arithmetic cost of one weight-activation product is decided entirely by the
structure of `levels`:

  shift_free == 'strict_pot'  every nonzero level is exactly +-2^e
                              -> x * level is ONE exponent add on a bf16/fp16 x.
  shift_free == 'add2'        every nonzero level is exactly +-2^a +- 2^b
                              -> two exponent adds and one fp add. Still no multiply.
  shift_free == 'none'        a general float codebook (INT4, NF4). Multiplies.

Normalization is irrelevant to accuracy -- the per-group scale is fitted by a
multiplicative grid search, so any global rescaling of a codebook is absorbed. It is
NOT irrelevant to the shift-free property: multiplying a codebook by a non-dyadic
constant destroys it. So codebooks here are kept on their natural dyadic lattice and
`levels.abs().max()` is *not* forced to 1 (see `apot_k2_b4`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

__all__ = [
    "Codebook", "strict_pot", "strict_pot_zero", "apot_k2_b4", "learned_add2",
    "fit_add2_codebook", "int4_asym", "int4_sym", "nf4", "shift_alphabet",
    "nearest_codeword", "get_codebook", "CODEBOOKS", "NEXP_CODEBOOKS",
]

STRICT_POT = "strict_pot"
ADD2 = "add2"
NONE = "none"


@dataclass
class Codebook:
    """A sorted 1-D float codebook plus its shift-freeness metadata."""

    levels: torch.Tensor
    name: str = "codebook"
    shift_free: str = NONE          # 'strict_pot' | 'add2' | 'none'
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.levels.dim() == 1, "codebook must be 1-D"
        assert torch.all(self.levels[1:] > self.levels[:-1]), "codebook must be sorted+unique"
        assert self.shift_free in (STRICT_POT, ADD2, NONE)
        self.meta = dict(self.meta)
        self.meta.setdefault("n_levels", int(self.levels.numel()))
        self.meta.setdefault("shift_free", self.shift_free)
        self.meta.setdefault("name", self.name)

    # -- tensor-ish conveniences -------------------------------------------------
    def __len__(self):
        return int(self.levels.numel())

    def to(self, *a, **k):
        return Codebook(self.levels.to(*a, **k), self.name, self.shift_free, self.meta)

    @property
    def device(self):
        return self.levels.device

    @property
    def absmax(self) -> float:
        return float(self.levels.abs().max())

    @property
    def is_shift_free(self) -> bool:
        return self.shift_free != NONE

    def bits(self) -> float:
        """Bits needed to index one codeword (ceil of log2 of the level count)."""
        import math
        return float(math.ceil(math.log2(len(self))))

    def __repr__(self):
        return (f"Codebook({self.name}, n={len(self)}, shift_free={self.shift_free}, "
                f"absmax={self.absmax:g})")


def _mk(vals, name, shift_free, dtype=torch.float32, **meta) -> Codebook:
    v = sorted(set(float(x) + 0.0 for x in vals))  # +0.0 normalizes -0.0
    return Codebook(torch.tensor(v, dtype=dtype), name, shift_free, meta)


# --------------------------------------------------------------------------- PoT
def strict_pot(nexp: int = 8) -> Codebook:
    """P2Q-strict: 1 sign bit + 3 exponent bits, NO zero.

    levels = +-{2^0, 2^-1, ..., 2^-(nexp-1)}; with nexp=8 that is exactly 16 levels,
    i.e. 4 bits. This is the PoT-PTQ (arXiv:2507.11959) weight format.
    """
    v = [2.0 ** (-i) for i in range(nexp)]
    return _mk([-x for x in v] + v, f"strict_pot{nexp}", STRICT_POT, n_exp=nexp)


def strict_pot_zero(nexp: int = 7) -> Codebook:
    """Strict PoT that spends one code on exact zero: 0 u +-{2^0 .. 2^-(nexp-1)}.

    nexp=7 -> 15 levels, which still fits in 4 bits (one code wasted). Exists so we
    can measure whether an exact zero is worth an exponent of dynamic range.
    """
    v = [2.0 ** (-i) for i in range(nexp)]
    return _mk([0.0] + v + [-x for x in v], f"strict_pot_zero{nexp}", STRICT_POT,
               n_exp=nexp)


def apot_k2_b4(signed: bool = True) -> Codebook:
    """Additive PoT, b=4 k=2 (Li et al., APoT, ICLR'20).

    Unsigned form (`signed=False`, 16 magnitudes, the one the APoT paper writes down):
        p0 in {0, 2^0, 2^-2, 2^-4},  p1 in {0, 2^-1, 2^-3, 2^-5},  level = p0 + p1.
    The paper multiplies by gamma = 2/3 so that max(level) = 1. gamma is NOT dyadic and
    would destroy shift-freeness; it is also a pure global rescale, which our per-group
    scale search absorbs exactly. So we DROP gamma and keep the raw dyadic lattice
    (max level 1.5). `meta['gamma']` records the factor for reference.

    Signed form (`signed=True`, the default, and the one usable as a 4-bit *weight*
    codebook): 1 sign bit + 3 magnitude bits. Three bits cannot be split 2+2, so the
    standard sign-magnitude split is used: p0 in {0, 2^0, 2^-2, 2^-4} (2 bits),
    p1 in {0, 2^-1} (1 bit) -> 8 magnitudes; with the sign that is 16 codes but only
    15 distinct values, because +0 == -0 (the usual sign-magnitude redundancy).

    Every level is a sum of at most two powers of two -> shift_free == 'add2'.
    """
    if signed:
        p0 = [0.0, 1.0, 2.0 ** -2, 2.0 ** -4]
        p1 = [0.0, 2.0 ** -1]
    else:
        p0 = [0.0, 1.0, 2.0 ** -2, 2.0 ** -4]
        p1 = [0.0, 2.0 ** -1, 2.0 ** -3, 2.0 ** -5]
    mags = sorted({a + b for a, b in itertools.product(p0, p1)})
    vals = mags if not signed else ([-m for m in mags] + mags)
    return _mk(vals, f"apot_k2_b4{'' if signed else '_unsigned'}", ADD2,
               gamma=2.0 / 3.0, signed=signed, magnitudes=mags)


def learned_add2() -> Codebook:
    """The 16 two-shift levels fitted to Qwen3-1.7B's weight distribution (P2Q-add2).

    +-{0.875, 0.5625, 0.375, 0.28125, 0.21875, 0.140625, 0.078125, 0.02734375}
    e.g. 0.875 = 2^0 - 2^-3, 0.5625 = 2^-1 + 2^-4, 0.02734375 = 2^-5 - 2^-8.
    Reproduce with `fit_add2_codebook`.
    """
    v = [0.875, 0.5625, 0.375, 0.28125, 0.21875, 0.140625, 0.078125, 0.02734375]
    return _mk([-x for x in v] + v, "learned_add2", ADD2, fitted_on="Qwen3-1.7B")


# ------------------------------------------------------- the two-shift alphabet
def shift_alphabet(lo: int = -9, hi: int = 1, cap: float = 2.0,
                   dtype=torch.float64, device=None) -> torch.Tensor:
    """S = {0} u {+-2^a} u {+-(2^a +- 2^b)} for a, b in [lo, hi], |value| <= cap.

    Sorted, unique, float64 (every member is a small dyadic rational, so float64 --
    and float32 for this exponent range -- holds them exactly).
    """
    out = {0.0}
    for a in range(lo, hi + 1):
        pa = 2.0 ** a
        out.add(pa); out.add(-pa)
        for b in range(lo, hi + 1):
            for s in (1.0, -1.0):
                v = pa + s * 2.0 ** b
                if v != 0.0:
                    out.add(v); out.add(-v)
    vals = sorted(v for v in out if abs(v) <= cap)
    return torch.tensor(vals, dtype=dtype, device=device)


def fit_add2_codebook(samples: torch.Tensor, n: int = 16, iters: int = 40,
                      lo: int = -9, hi: int = 1, name: str = "fit_add2") -> Codebook:
    """Constrained Lloyd-Max: 1-D k-means whose centroids are projected onto the
    two-shift alphabet S at every M-step.

    The unconstrained M-step optimum of a 1-D cluster is its mean; projecting that mean
    onto S is the exact constrained optimum for that cluster, because within-cluster SSE
    is (mean - c)^2 * |cluster| + const, monotone in |mean - c|. So this is a true
    coordinate-descent on the quantization MSE with the shift constraint enforced at
    every step, not a round-after-the-fact heuristic.

    Args:
        samples: 1-D tensor of *normalized* weights (e.g. w / group-absmax).
        n:       number of levels to fit.
    Returns:
        Codebook with shift_free == 'add2'.
    """
    x = samples.reshape(-1).double()
    S = shift_alphabet(lo, hi, device=x.device)

    # init from quantiles of the data, snapped onto S
    qs = torch.linspace(0.002, 0.998, n, device=x.device, dtype=torch.float64)
    C = torch.quantile(x, qs)
    C = S[(C[:, None] - S[None, :]).abs().argmin(1)]
    C = torch.unique(C)

    for _ in range(iters):
        idx = torch.bucketize(x, (C[1:] + C[:-1]) / 2)          # nearest-centroid, O(log n)
        newC = C.clone()
        for j in range(C.numel()):
            m = idx == j
            if not bool(m.any()):
                continue
            newC[j] = S[(x[m].mean() - S).abs().argmin()]       # projected M-step
        newC = torch.unique(newC)
        if newC.numel() < C.numel():
            # a collision collapsed clusters: re-seed the missing ones from the old set
            spare = C[~torch.isin(C, newC)][: C.numel() - newC.numel()]
            newC = torch.unique(torch.cat([newC, spare]))
        if newC.numel() == C.numel() and torch.equal(newC, C):
            break
        C = newC

    if C.numel() != n:                                          # top up from S if needed
        extra = S[~torch.isin(S, C)]
        pick = extra[(extra.abs() - C.abs().max()).abs().argsort()[: n - C.numel()]]
        C = torch.unique(torch.cat([C, pick]))
    return _mk(C.tolist(), name, ADD2, n_samples=int(x.numel()), lloyd_iters=iters)


# -------------------------------------------------------- reference (non-shift) books
def int4_asym() -> Codebook:
    """16 uniform levels on [-1, 1] (asymmetric INT4 after folding the zero point)."""
    n = 15
    return _mk([i / n * 2 - 1 for i in range(16)], "int4_asym", NONE)


def int4_sym() -> Codebook:
    """Symmetric INT4: {-7..7}/7, 15 levels."""
    n = 7
    return _mk([i / n for i in range(-n, n + 1)], "int4_sym", NONE)


def nf4() -> Codebook:
    """NormalFloat-4 (QLoRA)."""
    v = [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.28444138, -0.18477343,
         -0.09105003, 0.0, 0.07958029, 0.16093020, 0.24611230, 0.33791524,
         0.44070983, 0.56261700, 0.72295684, 1.0]
    return _mk(v, "nf4", NONE)


CODEBOOKS = {
    "strict_pot": strict_pot,
    "strict_pot_zero": strict_pot_zero,
    "apot_k2_b4": apot_k2_b4,
    "apot_k2_b4_unsigned": lambda: apot_k2_b4(signed=False),
    "learned_add2": learned_add2,
    "int4_asym": int4_asym,
    "int4_sym": int4_sym,
    "nf4": nf4,
}


#: codebooks whose alphabet size is parameterized by `nexp` (number of octaves).
NEXP_CODEBOOKS = ("strict_pot", "strict_pot_zero")


def get_codebook(name: str, device=None, dtype=torch.float32,
                 nexp: int | None = None) -> Codebook:
    """Build a codebook by name.

    `nexp` is a DIAGNOSTIC override for the number of octaves of `strict_pot` /
    `strict_pot_zero`. Leave it None (the default) to get the standard 4-bit books --
    `strict_pot(8)` and `strict_pot_zero(7)` -- byte-for-byte as before. Passing it for
    any other codebook is an error rather than a silent no-op.

    Note that nexp != the 4-bit default changes the CODE WIDTH: `strict_pot(16)` has 32
    levels = 5 bits/weight, which `Codebook.bits()` reports honestly and which the bpw
    accounting will therefore charge. Such runs exceed the project's 4.25 bpw ceiling
    and are reference/diagnostic arms only.
    """
    if name not in CODEBOOKS:
        raise KeyError(f"unknown codebook {name!r}; have {sorted(CODEBOOKS)}")
    if nexp is None:
        cb = CODEBOOKS[name]()
    else:
        if name not in NEXP_CODEBOOKS:
            raise ValueError(
                f"--codebook-nexp only applies to {list(NEXP_CODEBOOKS)}, not {name!r}")
        if int(nexp) < 2:
            raise ValueError(f"nexp must be >= 2, got {nexp}")
        cb = CODEBOOKS[name](int(nexp))
    return cb.to(device=device, dtype=dtype)


# ------------------------------------------------------------------ nearest codeword
def nearest_codeword(x: torch.Tensor, C: torch.Tensor):
    """Nearest-codeword index of every entry of `x` in the SORTED codebook `C`.

    O(log K) per element via bisection on the midpoints, instead of the O(K) broadcast
    `(x[..., None] - C).abs().argmin(-1)`. For a 200-point scale grid over a 128-wide
    group that is the difference between 26 MB and 420 MB of scratch.
    """
    bounds = (C[1:] + C[:-1]) * 0.5
    return torch.bucketize(x.contiguous(), bounds)


def quantize_to(x: torch.Tensor, C: torch.Tensor):
    """(codes, dequantized) for `x` against sorted codebook `C`."""
    idx = nearest_codeword(x, C)
    return idx, C[idx]
