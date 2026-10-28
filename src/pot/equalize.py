"""
Activation-aware, PoT-native column equalization (the AWQ/SmoothQuant/DFQ *insight*,
not their method).

The insight
-----------
A diagonal reparametrization  W -> W diag(c)^-1,  x -> diag(c) x  is mathematically
exact and changes nothing about the layer's function. It moves quantization difficulty
between input channels for free. Uniform-grid PTQ uses it to protect activation-salient
channels at the expense of unimportant ones, deliberately accepting MORE weight error
where the output does not care.

The log grid makes one half of it free and the other half INERT
---------------------------------------------------------------
`c` is per-input-COLUMN, so it is finer than the per-(row, group) scale `r` (groups are
64 wide) and captures structure `r` provably cannot. That half is real.

But the PoT-constrained half is not. With `c_j = 2^{t_j}`, `W[:, j] / c_j` is an EXACT
relabelling on a strict-PoT alphabet -- and *because* it is exact, it has no effect:

    THEOREM. Let A = {+-2^e : e = 0..-(n-1)} and let `s > 0`. For any `w` and integer `t`
    such that both `w/s` and `w/(2^t s)` snap to a NON-EXTREMAL magnitude of A, the
    reconstruction error is bit-for-bit identical:
        2^t * (Q(w / 2^t; s) * s) - w  ==  Q(w; s) * s - w .
    Proof. `Q(x; s) = s * sign(x) * 2^round(log2|x/s|)`, and away from the ends
    `round(log2|x/s| - t) = round(log2|x/s|) - t` exactly. So the reconstruction is
    `2^t * s * sign * 2^{round(log2|w/s|) - t} = Q(w; s)`. QED.

In words: a log grid is *exactly covariant* under multiplication by a power of two, so a
power-of-two column scale carries ZERO degrees of freedom. Verified numerically:
`max |dE| = 0.0` over 400k weights for every t in +-{1,2,3,5} on the alphabet interior
(see tests/pot/test_equalize.py::test_pot_column_scale_is_a_noop_in_the_interior).

The only thing an integer `t` can do is push weights past an END of the finite alphabet
-- trading clipping at 2^0 against underflow at 2^-(n-1). The per-(row, group) scale `r`
already optimizes exactly that trade-off, so the integer lever is expected to be, and
measures as, essentially powerless on real weights.

Measured on real Qwen3-1.7B weights (blocks 0 and 14, 128-row subsets, g=64 exactly as
the incumbent, unweighted objective), reduction in sum_j ||W_hat[:,j] - W[:,j]||^2:

    group     integer c        real c
    qkv       +0.000 / -0.000  -4.85 / -5.56   (L0 / L14)
    o         -0.000 / +0.000  -8.17 / -6.08
    gate,up   -0.001 / -0.000  -6.06 / -6.15
    down      +0.000 / -0.000  -10.37 / -10.46

The integer lever returns c == 1 for essentially every column (0.0 - 1.4% of columns move
at all, and only ever by clipping-driven whole octaves). The real lever moves 93-97% of
columns and every optimal log2(c) lies strictly INSIDE one octave, |log2 c| < 1.4 -- i.e.
the entire gain is in the fractional part, exactly as the theorem predicts.

Therefore, on this grid, all of the AWQ lever's power lives in the REAL-valued diagonal,
which moves the fractional part of log2|w| and so genuinely changes the rounding. That
is also free here: every Qwen3 linear absorbs its diagonal into an fp object that is
already stored (see `qwen3_equalize_plan`), so real-valued costs no more bits than
integer. This is the opposite of the uniform-grid intuition, where the PoT scale is the
cheap one.

Where the activation salience actually bites (the subtle part)
--------------------------------------------------------------
The objective is the diagonal output error

        J(c, r) = sum_j h_j * || W_hat[:, j] - W[:, j] ||^2 ,   h = diag(H),

with `W_hat[i, j] = code_ij * r[i, g(j)] * c_j`.  Note carefully: for a FIXED `r`, the
columns are independent and `h_j` is a positive constant multiplying column j's error
only -- so `argmin_{c_j} h_j E_j(c_j) = argmin_{c_j} E_j(c_j)` and the salience
CANCELS. Activation-awareness therefore cannot enter through the column lever alone.
It enters through the COUPLING: `r` is shared by 64 columns, and the h-weighted fit of
`r` lets the group scale be set by the salient columns, sacrificing the unimportant
ones -- which the free per-column lever then repairs. That is precisely the AWQ trade,
transcribed to this grid, and it is why this module fits `r` and `c` jointly rather
than fitting `c` against a weight-MSE `r` the way `scale_search.search_col_scale` does.

This matters empirically: h-weighting the group scale ALONE was measured to cost
-5.5 pp ARC-E on this project (`--scale-weight diag`), because it sacrifices the
non-salient columns for nothing. The hypothesis this module tests is that the sacrifice
is worth making once, and only once, there is a free lever to recover it.

Consistency of the objective
----------------------------
Writing `V = W diag(c)^-1`, the quantizer sees `V` and fits `r` to it, so the per-column
weight handed to the scale search must be `h_j * c_j^2`, not `h_j`: an error of `e` in
`V` is an error of `e * c_j` in `W`. Without the `c_j^2` the problem is degenerate
(inflating `c` shrinks the reported error for free). `run_pot.py` passes exactly
`h * c^2` into the sweep via `PoTGPTQ.quantize(scale_hess=...)`, so the artifact's `r`
minimizes the same J this module minimizes.

Exactness of the search
-----------------------
Both half-steps minimize the SAME J exactly:
  * given `c`, each (row, group) scale is independent -> `exact_group_scale` (the
    breakpoint solver) is the exact minimizer;
  * given `r`, each column (or tie-class of columns) is independent -> for integer `t`
    the candidate set is finite and is enumerated in full; for real-valued `c` a
    multiplicative grid with halving span is used (mirroring `search_col_scale`).
So the alternation is monotone in J. There are no tuned hyperparameters: the integer
range is +-8 rungs because the strict-PoT 4-bit alphabet spans exactly 8 octaves, so a
shift of more than 8 moves every weight in a column off one end of it.

Absorption (why this can cost ZERO bits) -- see `qwen3_equalize_plan`.
"""

from __future__ import annotations

import torch

from .codebooks import Codebook, nearest_codeword
from .exact_scale import exact_group_scale
from .scale_search import search_group_scale

__all__ = ["equalize_columns", "qwen3_equalize_plan", "gqa_tie_index",
           "apply_reparam_check", "INT_RANGE"]

INT_RANGE = 8          # strict_pot's 4-bit alphabet spans 8 octaves; see module doc.

# `exact_group_scale`'s own `max_elems` heuristic assumes the scratch tensor is
# [rows, g*(L-1)+1] when it is in fact [rows, g*(L-1)+1, g], so it under-counts by a
# factor of g. We therefore size the chunk here from the real footprint and pass the
# equivalent max_elems through.
_SCRATCH_BYTES = 3 * 10 ** 9


def _levels(C):
    return C.levels if isinstance(C, Codebook) else C


# --------------------------------------------------------------------- tie maps
def gqa_tie_index(num_heads: int, num_kv_heads: int, head_dim: int,
                  device=None) -> torch.Tensor:
    """Column -> free-parameter index for o_proj under grouped-query attention.

    o_proj's input channel (query head q, dim d) is fed by v_proj's output row
    (kv head q // rep, dim d) after `repeat_kv`. A diagonal on o_proj's input is
    absorbable into v_proj's output rows -- and hence into v_proj's ALREADY-STORED fp16
    group scales, for zero extra bits -- if and only if it is constant across the query
    heads that share a kv head. This returns that tie.
    """
    rep = num_heads // num_kv_heads
    j = torch.arange(num_heads * head_dim, device=device)
    return ((j // head_dim) // rep) * head_dim + (j % head_dim)


def qwen3_equalize_plan(cfg, device=None):
    """The absorption analysis for a Qwen3 decoder block, as data.

    Returns a list of dicts, one per *tied group of linears that share one input*:
        members   : linear names within the block
        absorb    : where the diagonal is folded, exactly, at deployment
        free      : True iff folding costs zero stored bits
        tie       : optional [I] index tensor constraining the diagonal
        n_free    : number of free parameters in the diagonal

    Derivation, per linear:

    q,k,v   input = input_layernorm(x).  RMSNorm output y_i = g_i * x_i / rms(x); rms is
            computed on x, so scaling g by d scales y by d EXACTLY. That output feeds
            ONLY q, k and v (the residual branch carries x, not y), so ONE diagonal
            shared by all three is absorbable into `input_layernorm.weight` -- an fp
            vector that already exists and is already uncharged. FREE, but the three
            matrices MUST share one c.

    gate,up input = post_attention_layernorm(h), feeding only gate and up. Same argument.
            FREE, shared.

    o       input = concat_heads(A @ V). Scaling input channel (h, d) by d_{hd} is
            identical to scaling v_proj's OUTPUT ROW for that channel, because the
            attention weights A are unaffected by V's channel scaling and V has no
            output norm in Qwen3 (only q_norm / k_norm exist, on q and k). Scaling an
            output row of v_proj is done by multiplying that row's already-stored fp16
            group scales -- zero extra bits, and v_proj's own relative error is
            unchanged because W and W_hat scale together. FREE, but GQA forces the tie
            above: 8 kv heads x 128 dims = 1024 free parameters for 2048 columns.

    down    input = silu(gate(x)) * up(x). silu is nonlinear so gate cannot absorb a
            diagonal, but the product is LINEAR in up(x): scaling up_proj's output row j
            by d_j scales down_proj's input channel j by d_j exactly. Same free
            mechanism as o/v. FREE and fully per-channel (6144 parameters).

    So all seven linears absorb for free; the price is not bits, it is the three tying
    constraints above.
    """
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // nh
    return [
        {"key": "qkv",
         "members": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
         "absorb": "input_layernorm.weight (fp, already stored)",
         "free": True, "tie": None, "n_free": cfg.hidden_size},
        {"key": "gateup",
         "members": ["mlp.gate_proj", "mlp.up_proj"],
         "absorb": "post_attention_layernorm.weight (fp, already stored)",
         "free": True, "tie": None, "n_free": cfg.hidden_size},
        {"key": "o",
         "members": ["self_attn.o_proj"],
         "absorb": "self_attn.v_proj output-row group scales (fp16, already stored)",
         "free": True, "tie": gqa_tie_index(nh, nkv, hd, device), "n_free": nkv * hd},
        {"key": "down",
         "members": ["mlp.down_proj"],
         "absorb": "mlp.up_proj output-row group scales (fp16, already stored)",
         "free": True, "tie": None, "n_free": cfg.intermediate_size},
    ]


# ------------------------------------------------------------------ the equalizer
@torch.no_grad()
def _fit_r(V: torch.Tensor, C: torch.Tensor, group: int,
           w: torch.Tensor, exact: bool = False) -> torch.Tensor:
    """Per-(row, group) scale for V, with per-column error weights `w` [I]. -> r [O, G].

    `exact=True` uses the breakpoint solver (`exact_scale.py`), the true minimizer. It is
    NOT the default here, for a cost reason worth recording: the breakpoint solver
    evaluates g*(L-1)+1 = 961 candidate intervals per group in float64 versus 201
    multipliers in float32 for the grid -- ~30x the work, at fp64's 1/32 rate on Ampere.
    Inside this alternation it is called 6 times per matrix per tied group, which
    measured at ~8 min for ONE (q,k,v) group of ONE block, i.e. ~12 h for the model. The
    grid is used instead, so the INNER r half-step is exact only within its 201-point
    grid (which always contains the absmax multiplier 1.0, so it is never worse than the
    absmax init). The ARTIFACT's scale is still the exact solver's, because
    `--exact-scale` is passed to the sweep -- only the CHOICE of `c` rests on the grid,
    never the deployed weights.
    """
    O, I = V.shape
    G = I // group
    Vg = V.reshape(O, G, group).reshape(O * G, group)
    wg = w.view(1, G, group).expand(O, G, group).reshape(O * G, group)
    if exact:
        L = C.numel()
        nr = group * (L - 1) + 1
        rows = max(1, int(_SCRATCH_BYTES // (nr * group * 8 * 6)))
        s, _ = exact_group_scale(Vg, C, hess_diag=wg, max_elems=rows * group * L)
    else:
        s, _ = search_group_scale(Vg, C, hess_diag=wg)
    return s.reshape(O, G)


@torch.no_grad()
def _col_err(W: torch.Tensor, C: torch.Tensor, R: torch.Tensor,
             c: torch.Tensor) -> torch.Tensor:
    """Per-column squared error in W-space: sum_i (code*R*c - W)^2. R is [O, I]."""
    S = R * c.view(1, -1)
    q = C[nearest_codeword(W / S, C)] * S
    return ((q - W) ** 2).sum(0)


@torch.no_grad()
def equalize_columns(mats, C, groups, hdiag=None, mode: str = "int",
                     iters: int = 4, tie: torch.Tensor | None = None,
                     int_range: int = INT_RANGE, n_grid: int = 33,
                     span: float = 1.0, exact_r: bool = False):
    """Fit ONE per-input-column diagonal `c` shared by every matrix in `mats`.

    Args:
        mats:   list of [O_k, I] fp32 weights that share an input (and hence share c).
        C:      Codebook or sorted level tensor.
        groups: per-matrix group width (list, same length as `mats`).
        hdiag:  [I] activation salience h = diag(H). None -> uniform (the ablation that
                removes the AWQ ingredient while keeping the lever).
        mode:   'int'  -> c_j = 2^{t_j}, t integer, exact enumeration over +-int_range.
                'real' -> c_j unconstrained positive, multiplicative grid, halving span.
        tie:    optional [I] long tensor; columns with the same value share one c.
        iters:  alternations (each half-step is an exact minimization of the same J).

    Returns:
        (c [I] float32, info dict)
    """
    assert mode in ("int", "real")
    dev = mats[0].device
    C = _levels(C).to(dev, torch.float32)
    I = mats[0].shape[1]
    for M in mats:
        assert M.shape[1] == I, "mats sharing a diagonal must share in_features"
    h = (torch.ones(I, device=dev, dtype=torch.float32) if hdiag is None
         else hdiag.to(dev, torch.float32).clamp_min(1e-30))
    c = torch.ones(I, device=dev, dtype=torch.float32)

    if mode == "int":
        cand = torch.exp2(torch.arange(-int_range, int_range + 1, device=dev,
                                       dtype=torch.float32))
    n_ties = I if tie is None else int(tie.max()) + 1

    def objective(c_):
        tot = 0.0
        for M, g in zip(mats, groups):
            V = M / c_.view(1, -1)
            r = _fit_r(V, C, g, h * c_ ** 2, exact_r)
            R = r.repeat_interleave(g, dim=1)
            tot += float((_col_err(M, C, R, c_) * h).sum())
        return tot

    j0 = objective(c)
    sp = span
    for it in range(iters):
        Rs = []
        for M, g in zip(mats, groups):
            V = M / c.view(1, -1)
            r = _fit_r(V, C, g, h * c ** 2, exact_r)
            Rs.append(r.repeat_interleave(g, dim=1))
        if mode == "int":
            offs = cand
        else:
            offs = torch.exp2(torch.linspace(-sp, sp, n_grid, device=dev))
            sp *= 0.5
        E = torch.empty(offs.numel(), n_ties, device=dev, dtype=torch.float64)
        for k, o in enumerate(offs):
            e = torch.zeros(I, device=dev, dtype=torch.float32)
            for M, R in zip(mats, Rs):
                e += _col_err(M, C, R, c * o)
            e = (e.double() * h.double())
            if tie is None:
                E[k] = e
            else:
                E[k] = torch.zeros(n_ties, device=dev, dtype=torch.float64
                                   ).index_add_(0, tie, e)
        best = E.argmin(0)                                     # [n_ties]
        m = offs[best] if tie is None else offs[best][tie]
        c = c * m
    j1 = objective(c)
    info = {"mode": mode, "iters": iters, "obj_before": j0, "obj_after": j1,
            "obj_rel": (j1 - j0) / max(j0, 1e-30),
            "n_ties": n_ties,
            "log2c_min": float(torch.log2(c).min()),
            "log2c_max": float(torch.log2(c).max()),
            "log2c_absmean": float(torch.log2(c).abs().mean())}
    return c, info


# --------------------------------------------------------------- exactness check
@torch.no_grad()
def apply_reparam_check(W: torch.Tensor, x: torch.Tensor, c: torch.Tensor):
    """(W diag(c)^-1) @ (diag(c) x) == W @ x. Returns the max abs deviation."""
    a = W @ x
    b = (W / c.view(1, -1)) @ (c.view(-1, 1) * x)
    return float((a - b).abs().max())
