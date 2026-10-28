"""
GPTQ / LDLQ with an arbitrary scalar codebook oracle, and the in-loop rescale fix.

Why this file exists
--------------------
QuIP#/BlockLDLQ give the formal license: GPTQ is adaptive rounding with linear feedback
parameterized by an arbitrary rounding oracle Q; a scalar non-uniform codebook needs no
algorithmic change, only `round()` -> `nearest_codeword()`. But PoT-PTQ (arXiv:2507.11959,
Table 2) reports that doing exactly that gives 8.27e4 PPL on LLaMA1-7B and NaN on
LLaMA2-7B.

The diagnosis this file acts on: GPTQ *mutates* the not-yet-quantized columns as it
sweeps -- that is the entire point of the error feedback. A group scale computed once,
up front, from the ORIGINAL W is therefore stale by the time the sweep reaches that group.
On a uniform grid a stale scale costs a fraction of a step. On a log grid, whose levels
are a factor of two apart, a stale scale misassigns by WHOLE EXPONENTS: the reconstruction
is off by 2x, the residual fed forward is huge, it inflates the next group's weights,
which makes ITS scale staler still. That is a divergence, and it is what 8.27e4 looks like.

`rescale_in_loop=True` re-runs the per-group scale search on the CURRENT, already
compensated columns at each group boundary. `rescale_in_loop=False` reproduces the naive
behaviour exactly, so the two can be ablated against each other.

Everything else is canonical GPTQ (Frantar et al.), kept deliberately close to the
reference implementation so it can be checked line by line:
    H  = (2/n) sum_t X_t^T X_t, accumulated as a running average
    H += percdamp * mean(diag H) * I
    Hinv = cholesky( cholesky_inverse( cholesky(H) ), upper=True )
    for each column i:  q = Q(w_i);  err = (w_i - q) / Hinv[i,i]
                        W[:, i+1:] -= err (x) Hinv[i, i+1:]

Column scale
------------
`col_scale` c folds into the activation (x' = c*x) and out of the weight (W' = W/c),
leaving the layer output unchanged. One can show the two formulations are IDENTICAL, not
merely similar: H' = diag(c) H diag(c), and the upper Cholesky factor of Hinv' is
U' = U diag(c)^-1, so d'_i = U_ii/c_i, err'_i = (w_i-q_i)/U_ii = err_i, and the update to
column j in original coordinates is -err_i * U[i,j] either way. So we run the sweep in the
ORIGINAL coordinates with the original H, and simply use the per-column effective scale
s[:, j] = r[:, g(j)] * c[j] in the rounding oracle. That keeps the arithmetic in the
weights' own scale and, importantly, makes the stored artifact exactly `code * (r*c)`,
which is what the shift-free verifier checks bit-for-bit.

Two caveats on that equivalence, both verified in tests/pot/test_gptq_pot.py:
  * it is exact only at percdamp=0, because `percdamp * mean(diag H) * I` is not
    scale-covariant: mean(c_i^2 H_ii) != c_i^2 mean(H_ii);
  * it is an identity in EXACT arithmetic. In float32, (w/c)/r and w/(r*c) differ by an
    ulp, and on a grid whose levels are a factor of two apart an ulp at a decision
    boundary flips a whole exponent. So individual codes diverge between the two
    formulations (~3% of them) while the achieved proxy error agrees to ~0.2%. This is
    also a standing warning about the log grid itself: its rounding decisions are
    razor-edged, which is precisely why the scale must be searched rather than solved.
"""

from __future__ import annotations

import math

import torch

from .codebooks import Codebook, nearest_codeword
from .scale_search import search_group_scale, pot_round
from .exact_scale import exact_group_scale
from .log_scale import (log_group_scale, log_nearest_codeword, clamp_diagnostics)

# Half the relative gap between adjacent strict-PoT rungs: the midpoint between
# 2^e and 2^(e-1) sits at 1.5*2^(e-1), so a value v is at most v/3 from its rung
# and the representable step around v is ~v/3.5. Used only by feedback_clip.
_LOG_STEP_FRAC = 1.0 / 3.5

__all__ = ["PoTGPTQ", "quantize_linear"]


def _levels(C):
    return C.levels if isinstance(C, Codebook) else C


class PoTGPTQ:
    """GPTQ for one nn.Linear (or a raw [out, in] weight), with a codebook oracle.

    Usage:
        g = PoTGPTQ(linear)
        for X in calib_batches: g.add_batch(X)         # X: [..., in_features]
        out = g.quantize(codebook, group=128)
        linear.weight.data = out['W_hat'].to(linear.weight.dtype)
    """

    def __init__(self, layer, device=None):
        if isinstance(layer, torch.nn.Module):
            W = layer.weight.data
        else:
            W = layer
        assert W.dim() == 2, "expected a 2-D [out_features, in_features] weight"
        self.layer = layer if isinstance(layer, torch.nn.Module) else None
        self.dev = device or W.device
        self.W = W.detach().to(self.dev, torch.float32).clone()
        self.rows, self.columns = self.W.shape
        self.H = torch.zeros(self.columns, self.columns, device=self.dev,
                             dtype=torch.float32)
        self.nsamples = 0
        # First moment of the activations, accumulated as a PLAIN sum alongside H.
        # `H` is a running average of (2/n) X^T X and `nsamples` is reset to 1 when
        # --method rtn overwrites H with the identity, so neither survives as a record
        # of the calibration statistics. These two do, and nothing else reads them:
        # they exist for pot.out_correct's post-hoc per-output gain/bias, which needs
        # E[x] and cannot recover it from a second moment.
        self.xsum = torch.zeros(self.columns, device=self.dev, dtype=torch.float64)
        self.ntokens = 0

    # ------------------------------------------------------------------ Hessian
    @torch.no_grad()
    def add_batch(self, X: torch.Tensor):
        """Accumulate H as the canonical GPTQ running average of (2/n) X^T X."""
        X = X.detach()
        if X.dim() > 2:
            X = X.reshape(-1, X.shape[-1])
        X = X.to(self.dev, torch.float32)
        assert X.shape[1] == self.columns, (
            f"activation has {X.shape[1]} features, weight expects {self.columns}")
        t = X.shape[0]
        self.xsum += X.sum(0).double()          # BEFORE the sqrt(2/n) rescale below
        self.ntokens += t
        n = self.nsamples
        self.H *= n / (n + t)
        self.nsamples = n + t
        X = X * math.sqrt(2 / self.nsamples)
        self.H += X.t() @ X

    def free(self):
        self.H = None
        self.W = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def _group_scale(self, Wsub: torch.Tensor, C, hdiag=None, exact_scale=False,
                     log_scale=False, **kw):
        """Best per-row scale for one group of columns. Wsub: [rows, g].

        `exact_scale=True` replaces the `n_grid`/`lo`/`hi` grid search with the exact
        breakpoint-based search in `exact_scale.py` (same objective, no sampling). The
        two are mutually exclusive: when `exact_scale` is set, `n_grid`/`lo`/`hi` in
        `kw` are simply unused (exact search has no grid hyperparameters).

        `log_scale=True` replaces the OBJECTIVE itself with the format-native
        log-domain distortion `sum_i h_i (log2|w_hat_i| - log2|w_i|)^2` and solves it
        exactly by the same kind of breakpoint argument (`log_scale.py`). It also has
        no grid hyperparameters, and it supersedes `exact_scale` when both are set
        (the caller is expected to forbid that combination).
        """
        offset = kw.pop("scale_offset_log2", 0.0) or 0.0
        rnd = kw.pop("scale_round_log2", None)
        f16 = kw.pop("scale_fp16", False)
        if log_scale:
            s, _ = log_group_scale(Wsub, C, hess_diag=hdiag,
                                   pot_scale=kw.get("pot_scale", False))
        elif exact_scale:
            s, _ = exact_group_scale(Wsub, C, hess_diag=hdiag,
                                     pot_scale=kw.get("pot_scale", False))
        else:
            s, _ = search_group_scale(Wsub, C, hess_diag=hdiag, **kw)
        return self._post_scale(s, offset, rnd, f16)                    # [rows, 1]

    @staticmethod
    def _post_scale(s, offset_log2=0.0, round_log2=None, fp16=False):
        """Post-transform the searched scale. Both transforms default to inert.

        `offset_log2` (delta) multiplies every scale by 2**delta -- a SYSTEMATIC,
        equal-for-every-group shift of the representable window. It is the direct probe
        of the one continuous parameter the format has: raising the window trades
        bottom-rung clamping (unbounded relative error, the format's only such failure
        mode) against top-rung clipping (bounded). The searched scale is the exact
        minimiser of group MSE; whether that is also the ACCURACY optimum has never been
        measured, and this option measures it.

        `round_log2` (Delta) snaps log2(s) to a multiple of Delta -- a RANDOM,
        group-by-group perturbation of magnitude <= Delta/2 octaves with no systematic
        direction. Two uses: (a) it is the perturbation-sensitivity probe that the
        offset sweep is not, and (b) it is exactly the quantisation a low-precision
        stored scale would suffer, so the sweep prices how many bits a per-group scale
        needs before any such scheme is built.

        Both leave bpw untouched -- the scale is still stored as fp16 -- so arms using
        them are diagnostics at the incumbent's exact bit cost, not deliverables.
        """
        if offset_log2:
            s = s * float(2.0 ** offset_log2)
        if round_log2:
            s = torch.exp2(torch.round(torch.log2(s.clamp_min(1e-30)) / round_log2)
                           * round_log2)
        if fp16:
            s = s.half().float()
        return s

    # ------------------------------------------------------- scale-search weights
    SCALE_WEIGHTS = ("none", "diag", "ldl", "invdiag")

    POWDIAG = "powdiag:"

    @staticmethod
    def _parse_powdiag(mode):
        """`'powdiag:<alpha>'` -> float alpha, else None.

        `diag(H)^alpha` is the one-parameter family that contains every other value here:
        alpha=+1 is 'diag', alpha=0 is 'none' (uniform), alpha=-1 is 'invdiag'. It exists
        so the response of the downstream metric to `tr(E H E^T)` can be DIALED to a
        target on the degradation side (alpha in (-1, 0)) instead of sampled at one point.
        Diagnostic only, exactly like 'invdiag' -- see `_scale_weights`.
        """
        if isinstance(mode, str) and mode.startswith(PoTGPTQ.POWDIAG):
            try:
                return float(mode[len(PoTGPTQ.POWDIAG):])
            except ValueError:
                raise ValueError(f"bad powdiag exponent in {mode!r}")
        return None

    @staticmethod
    def _norm_scale_weight(hess_weighted_scale):
        """Normalize the historical bool / the new string into one of SCALE_WEIGHTS.

        The boolean spelling is load-bearing: every run recorded in
        results/pot/pot_runs.jsonl passed a bool, so `False -> 'none'` and
        `True -> 'diag'` must hold forever or those runs stop being reproducible.
        """
        if hess_weighted_scale is False or hess_weighted_scale is None:
            return "none"
        if hess_weighted_scale is True:
            return "diag"
        if hess_weighted_scale in PoTGPTQ.SCALE_WEIGHTS:
            return hess_weighted_scale
        if PoTGPTQ._parse_powdiag(hess_weighted_scale) is not None:
            return hess_weighted_scale
        raise ValueError(
            f"unknown hess_weighted_scale {hess_weighted_scale!r}; expected a bool or "
            f"one of {PoTGPTQ.SCALE_WEIGHTS}")

    @staticmethod
    def _scale_weights(mode, Hdamped, Hinv):
        """Per-column weights `h_j` for the group scale search, in SWEEP coordinates.

        Why 'ldl' and not 'diag'. The scale search minimizes `sum_j h_j (q_j - w_j)^2`
        over one group. The quantity GPTQ actually minimizes is `tr(E H E^T)`, and under
        the LDL identity the sweep is built on that decomposes EXACTLY as

            tr(E H E^T) = sum_j  ||w_j - q_j||^2 / Hinv[j,j]^2

        where `w_j` is the already-compensated column and `Hinv` is the upper Cholesky
        factor of H^-1. So the additive per-column weight implied by GPTQ's own objective
        is `1/Hinv[j,j]^2` -- the CONDITIONAL precision of column j given all columns
        processed after it -- not `diag(H)`, which is the MARGINAL precision and ignores
        that the sweep gets to push column j's residual into its successors.

        Measured on 25 real Qwen3-1.7B linears (blocks 0/7/14/20/27 x q/v/o/gate/down,
        g=64, strict_pot8), against the uniform-weight default:

            'diag'  ->  tr(E H E^T)  -9.0 %  (sd 8.0)
            'ldl'   ->  tr(E H E^T) -10.7 %  (sd 8.6),  negative on 25/25 layers

        i.e. 'ldl' strictly dominates 'diag' at IDENTICAL cost -- both are a [cols]
        vector handed to the same search. It is not, however, expected to move ARC-Easy:
        see llmdocs and FINDINGS_C, where a -9.0 % objective cut (this flag's 'diag'
        value) measured 0.0 pp over five draws.
        """
        if mode == "none":
            return None
        if mode == "diag":
            return torch.diag(Hdamped)
        if mode == "ldl":
            return 1.0 / torch.diagonal(Hinv).clamp_min(1e-30) ** 2
        if mode == "invdiag":
            # DIAGNOSTIC ONLY -- NEVER a deliverable. This is deliberately the WRONG
            # weight: the reciprocal of 'diag', so the scale search spends its accuracy
            # on the columns the output error cares about LEAST. It exists as the
            # falsification control for the "monotone" hypothesis (three interventions
            # that improved tr(E H E^T) all moved ARC-Easy nominally DOWN, ordered by
            # size of the improvement). If optimizing this objective harder really costs
            # accuracy, deliberately optimizing it WORSE should pay. Any good ARC-Easy
            # score from this value is a statement about the metric, not about the
            # method, and must not be promoted to the artifact.
            return 1.0 / torch.diag(Hdamped).clamp_min(1e-30)
        a = PoTGPTQ._parse_powdiag(mode)
        if a is not None:
            # DIAGNOSTIC ONLY, same standing rule as 'invdiag'. diag(H)^alpha: alpha=+1
            # reproduces 'diag', alpha=0 is uniform, alpha=-1 reproduces 'invdiag'.
            # Intermediate negative alpha dials the objective degradation continuously,
            # which is what maps out the response curve on the degradation side.
            return torch.diag(Hdamped).clamp_min(1e-30) ** a
        raise ValueError(f"unknown scale weight mode {mode!r}")

    # --------------------------------------------------------------- orderings
    ORDERS = ("none", "act", "err", "errasc", "errmag",
              "logmean", "logmeanasc", "salogmean", "rand")

    @torch.no_grad()
    def _rtn_col_err(self, W, C, group, sk, col_scale, hdiag_for_scale, nc=None):
        """Per-column squared PoT quantization error `e_j = sum_i (W_ij - Q_ij)^2` from a
        cheap RTN pre-pass on the ORIGINAL W, using the pipeline's own scale search and
        contiguous groups. One extra pass over the matrix; no error feedback.
        """
        rows, cols = W.shape
        e = torch.zeros(cols, device=W.device, dtype=torch.float32)
        for gi in range(cols // group):
            sl = slice(gi * group, (gi + 1) * group)
            sub = W[:, sl] if col_scale is None else W[:, sl] / col_scale[sl]
            hd = None if hdiag_for_scale is None else hdiag_for_scale[sl]
            s = self._group_scale(sub, C, hd, **sk)
            q = C[(nc or nearest_codeword)(sub / s, C)] * s
            if col_scale is not None:
                q = q * col_scale[sl]
            e[sl] = ((W[:, sl] - q) ** 2).sum(0)
        return e

    @torch.no_grad()
    def _column_perm(self, order, W, C, Hdiag0, group, sk, col_scale,
                     hdiag_for_scale, nc=None, perm_hess=None, perm_seed=0):
        """Column processing order. Returns None for identity (no g_idx permutation).

        `act` is descending diag(H): the canonical GPTQ heuristic, which is *exactly*
        right on a uniform grid because there every column's rounding error is bounded by
        the same Delta/2, so the column's contribution to the output error, ~ H_jj * e_j,
        is proportional to H_jj alone. On a log grid e_j is proportional to the column's
        magnitude and spans orders of magnitude, so it does not factor out and diag(H) is
        no longer the right key. `err` sorts on the measured product H_jj * e_j and
        `errmag` on the pre-pass-free proxy H_jj * ||W[:,j]||^2. `errasc` is the ascending
        falsification control for `err`.

        The log-magnitude family (`logmean`, `logmeanasc`, `salogmean`) is derived from a
        different place. Under `--method rtn` the sweep has H := I, so Hinv is the
        identity, no residual ever crosses a column boundary, and the permutation's ONLY
        effect is on GROUP MEMBERSHIP: groups are contiguous slices of the PERMUTED column
        order (see `quantize`, `W = W[:, perm]` before the group loop). The partition is
        therefore the one degree of freedom that the per-column argmin does not cancel
        (equalize.py's covariance theorem), and what a group's partition controls is its
        within-group DYNAMIC RANGE -- how many members fall more than `nexp - 1` octaves
        below the scale and clamp onto the bottom rung.

        So the key is a column's LOCATION in log space, `mu_j = mean_i log2|W_ij|`
        (`logmean`). `salogmean` is its salience-weighted form, obtained by applying the
        same geometric mean to the per-weight output energy `h_j W_ij^2` of section 21.1:
        `k_j = mean_i log2(h_j W_ij^2) = log2 h_j + 2 mu_j`. The 2 is fixed by the error
        model, not tuned.

        Two consequences constrain what the controls can be: ascending vs descending is a
        NULL change under rtn (reversing a permutation maps group g to group G-1-g and
        leaves every group SET identical when cols % group == 0), so `logmeanasc` is a
        machinery test, not a falsification control; the real control is `rand`, a random
        permutation with exactly the same g_idx cost.
        """
        if order == "none":
            return None
        if order == "act":
            return torch.argsort(Hdiag0, descending=True)
        if order == "rand":
            # FALSIFICATION CONTROL. A permutation with exactly the same g_idx cost and
            # the same group sizes as any derived order, but no relation to the weights.
            # Deterministic given perm_seed (CPU generator, so it is device-independent).
            gen = torch.Generator().manual_seed(int(perm_seed) + W.shape[1])
            return torch.randperm(W.shape[1], generator=gen).to(W.device)
        if order in ("logmean", "logmeanasc", "salogmean"):
            # See the module note on log-magnitude grouping. mu_j is the column's
            # LOCATION in log space -- the geometric-mean magnitude -- which is the
            # right average on an alphabet that is uniform in log space.
            mu = torch.log2(W.abs().clamp_min(1e-30)).mean(0)
            if order == "salogmean":
                # Section 21.1: PoT salience is the per-weight output energy w_ij^2 h_j.
                # Reduced to a COLUMN statistic by the same geometric mean:
                #   k_j = mean_i log2(h_j W_ij^2) = log2 h_j + 2 mu_j.
                # The coefficient on mu is fixed at 2 by the error model, so this form
                # has no free parameter to tune.
                h = Hdiag0 if perm_hess is None else perm_hess.to(W.device, torch.float32)
                assert h.numel() == W.shape[1], (
                    f"perm_hess has {h.numel()} entries, expected {W.shape[1]}")
                key = 2.0 * mu + torch.log2(h.clamp_min(1e-30))
            else:
                key = mu
            return torch.argsort(key, descending=(order != "logmeanasc"))
        if order == "errmag":
            key = Hdiag0 * (W ** 2).sum(0)
        elif order in ("err", "errasc"):
            key = Hdiag0 * self._rtn_col_err(W, C, group, sk, col_scale,
                                             hdiag_for_scale, nc=nc)
        else:
            raise ValueError(f"unknown order {order!r}; expected one of {self.ORDERS}")
        return torch.argsort(key, descending=(order != "errasc"))

    # ----------------------------------------------------------------- quantize
    @torch.no_grad()
    def quantize(self, codebook, group: int = 128, percdamp: float = 0.01,
                 blocksize: int = 128, act_order: bool = False,
                 rescale_in_loop: bool = True, col_scale: torch.Tensor | None = None,
                 pot_scale: bool = False, n_grid: int = 200, lo: float = 0.3,
                 hi: float = 1.2, hess_weighted_scale: bool | str = False,
                 exact_scale: bool = False, log_scale: bool = False,
                 log_round: bool = False, order: str | None = None,
                 feedback_damp: float = 1.0, feedback_clip: float | None = None,
                 scale_hess: torch.Tensor | None = None,
                 scale_offset_log2: float = 0.0,
                 scale_round_log2: float | None = None,
                 scale_fp16: bool = False,
                 perm_hess: torch.Tensor | None = None, perm_seed: int = 0,
                 return_targets: bool = False, verbose: bool = False):
        """Run the GPTQ sweep.

        Args:
            codebook:        Codebook (or sorted 1-D level tensor).
            group:           columns per scale group; <=0 or >= in_features means one
                             group per row.
            percdamp:        Hessian damping, as a fraction of mean(diag H).
            blocksize:       lazy-update block width. Must be a multiple of `group`
                             (else a group would straddle a stale block boundary).
            act_order:       process columns in decreasing diag(H) order. Kept as an
                             alias for `order='act'`; used only when `order` is None.
            order:           column processing order, one of
                             'none' (identity), 'act' (descending diag(H)),
                             'err' (descending H_jj * e_j with e_j the measured PoT
                             error of column j from an RTN pre-pass), 'errmag'
                             (descending H_jj * ||W[:,j]||^2, the pre-pass-free proxy),
                             'errasc' (ascending 'err', a falsification control),
                             'logmean' (descending mean_i log2|W_ij|, the column's log
                             location -- groups columns of similar log-magnitude so the
                             within-group dynamic range, and hence bottom-rung clamping,
                             falls), 'logmeanasc' (its reversal; a NULL change under rtn),
                             'salogmean' (log2 h_j + 2 mu_j, the salience-weighted form)
                             or 'rand' (a random permutation, the falsification control).
                             Default None means "take it from `act_order`", so leaving
                             this argument alone reproduces the previous behaviour
                             bit-for-bit.
            rescale_in_loop: THE point of this file. Re-search each group's scale on the
                             already-compensated columns at the group boundary. False
                             reproduces the naive stale-scale behaviour for ablation.
            col_scale:       optional [in_features] per-column multiplicative scale,
                             folded into the activation at inference time. Ablation only.
            pot_scale:       constrain the group scale to a power of two.
            hess_weighted_scale: per-column weights `h_j` for the scale-search error.
                             One of the values in `SCALE_WEIGHTS`:
                               False / 'none'  -- uniform weights (the DEFAULT; this is
                                     bit-for-bit the behaviour from before this option
                                     grew a third value, asserted in
                                     tests/pot/test_scale_weight.py);
                               True / 'diag'   -- weight by diag(H). Unchanged meaning of
                                     the historical boolean flag; `True` is still exactly
                                     this and nothing else;
                               'ldl'           -- weight by 1/Hinv[j,j]^2, the weight
                                     GPTQ's OWN objective decomposition implies;
                               'invdiag'       -- weight by 1/diag(H). DELIBERATELY WRONG,
                                     a falsification control, never a deliverable;
                               'powdiag:<a>'   -- weight by diag(H)^a. The family that
                                     contains the three above (a=1/0/-1). Diagnostic only.
                             See `_scale_weights` for why 'ldl' is the correct one and
                             'diag' is not.
            exact_scale:     replace the `n_grid`/`lo`/`hi` grid scale search with the
                             exact breakpoint-based search (`exact_scale.py`). Default
                             OFF; with it off this function is bit-for-bit unchanged
                             from before this argument existed.
            log_scale:       fit the group scale by minimizing the FORMAT-NATIVE
                             log-domain distortion `sum_i h_i (log2|w_hat|-log2|w|)^2`
                             instead of squared error on weights, solved exactly
                             (`log_scale.py`). Supersedes `exact_scale`. Default OFF.
            log_round:       assign codes by nearest-in-LOG (geometric-midpoint
                             boundary `sqrt(2)*2^(e-1)`) instead of nearest-in-LINEAR
                             (arithmetic midpoint `1.5*2^(e-1)`). Independent switch
                             from `log_scale` so the 2x2 can be ablated. Default OFF.
                             With both OFF this function is bit-for-bit unchanged.
            perm_hess:       [cols] per-column salience used by order='salogmean'.
                             Needed because --method rtn overwrites self.H with I BEFORE
                             quantize() runs, so diag(H) is no longer the calibration
                             quantity; run_pot.py captures the real one first and passes
                             it here. None means "use diag(H)".
            perm_seed:       seed for order='rand'. Drawn from a CPU generator seeded with
                             perm_seed + cols, so it is deterministic and
                             device-independent.
            feedback_damp:   scalar lambda in [0, 1] on the residual propagated to later
                             columns. 1.0 (default) is stock GPTQ, bit-for-bit. 0.0
                             propagates nothing. Motivated by the measured fact that the
                             error feedback is worth +1.21 pp to int4_sym and -2.21 pp to
                             strict_pot at identical bpw.
            feedback_clip:   grid-aware cap kappa: each element's correction is clipped to
                             kappa * |w| / 3.5, the local representable step of a log grid.
                             None (default) disables the clip entirely. This is the
                             PoT-native half -- the cap tracks |w| because the log grid's
                             resolution tracks |w|.
            scale_hess:      explicit [in_features] per-column weights `h_j` for the
                             group scale search, overriding `hess_weighted_scale`
                             entirely (it is permuted alongside the columns when an
                             `order` is used). This exists because under `--method rtn`
                             self.H is deliberately the IDENTITY, so no value of
                             `hess_weighted_scale` can express "weight the scale fit by
                             the real calibration salience" -- which is exactly what
                             `equalize.py` needs, together with the c^2 factor that makes
                             the objective consistent in W-space. None (default) leaves
                             this function bit-for-bit unchanged.
            return_targets:  also return 'W_tgt' [out, in] fp32, the ALREADY-COMPENSATED
                             column values that the rounding oracle actually saw
                             (`W_hat = C[nearest_codeword(W_tgt / S)] * S` exactly, at the
                             scale the sweep used). This is the correct target for any
                             post-hoc scale refinement: re-rounding against the original W
                             would throw away the GPTQ feedback, and re-rounding against
                             W_hat is a no-op. Costs one extra [out, in] fp32 tensor.

        Returns:
            dict with W_hat [out, in] fp32, codes (uint8), scales [out, n_groups],
            g_idx [in], col_scale, and error/diagnostic fields.
        """
        C = _levels(codebook).to(self.dev, torch.float32)
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

        # clone: the damping and dead-column fixups below are in-place, and self.H must
        # stay pristine so the same accumulated Hessian can be re-quantized with
        # different settings (that is exactly what the ablations do).
        H = self.H.clone()
        assert self.nsamples > 0, "call add_batch() before quantize()"

        # dead columns: an input feature never activated carries no information; GPTQ's
        # reference zeroes them so the Cholesky stays PD.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if col_scale is not None:
            col_scale = col_scale.to(self.dev, torch.float32).clamp_min(1e-12)
            assert col_scale.numel() == cols
        Hdiag0 = torch.diag(H).clone()

        # `sk` (scale-search kwargs) and `hdiag_for_scale` are needed by the `err`
        # pre-pass, so they are built here, before the permutation is chosen.
        nc = log_nearest_codeword if log_round else nearest_codeword
        sk = dict(n_grid=n_grid, lo=lo, hi=hi, pot_scale=pot_scale)
        if exact_scale:
            sk = dict(pot_scale=pot_scale, exact_scale=True)
        if log_scale:
            sk = dict(pot_scale=pot_scale, log_scale=True)
        sk["scale_offset_log2"] = scale_offset_log2
        sk["scale_round_log2"] = scale_round_log2
        sk["scale_fp16"] = scale_fp16
        # `hdiag_for_scale` is needed by the `err` pre-pass BEFORE the permutation is
        # chosen, i.e. before Hinv exists. 'ldl' weights are therefore unavailable at
        # pre-pass time and the pre-pass runs unweighted; the pre-pass is an ordering
        # heuristic on a throwaway RTN pass, not part of the artifact, so this only
        # affects which permutation `--order err/errasc` picks, never the sweep itself.
        # `--order act` (the arm this option was added for) needs no pre-pass at all.
        scale_weight = self._norm_scale_weight(hess_weighted_scale)
        hdiag_for_scale = Hdiag0 if scale_weight == "diag" else None

        if order is None:
            order = "act" if act_order else "none"
        if order not in self.ORDERS:
            raise ValueError(f"unknown order {order!r}; expected one of {self.ORDERS}")

        perm = self._column_perm(order, W, C, Hdiag0, group, sk, col_scale,
                                 hdiag_for_scale, nc=nc, perm_hess=perm_hess,
                                 perm_seed=perm_seed)
        if perm is not None:
            W = W[:, perm]
            H = H[perm][:, perm]
            if col_scale is not None:
                col_scale = col_scale[perm]

        damp = percdamp * torch.mean(torch.diag(H))
        H[range(cols), range(cols)] += damp
        # Hinv := upper Cholesky factor of H^-1
        L = torch.linalg.cholesky(H)
        Hi = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hi, upper=True)

        # The real weights, in the (possibly permuted) sweep coordinates and damped like
        # the sweep's own H. 'ldl' needs Hinv, so it can only be built here.
        hdiag_for_scale = self._scale_weights(scale_weight, H, Hinv)
        if scale_hess is not None:
            hs = scale_hess.to(self.dev, torch.float32)
            assert hs.numel() == cols, (
                f"scale_hess has {hs.numel()} entries, expected {cols}")
            hdiag_for_scale = hs[perm] if perm is not None else hs

        Q = torch.zeros_like(W)
        T = torch.zeros_like(W) if return_targets else None
        codes = torch.zeros(rows, cols, dtype=torch.uint8, device=self.dev)
        scales = torch.zeros(rows, n_groups, device=self.dev, dtype=torch.float32)
        Losses = torch.zeros_like(W)

        if not rescale_in_loop:
            # NAIVE path: every group scale is computed ONCE, up front, from the original
            # (un-compensated) W. This is what PoT-PTQ reports diverging.
            for gi in range(n_groups):
                sl = slice(gi * group, (gi + 1) * group)
                hd = None if hdiag_for_scale is None else hdiag_for_scale[sl]
                sub = W[:, sl] if col_scale is None else W[:, sl] / col_scale[sl]
                scales[:, gi] = self._group_scale(sub, C, hd, **sk).squeeze(1)

        cur_s = None                                     # [rows, 1] active group scale
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
                        # ---- THE FIX ----
                        # search the scale on the CURRENT, already-compensated columns of
                        # this block, not on the original W. Everything from i1 to i2 has
                        # already absorbed the feedback from every previously quantized
                        # column, and the group lies entirely inside this block (enforced
                        # by the blocksize/group divisibility check), so W1 is exact here.
                        sub = W1[:, i:i + group]
                        if col_scale is not None:
                            sub = sub / col_scale[col:col + group]
                        hd = (None if hdiag_for_scale is None
                              else hdiag_for_scale[col:col + group])
                        cur_s = self._group_scale(sub, C, hd, **sk)
                        scales[:, gi] = cur_s.squeeze(1)
                    else:
                        cur_s = scales[:, gi:gi + 1]

                w = W1[:, i]
                if T is not None:
                    T[:, col] = w
                d = Hinv1[i, i]
                s = cur_s.squeeze(1)
                if col_scale is not None:
                    s = s * col_scale[col]
                idx = nc(w / s, C)
                q = C[idx] * s

                Q1[:, i] = q
                codes[:, col] = idx.to(torch.uint8)
                Losses[:, col] = (w - q) ** 2 / d ** 2

                err = (w - q) / d

                # ---- PoT-NATIVE FEEDBACK CONTROL -------------------------------
                # GPTQ retargets the remaining columns to absorb this column's
                # residual. On a UNIFORM grid every retarget is equally
                # representable (error <= Delta/2 wherever you move it), so the
                # trade always pays. On a LOG grid the representable step at a
                # value v is ~|v|/3.5, so moving a weight to a LARGER magnitude
                # moves it to a COARSER part of the grid and the retarget buys
                # less than it costs. Measured: GPTQ - RTN is +1.21 pp for
                # int4_sym and -2.21 pp for strict_pot at identical bpw.
                #
                # feedback_damp: scalar lambda on the propagated residual.
                #   1.0 = stock GPTQ (default, bit-for-bit unchanged)
                #   0.0 = no propagation at all
                # feedback_clip: grid-aware cap. Limits each element's correction
                #   to kappa * (local representable step), i.e. never ask a weight
                #   to move further than its own grid can follow. kappa=None
                #   disables. This is the PoT-specific half: the cap is
                #   proportional to |w| exactly because the log grid's resolution
                #   is proportional to |w|.
                corr = err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                if feedback_clip is not None:
                    step = W1[:, i:].abs() * _LOG_STEP_FRAC
                    cap = feedback_clip * step
                    corr = torch.clamp(corr, -cap, cap)
                if feedback_damp != 1.0:
                    corr = corr * feedback_damp
                # column i itself must still land exactly on q, whatever the
                # feedback settings do to the columns after it.
                W1[:, i:] -= corr
                W1[:, i] = q
                # E1 carries the residual across the block boundary, so it must be
                # damped identically or the cross-block update would reintroduce
                # the full-strength feedback the in-block update just suppressed.
                E1[:, i] = err * feedback_damp

            Q[:, i1:i2] = Q1
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
            if verbose:
                print(f"  cols {i1}-{i2}  loss={Losses[:, i1:i2].sum().item():.4g}",
                      flush=True)

        if perm is not None:
            inv = torch.argsort(perm)
            Q = Q[:, inv]
            codes = codes[:, inv]
            if T is not None:
                T = T[:, inv]
            if col_scale is not None:
                col_scale = col_scale[inv]
            g_idx = (torch.arange(cols, device=self.dev) // group)[inv]
        else:
            g_idx = torch.arange(cols, device=self.dev) // group

        # Rebuild W_hat from the STORED parameters, exactly as an inference kernel would.
        # This is not cosmetic: it guarantees the artifact the verifier sees is literally
        # code * scale, with no residue of the intermediate arithmetic in the sweep.
        S = scales[:, g_idx]
        if col_scale is not None:
            S = S * col_scale.view(1, -1)
        W_hat = C[codes.long()] * S
        assert torch.equal(W_hat, Q), "dequant from stored params disagrees with the sweep"

        Wref = self.W
        err = (W_hat - Wref)
        out = {
            "W_hat": W_hat,
            "codes": codes,
            "scales": scales,
            "g_idx": g_idx,
            "col_scale": col_scale,
            "group": group,
            "n_groups": n_groups,
            "act_order": act_order,
            "scale_weight": scale_weight,
            "order": order,
            "rescale_in_loop": rescale_in_loop,
            "pot_scale": pot_scale,
            "exact_scale": exact_scale,
            "scale_offset_log2": scale_offset_log2,
            "scale_round_log2": scale_round_log2,
            "scale_fp16": scale_fp16,
            "log_scale": log_scale,
            "log_round": log_round,
            "gptq_loss": float(Losses.sum()),
            "rel_fro_err": float(err.norm() / Wref.norm().clamp_min(1e-30)),
            "rel_mse": float((err ** 2).sum() / (Wref ** 2).sum().clamp_min(1e-30)),
            "proxy_err": float(torch.einsum("oi,ij,oj->", err, self.H, err)),
            "nsamples": self.nsamples,
        }
        # Format-native diagnostics: always computed (they never change the artifact),
        # because the bottom-rung clamp fraction and its RELATIVE error are the numbers
        # that distinguish a healthy log-grid fit from the rel_mse-optimal dead one.
        try:
            out.update(clamp_diagnostics(Wref, W_hat, C, codes=codes))
        except Exception as exc:                      # zero-containing alphabets, etc.
            out["clamp_diag_error"] = repr(exc)
        if T is not None:
            out["W_tgt"] = T
        return out


@torch.no_grad()
def quantize_linear(layer, Xs, codebook, **kw):
    """Convenience: build the Hessian from an iterable of activation batches, quantize,
    and write W_hat back into `layer`. Returns the result dict."""
    g = PoTGPTQ(layer)
    for X in Xs:
        g.add_batch(X)
    out = g.quantize(codebook, **kw)
    if isinstance(layer, torch.nn.Module):
        layer.weight.data = out["W_hat"].to(layer.weight.dtype)
    g.free()
    return out
