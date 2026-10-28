"""
Post-GPTQ discrete coordinate descent on the CODE ASSIGNMENT (not the scale, not the
codebook).

Why
---
GPTQ is a single greedy left-to-right pass over columns with error feedback: once
column j's code is chosen, it is never revisited, even though every later column's
error-feedback update changes the context that made that choice look locally best.
On a fine (INT4-uniform) grid the residual damage of that greediness is small. Strict
PoT is a much coarser alphabet (measured ~2.9x INT4's weight MSE), so more weights end
up stranded on the wrong side of a decision boundary and there is more to recover by
revisiting the assignment once the whole row is known.

This module does exactly one more thing after `PoTGPTQ.quantize` has run: for every
weight, try moving its code to an ADJACENT codebook level (index +-1 in the sorted
codebook) and accept the move iff it strictly reduces the row's own GPTQ proxy
objective. It never touches the codebook, never touches the scale, and never adds any
stored quantity -- it only changes which of the already-existing 16 codes each weight
holds. That means:

  * charter C1 (strict PoT alphabet) is preserved by construction: every new code is
    still drawn from the same sorted `Codebook.levels`, so `W_hat = code * scale` for a
    code in the strict-PoT alphabet, unchanged;
  * charter C3 (bit budget) costs exactly zero: nothing new is stored, so
    `check_scale_storable` / `account_bpw` are unaffected;
  * the shift-free verifier (`verify_shiftfree.py`) re-runs unchanged on the refined
    artifact, because the artifact is still literally `levels[code] * scale`.

The objective and the update rule
----------------------------------
GPTQ's own objective for one nn.Linear is the Hessian-weighted output error

    L(W_hat) = ||(W - W_hat) X||_F^2 = tr( E H E^T ),   E = W - W_hat,   H = (2/n) X^T X

which `PoTGPTQ` already accumulates as `self.H`. Rows of E are independent given H (H
does not mix rows), so the objective decomposes as a sum over rows of a per-row
quadratic form:

    L = sum_i  e_i H e_i^T,           e_i = row i of E

**Single-entry move.** Fix row i, column j. Changing that one weight's code moves
`W_hat[i, j]` by `delta_w`, which moves the row's error by the opposite amount:
`e_i[j] -> e_i[j] + d` with `d = -delta_w`. Writing e' = e + d * u_j (u_j the j-th
standard basis vector) and expanding the quadratic form:

    e'^T H e' = e^T H e + 2 d (u_j^T H e) + d^2 (u_j^T H u_j)
              = e^T H e + 2 d * (H e)_j     + d^2 * H[j,j]

so, defining the maintained residual-gradient vector `r_i = H @ e_i` (r_i[j] = (H e_i)_j
= H[j,:] . e_i, using H symmetric),

    delta_L_i = 2 * d * r_i[j] + d^2 * H[j,j]                                  (*)

This is exactly the algebra the brief proposed, confirmed by direct expansion above (the
sign convention pinned down here: `d` is the change in the ERROR e, i.e. `d =
-delta_w`, not the change in the weight itself -- get that backwards and the sign of the
linear term flips).

**Maintaining r cheaply.** After accepting a move at column j with error-change `d_i`
for row i, the FULL residual-gradient vector for that row updates by a rank-1 term:

    r_i_new = H @ (e_i + d_i * u_j) = r_i + d_i * H[:, j]

i.e. `r += d[:, None] * H[j, :][None, :]` across all rows at once (H symmetric, so
`H[:, j] == H[j, :]`). That is O(cols) work for a whole column of moves (vectorized
over every row simultaneously, which is exactly the "vectorise over rows, not columns"
structure the brief calls for -- all rows share the same H), so a full sweep over
columns costs O(cols) rank-1 updates = O(cols^2) total, matching GPTQ's own sweep cost.

**Candidate moves and acceptance.** For column j, row i, the candidates are the codebook
levels immediately adjacent to the current code's index (idx-1 and idx+1, clipped at the
alphabet's ends), scaled by that entry's ALREADY-DETERMINED effective scale
`S[i, j] = scales[i, g_idx[j]] * col_scale[j]` -- never re-derived, never re-rounded from
scratch (re-rounding on top of a GPTQ artifact is documented in `scale_learn.py` as
1.6-4.8x worse because it discards the error feedback; here we are making a strictly
local correction to an existing solution, so the scale and the "what counts as a
neighbour" structure are both taken as given). Each candidate's `delta_L` is computed by
(*); the better of the two candidates is taken, and the move is accepted only if
`delta_L < 0` (strict improvement). Declining to move (`delta_L >= 0` for both
neighbours, or at a codebook boundary) is always a legal, taken option, so the process is
a genuine coordinate descent: the row objective is non-increasing after every column,
hence over every sweep, hence over the whole pass. This also gives the anchor property:
with zero sweeps, or if no move on some layer ever improves, the output is bit-identical
to the unrefined GPTQ result.

Sweep order and count are the only knobs (`--refine-sweeps`, default 1); 1-3 full sweeps
is what the brief calls "plenty," consistent with coordinate descent converging fast once
GPTQ has already found a good column ordering and most weights are close to a local
optimum.

Invariant this file preserves
------------------------------
`W_hat[i, j] = levels[codes[i, j]] * scale[i, j]` for `codes[i, j]` drawn from the same
sorted `Codebook.levels` GPTQ used, with `scale` UNCHANGED. Nothing here can produce a
code outside that alphabet (candidates are always valid indices into the same `levels`
tensor) and nothing here stores any new per-weight or per-group quantity.
"""

from __future__ import annotations

import torch

from .codebooks import Codebook
from .verify_shiftfree import expand_scales

__all__ = ["refine_codes", "row_objective"]


def _levels(C):
    return C.levels if isinstance(C, Codebook) else C


@torch.no_grad()
def row_objective(E: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """tr(E H E^T) as a per-row vector: sum_i e_i . (H e_i). [rows] -> scalar sum if
    you want the total, but returned per-row so callers can also track it per-row."""
    R = E @ H
    return (E * R).sum(dim=1)


@torch.no_grad()
def refine_codes(W: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                 g_idx: torch.Tensor, col_scale: torch.Tensor | None,
                 codebook, H: torch.Tensor, sweeps: int = 1,
                 group: int | None = None, trace: bool = False):
    """Post-GPTQ coordinate descent on the code assignment. See module docstring.

    Args:
        W:         [rows, cols] fp32 ORIGINAL (pre-quantization) weight -- the same `W`
                   `PoTGPTQ` was built from (`g.W`).
        codes:     [rows, cols] uint8/long, GPTQ's code assignment.
        scales:    [rows, n_groups] (or already [rows, cols]) as returned by
                   `PoTGPTQ.quantize`.
        g_idx:     [cols] column -> group map, as returned by `PoTGPTQ.quantize`.
        col_scale: optional [cols] per-column scale, as returned by `PoTGPTQ.quantize`,
                   or None.
        codebook:  the SAME Codebook (or sorted level tensor) GPTQ quantized against.
        H:         [cols, cols] the Hessian GPTQ used for this linear (`g.H`, i.e. the
                   pristine one BEFORE the damping/dead-column fixups `quantize()`
                   applies to its own working copy -- damping only ridge-regularizes,
                   it does not change what layer output error means, so using the
                   undamped H here is the more honest objective; damping's effect on
                   this pass is second-order and this keeps the module needing nothing
                   from `quantize()`'s internals).
        sweeps:    number of full left-to-right passes over columns. 0 is a no-op
                   (anchor case): returns `codes` and rebuilt `W_hat` unchanged.
        trace:     if True, additionally return `obj_trace` (the objective tr(E H E^T)
                   after each sweep, length `sweeps`), `err2_trace` (tr(E E^T) =
                   ||E||_F^2 after each sweep) and `moves_trace` (accepted moves
                   per sweep). The RATIO obj/err2 is the quantity of interest: RTN's
                   error is naturally ANTI-aligned with the curvature (measured
                   tr(EHE)/tr(EE) normalised by a random orientation: 0.569 strict_pot,
                   0.836 int4_sym), and a method that only redirects the error while
                   inflating ||E|| is not improving alignment at fixed norm.
                   Costs one extra [rows,cols]x[cols,cols] matmul per sweep
                   and changes NOTHING about the returned codes -- default False keeps
                   the function bit-identical to before this argument existed.
        group:     unused directly (kept for call-site symmetry with `quantize`); the
                   group structure is already baked into `scales`/`g_idx`.

    Returns:
        dict with `codes` (refined, same dtype as input), `W_hat` (rebuilt from the
        refined codes and the UNCHANGED `scales`/`g_idx`/`col_scale`), `n_moves` (total
        accepted moves), and `obj_before`/`obj_after` (tr(E H E^T), summed over rows).
    """
    C = _levels(codebook).to(device=W.device, dtype=torch.float32)
    K = C.numel()
    rows, cols = W.shape
    codes = codes.clone().long()
    H = H.to(W.device, torch.float32)

    # normaliser for the alignment ratio: for an E with a RANDOM orientation,
    # E[tr(E H E^T) / tr(E E^T)] = tr(H)/cols, so
    #   align = (obj/err2) / h_norm  is 1.0 at random orientation, <1 anti-aligned.
    h_norm = float(torch.diag(H).mean())

    S = expand_scales(scales.to(W.device, torch.float32), cols, group, col_scale, g_idx)
    assert S.shape == (rows, cols)

    W_hat = C[codes] * S
    _E0 = W - W_hat
    obj_before = float(row_objective(_E0, H).sum())
    err2_before = float((_E0 ** 2).sum())
    del _E0

    if sweeps <= 0:
        out = {"codes": codes.to(torch.uint8) if K <= 256 else codes,
               "W_hat": W_hat, "n_moves": 0,
               "obj_before": obj_before, "obj_after": obj_before,
               "err2_before": err2_before, "err2_after": err2_before,
               "h_norm": h_norm}
        if trace:
            out["obj_trace"] = []
            out["err2_trace"] = []
            out["moves_trace"] = []
        return out

    E = W - W_hat                                    # [rows, cols], maintained
    R = E @ H                                         # [rows, cols], R[i,:] = H @ e_i

    Hdiag = torch.diag(H)                             # [cols]
    # accumulated on-device; read back once per sweep, never per column
    move_count = torch.zeros((), device=W.device, dtype=torch.long)
    n_moves = 0

    obj_trace: list[float] = []
    err2_trace: list[float] = []
    moves_trace: list[int] = []

    for _sw in range(sweeps):
        sweep_start = int(move_count)
        for j in range(cols):
            cur_idx = codes[:, j]                                       # [rows]
            cur_level = C[cur_idx]
            s = S[:, j]
            r_j = R[:, j]
            Hjj = Hdiag[j]

            idx_lo = (cur_idx - 1).clamp(0, K - 1)
            idx_hi = (cur_idx + 1).clamp(0, K - 1)
            mask_lo = cur_idx > 0
            mask_hi = cur_idx < K - 1

            d_lo = -(C[idx_lo] - cur_level) * s
            d_hi = -(C[idx_hi] - cur_level) * s

            delta_lo = 2 * d_lo * r_j + d_lo * d_lo * Hjj
            delta_hi = 2 * d_hi * r_j + d_hi * d_hi * Hjj
            delta_lo = torch.where(mask_lo, delta_lo,
                                   torch.full_like(delta_lo, float("inf")))
            delta_hi = torch.where(mask_hi, delta_hi,
                                   torch.full_like(delta_hi, float("inf")))

            hi_better = delta_hi < delta_lo
            best_delta = torch.where(hi_better, delta_hi, delta_lo)
            accept = best_delta < 0

            chosen_idx = torch.where(hi_better, idx_hi, idx_lo)
            chosen_d = torch.where(hi_better, d_hi, d_lo)

            new_idx = torch.where(accept, chosen_idx, cur_idx)
            d = torch.where(accept, chosen_d, torch.zeros_like(chosen_d))

            # NO host sync inside the column loop. Guarding these writes with
            # `if bool(accept.any())` costs a cudaDeviceSynchronize per column (measured
            # 4.2 ms) plus another for the move count (4.6 ms) against ~0.2 ms of real
            # work -- a ~12x slowdown that made a 28-layer 3-sweep run cost ~5 h.
            # Applying them unconditionally is identical: where `accept` is False,
            # `new_idx == cur_idx` and `d == 0`, so every write is a no-op.
            move_count += accept.sum()
            codes[:, j] = new_idx
            # e_i[j] += d_i  (weight moved by -d_i, error moves by +d_i)
            E[:, j] += d
            # rank-1 residual-gradient update, all rows at once (BLAS ger)
            R.addr_(d, H[j, :])

        if trace:
            obj_trace.append(float((E * (E @ H)).sum()))
            err2_trace.append(float((E ** 2).sum()))
            moves_trace.append(int(move_count) - sweep_start)

    W_hat = C[codes] * S
    n_moves = int(move_count)
    _E1 = W - W_hat
    obj_after = float(row_objective(_E1, H).sum())
    err2_after = float((_E1 ** 2).sum())
    del _E1

    out = {"codes": codes.to(torch.uint8) if K <= 256 else codes,
           "W_hat": W_hat, "n_moves": n_moves,
           "obj_before": obj_before, "obj_after": obj_after,
           "err2_before": err2_before, "err2_after": err2_after,
           "h_norm": h_norm}
    if trace:
        out["obj_trace"] = obj_trace
        out["err2_trace"] = err2_trace
        out["moves_trace"] = moves_trace
    return out
