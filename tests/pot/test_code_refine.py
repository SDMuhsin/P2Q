"""Tests for the post-GPTQ discrete code-assignment refinement (src/pot/code_refine.py).

Covers the four properties the design promises:
  * invariant: refined weights are still exactly code * scale in the strict-PoT alphabet
  * monotonicity: tr(E H E^T) never increases across a sweep
  * anchor: sweeps=0 (or nothing improving) reproduces plain GPTQ bit-for-bit
  * correctness: the incremental delta formula agrees with brute-force recomputation
"""
import torch

from pot.codebooks import get_codebook
from pot.gptq_pot import PoTGPTQ
from pot.code_refine import refine_codes, row_objective
from pot.verify_shiftfree import verify_strict_pot


def _make_hessian(cols, device, seed=0):
    """A PD-ish Hessian from random correlated activations, same shape GPTQ builds."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(64, cols, generator=g)
    # give it some cross-column correlation so the feedback term is nonzero
    mix = torch.randn(cols, cols, generator=g) * 0.3
    X = X @ (torch.eye(cols) + mix)
    H = (2.0 / X.shape[0]) * X.t() @ X
    return H


def _fit(W, C, group=64, sweeps=1, **kw):
    rows, cols = W.shape
    g = PoTGPTQ(W)
    H = _make_hessian(cols, W.device, seed=kw.pop("seed", 0))
    g.H = H
    g.nsamples = 1
    out = g.quantize(C, group=group, percdamp=0.01, **kw)
    rinfo = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                         C, g.H, sweeps=sweeps, group=group)
    return out, rinfo, H


def test_invariant_strict_pot_alphabet():
    torch.manual_seed(0)
    C = get_codebook("strict_pot")
    W = torch.randn(32, 256) * 0.05
    out, rinfo, H = _fit(W, C, group=64, sweeps=2)
    ok, diag = verify_strict_pot(rinfo["W_hat"], out["scales"], group=out["group"],
                                 col_scale=out["col_scale"], g_idx=out["g_idx"])
    assert ok, diag["examples"]
    # also sanity-check codes are within the alphabet bounds
    assert rinfo["codes"].max() < len(C)
    assert rinfo["codes"].min() >= 0


def test_invariant_strict_pot_zero_alphabet():
    torch.manual_seed(1)
    C = get_codebook("strict_pot_zero")
    W = torch.randn(24, 192) * 0.03
    out, rinfo, H = _fit(W, C, group=64, sweeps=3)
    ok, diag = verify_strict_pot(rinfo["W_hat"], out["scales"], group=out["group"],
                                 col_scale=out["col_scale"], g_idx=out["g_idx"])
    assert ok, diag["examples"]


def test_monotonicity_across_sweep():
    """tr(E H E^T) must never increase from one sweep boundary to the next."""
    torch.manual_seed(2)
    C = get_codebook("strict_pot")
    W = torch.randn(16, 128) * 0.08
    g = PoTGPTQ(W)
    H = _make_hessian(128, W.device, seed=2)
    g.H = H
    g.nsamples = 1
    out = g.quantize(C, group=64, percdamp=0.01)

    objs = []
    codes, scales, g_idx, col_scale = out["codes"], out["scales"], out["g_idx"], out["col_scale"]
    for n in range(0, 5):
        rinfo = refine_codes(g.W, codes, scales, g_idx, col_scale, C, H, sweeps=n)
        objs.append(rinfo["obj_after"])
    for a, b in zip(objs, objs[1:]):
        assert b <= a + 1e-6, (objs)


def test_anchor_zero_sweeps_is_identical():
    torch.manual_seed(3)
    C = get_codebook("strict_pot")
    W = torch.randn(20, 128) * 0.04
    out, rinfo, H = _fit(W, C, group=64, sweeps=0)
    assert torch.equal(rinfo["W_hat"], out["W_hat"])
    assert torch.equal(rinfo["codes"].to(out["codes"].dtype), out["codes"])
    assert rinfo["n_moves"] == 0
    assert rinfo["obj_before"] == rinfo["obj_after"]


def test_anchor_no_improving_move_is_identical():
    """With H = 0 (no output-error signal at all), every candidate delta is exactly 0,
    so nothing is strictly improving (delta < 0 is required) and the pass must be a
    no-op even with many sweeps requested."""
    torch.manual_seed(4)
    C = get_codebook("strict_pot")
    W = torch.randn(12, 64) * 0.05
    g = PoTGPTQ(W)
    cols = W.shape[1]
    g.H = torch.zeros(cols, cols)
    g.nsamples = 1
    out = g.quantize(C, group=32, percdamp=0.01)
    rinfo = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                         C, g.H, sweeps=3, group=32)
    assert rinfo["n_moves"] == 0
    assert torch.equal(rinfo["W_hat"], out["W_hat"])


def test_delta_formula_matches_bruteforce():
    """The incremental delta_L = 2*d*r_j + d^2*H[j,j] must agree with directly
    recomputing tr(E H E^T) before and after a single hypothetical flip."""
    torch.manual_seed(5)
    C = get_codebook("strict_pot")
    W = torch.randn(6, 32) * 0.05
    H = _make_hessian(32, W.device, seed=5)
    g = PoTGPTQ(W)
    g.H = H
    g.nsamples = 1
    out = g.quantize(C, group=32, percdamp=0.01)

    codes = out["codes"].long()
    scales = out["scales"]
    from pot.verify_shiftfree import expand_scales
    S = expand_scales(scales, W.shape[1], out["group"], out["col_scale"], out["g_idx"])
    W_hat = C.levels[codes] * S
    E = W - W_hat
    obj0 = row_objective(E, H).sum()
    R = E @ H

    K = len(C)
    for j in [0, 5, 17, 31]:
        for offset in (-1, 1):
            cur_idx = codes[:, j]
            cand_idx = (cur_idx + offset).clamp(0, K - 1)
            valid = cand_idx != cur_idx
            if not bool(valid.any()):
                continue
            cand_level = C.levels[cand_idx]
            cur_level = C.levels[cur_idx]
            s = S[:, j]
            d = -(cand_level - cur_level) * s
            r_j = R[:, j]
            Hjj = H[j, j]
            delta_incr = 2 * d * r_j + d * d * Hjj

            # brute force: apply the flip to row i alone, recompute tr(E H E^T) fully
            for i in range(W.shape[0]):
                if not bool(valid[i]):
                    continue
                codes2 = codes.clone()
                codes2[i, j] = cand_idx[i]
                W_hat2 = C.levels[codes2] * S
                E2 = W - W_hat2
                obj2 = row_objective(E2, H).sum()
                delta_true = float(obj2 - obj0)
                assert abs(delta_true - float(delta_incr[i])) < 1e-3, (
                    j, offset, i, delta_true, float(delta_incr[i]))


def test_relative_mse_improvement_reported():
    """Not a pass/fail assertion -- measures and prints the CPU rel-MSE improvement on a
    small synthetic case so the PI can decide whether to spend GPU time on this."""
    torch.manual_seed(6)
    C = get_codebook("strict_pot")
    W = torch.randn(128, 512) * 0.03
    g = PoTGPTQ(W)
    H = _make_hessian(512, W.device, seed=6)
    g.H = H
    g.nsamples = 1
    out = g.quantize(C, group=64, percdamp=0.01, act_order=True)
    rinfo = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                         C, H, sweeps=2, group=64)
    rel_before = out["rel_mse"]
    err = rinfo["W_hat"] - W
    rel_after = float((err ** 2).sum() / (W ** 2).sum())
    print(f"\n[code_refine] proxy tr(EHE^T) before={rinfo['obj_before']:.6g} "
          f"after={rinfo['obj_after']:.6g}  "
          f"proxy improvement="
          f"{(1 - rinfo['obj_after'] / rinfo['obj_before']) * 100:.3f}%")
    print(f"[code_refine] plain weight rel_mse before={rel_before:.6g} "
          f"after={rel_after:.6g}  moves={rinfo['n_moves']}")
    # the pass is a coordinate descent on the PROXY (Hessian-weighted) objective, not
    # on plain weight MSE, so only the proxy is guaranteed non-increasing.
    assert rinfo["obj_after"] <= rinfo["obj_before"] + 1e-6
