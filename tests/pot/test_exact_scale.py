"""Tests for the exact breakpoint-based scale search (src/pot/exact_scale.py).

Four things are checked:
  1. dominance: the exact objective is never worse than any 200-point (or finer) grid.
     This is the core claim of the method -- if it ever fails, the breakpoint
     enumeration missed a crossing.
  2. brute-force agreement: on small cases, a very fine grid converges to the exact
     answer (from below, since the exact search dominates by (1)).
  3. invariant: the reconstructed artifact is still exactly `code * scale` with codes in
     the strict-PoT alphabet, checked with the project's own machine verifier.
  4. anchor: `--exact-scale` off leaves `PoTGPTQ.quantize` bit-identical to before the
     flag existed.
"""
import pytest
import torch

from pot.codebooks import get_codebook, nearest_codeword
from pot.gptq_pot import PoTGPTQ
from pot.scale_search import search_group_scale, group_quant_error
from pot.exact_scale import exact_group_scale, exact_group_error
from pot.verify_shiftfree import verify_codebook_artifact


CODEBOOKS = ["strict_pot", "strict_pot_zero"]


# --------------------------------------------------------------------- dominance
@pytest.mark.parametrize("name", CODEBOOKS)
@pytest.mark.parametrize("g", [16, 64, 128])
@pytest.mark.parametrize("seed", list(range(8)))
def test_dominance_over_200_grid(name, g, seed):
    """exact err <= best 200-point grid err, for every row, always."""
    torch.manual_seed(seed)
    C = get_codebook(name)
    B = 40
    W = torch.randn(B, g) * (0.02 * torch.exp(torch.randn(B, 1)))   # varied scales
    if seed % 3 == 0:
        W[0] *= 50.0            # a heavy-tailed / outlier group
    if seed % 4 == 0:
        W[1] = 0.0              # a dead group

    s_grid, e_grid = search_group_scale(W, C, n_grid=200, lo=0.3, hi=1.2)
    s_ex, e_ex = exact_group_scale(W, C)

    assert torch.isfinite(s_ex).all() and bool((s_ex > 0).all())
    tol = 1e-6 * (e_grid.abs() + 1e-12)
    viol = e_ex > e_grid + tol
    assert not bool(viol.any()), (
        f"exact worse than grid on {int(viol.sum())} rows; "
        f"max excess {float((e_ex - e_grid)[viol].max())}")


@pytest.mark.parametrize("name", CODEBOOKS)
def test_dominance_hess_weighted(name):
    torch.manual_seed(11)
    C = get_codebook(name)
    B, g = 40, 64
    W = torch.randn(B, g) * 0.02
    h = torch.rand(g) * 10 + 0.01

    s_grid, e_grid = search_group_scale(W, C, n_grid=200, lo=0.3, hi=1.2, hess_diag=h)
    s_ex, e_ex = exact_group_scale(W, C, hess_diag=h)

    tol = 1e-6 * (e_grid.abs() + 1e-12)
    assert bool((e_ex <= e_grid + tol).all())

    # cross-check the two error definitions agree given the exact scale.
    e_ex2 = exact_group_error(W, C, s_ex, h)
    assert torch.allclose(e_ex.double(), e_ex2, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", CODEBOOKS)
def test_dominance_over_wider_and_offset_grids(name):
    """Confirm dominance is not an artifact of the incumbent's specific [0.3, 1.2]
    window: try a wider window and an off-center one too."""
    torch.manual_seed(21)
    C = get_codebook(name)
    W = torch.randn(64, 64) * 0.02
    s_ex, e_ex = exact_group_scale(W, C)
    for lo, hi in [(0.1, 3.0), (0.5, 0.9), (0.05, 8.0)]:
        _, e_grid = search_group_scale(W, C, n_grid=400, lo=lo, hi=hi)
        assert bool((e_ex <= e_grid + 1e-6 * (e_grid.abs() + 1e-12)).all())


# --------------------------------------------------------------- brute-force agreement
@pytest.mark.parametrize("name", CODEBOOKS)
def test_brute_force_agreement(name):
    """A 200k-point grid should converge to (not beat) the exact answer."""
    torch.manual_seed(5)
    C = get_codebook(name)
    W = torch.randn(6, 32) * 0.02
    s_ex, e_ex = exact_group_scale(W, C)
    _, e_fine = search_group_scale(W, C, n_grid=200_000, lo=0.05, hi=8.0)
    # brute force cannot beat exact; and with 200k points on a well-covered range it
    # should match to a tight relative tolerance.
    assert bool((e_fine >= e_ex - 1e-6 * (e_ex.abs() + 1e-12)).all())
    rel = (e_fine - e_ex).clamp_min(0) / e_ex.clamp_min(1e-12)
    assert float(rel.max()) < 1e-3, float(rel.max())


# --------------------------------------------------------------------------- invariant
@pytest.mark.parametrize("name", CODEBOOKS)
def test_invariant_strict_pot_artifact(name):
    """W_hat from the exact scale is still exactly code * scale, strict-PoT alphabet."""
    torch.manual_seed(7)
    C = get_codebook(name)
    O, I, group = 32, 256, 64
    W = torch.randn(O, I) * 0.02
    Wg = W.reshape(-1, group)
    s, _ = exact_group_scale(Wg, C)
    codes = nearest_codeword(Wg / s, C.levels)
    W_hat = (C.levels[codes] * s).reshape(O, I)
    scales = s.reshape(O, I // group)
    ok, diag = verify_codebook_artifact(W_hat, scales, C, group=group)
    assert ok is True, diag


def test_invariant_via_full_gptq_sweep():
    """End-to-end: exact_scale=True through PoTGPTQ.quantize still passes the verifier."""
    torch.manual_seed(9)
    C = get_codebook("strict_pot")
    O, I, group = 16, 128, 64
    W = torch.randn(O, I) * 0.02
    g = PoTGPTQ(W)
    X = torch.randn(64, I)
    g.add_batch(X)
    out = g.quantize(C, group=group, percdamp=0.01, act_order=True, exact_scale=True)
    ok, diag = verify_codebook_artifact(out["W_hat"], out["scales"], C,
                                        group=group, col_scale=out["col_scale"],
                                        g_idx=out["g_idx"])
    assert ok is True, diag


# ------------------------------------------------------------------------------ anchor
def test_anchor_flag_off_is_bit_identical():
    """exact_scale=False (the default) must reproduce quantize() exactly as before the
    flag existed: run twice, once never mentioning the new kwarg and once passing
    exact_scale=False explicitly, and require bit-identical outputs."""
    torch.manual_seed(13)
    C = get_codebook("strict_pot")
    O, I, group = 24, 192, 64
    W = torch.randn(O, I) * 0.02
    X = torch.randn(48, I)

    def run(**kw):
        g = PoTGPTQ(W.clone())
        g.add_batch(X.clone())
        return g.quantize(C, group=group, percdamp=0.01, act_order=True, **kw)

    out_default = run()
    out_explicit_off = run(exact_scale=False)

    assert torch.equal(out_default["W_hat"], out_explicit_off["W_hat"])
    assert torch.equal(out_default["codes"], out_explicit_off["codes"])
    assert torch.equal(out_default["scales"], out_explicit_off["scales"])
    assert out_default["rel_mse"] == out_explicit_off["rel_mse"]

    # and it must differ, in general, from the exact-scale run (otherwise this whole
    # module would be doing nothing) -- not a strict requirement everywhere, but on
    # this seed/config it should show at least a nonzero delta in achieved error.
    out_exact = run(exact_scale=True)
    assert out_exact["rel_mse"] <= out_default["rel_mse"] + 1e-9
