"""Tests for the exact per-(row,group) scale re-solve (codes frozen)."""
import sys, os, torch, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pot.scale_solve import solve_group_scales, solve_diagnostics


def _case(out=6, inp=32, group=8, n=400, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(out, inp)
    levels = torch.tensor([-1., -.5, -.25, -.125, .125, .25, .5, 1.])
    codes_val = levels[torch.randint(0, len(levels), (out, inp))]
    X = torch.randn(n, inp)
    H = X.T @ X
    G = inp // group
    s0 = torch.rand(out, G) + 0.5
    return W, codes_val, H, group, s0, X


def test_mu_zero_matches_an_explicit_least_squares_on_the_samples():
    """The closed form must reproduce a direct lstsq over the actual activations."""
    W, C, H, g, s0, X = _case()
    S = solve_group_scales(W, C, H, g, mu=0.0, s0=s0).double()
    G = s0.shape[1]
    for r in range(W.shape[0]):
        # design matrix: column g is the partial sum of group g over the samples
        Z = torch.stack([(C[r].double() * m) @ X.double().T
                         for m in torch.eye(G).repeat_interleave(g, 1)], 1)   # [n, G]
        y = (W[r].double() @ X.double().T)
        ref = torch.linalg.lstsq(Z, y.unsqueeze(1)).solution.squeeze(1)
        assert torch.allclose(S[r], ref, rtol=1e-5, atol=1e-6), f"row {r}: {S[r]} vs {ref}"


def test_mu_zero_never_increases_the_output_error():
    W, C, H, g, s0, X = _case(seed=2)
    S = solve_group_scales(W, C, H, g, mu=0.0, s0=s0)
    d = solve_diagnostics(W, C, H, g, s0, S)
    assert d["out_err_rel_change"] <= 1e-9, d


def test_huge_mu_returns_the_incumbent_scales():
    """mu -> inf is a provable no-op, which is what makes the default inert."""
    W, C, H, g, s0, _X = _case(seed=3)
    S = solve_group_scales(W, C, H, g, mu=1e12, s0=s0)
    assert torch.allclose(S, s0, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("mu", [0.0, 0.1, 1.0, 10.0])
def test_error_is_monotone_in_mu(mu):
    """More ridge => closer to the incumbent => never better than the mu=0 optimum."""
    W, C, H, g, s0, _X = _case(seed=4)
    e0 = solve_diagnostics(W, C, H, g, s0,
                           solve_group_scales(W, C, H, g, mu=0.0, s0=s0))["out_err_after"]
    e = solve_diagnostics(W, C, H, g, s0,
                          solve_group_scales(W, C, H, g, mu=mu, s0=s0))["out_err_after"]
    assert e >= e0 - 1e-9


def test_codes_are_untouched_so_C1_still_holds():
    """The whole point: only scales change, so W_hat/scale is still the alphabet."""
    W, C, H, g, s0, _X = _case(seed=5)
    S = solve_group_scales(W, C, H, g, mu=1.0, s0=s0)
    Wh = C * S.repeat_interleave(g, dim=1)
    recovered = Wh / S.repeat_interleave(g, dim=1)
    assert torch.allclose(recovered, C, rtol=1e-5, atol=1e-6)


def test_row_chunking_is_exact():
    W, C, H, g, s0, _X = _case(out=20, seed=6)
    a = solve_group_scales(W, C, H, g, mu=1.0, s0=s0, row_chunk=20)
    b = solve_group_scales(W, C, H, g, mu=1.0, s0=s0, row_chunk=3)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_row_chunking_is_exact():
    """R is a pure memory knob: every block size must give bit-identical scales.

    The 2026-09-12 validation of --solve-scale lost all four arms to CUDA OOM inside
    the solve (`cH` is R*in*G float64), so the block size is now planned from free
    VRAM and halved on OOM. That is only safe if the result does not depend on it.
    """
    W, C, H, g, s0, _ = _case(out=13, inp=48, group=8, seed=7)
    ref = solve_group_scales(W, C, H, g, mu=0.7, s0=s0, row_chunk=64)
    for rc in (1, 2, 5, 13, 512):
        got = solve_group_scales(W, C, H, g, mu=0.7, s0=s0, row_chunk=rc)
        assert torch.equal(got, ref), f"row_chunk={rc} changed the answer"


def test_plan_row_chunk_shrinks_with_problem_size_and_never_returns_zero():
    from pot.scale_solve import _plan_row_chunk
    dev = torch.device("cpu")
    big = _plan_row_chunk(512, 16384, 256, dev)     # gemma-2b down_proj at g=64
    small = _plan_row_chunk(512, 2048, 32, dev)     # Qwen3-1.7B q_proj at g=64
    assert 1 <= big <= small <= 512
    assert _plan_row_chunk(512, 1 << 22, 1 << 12, dev) >= 1
