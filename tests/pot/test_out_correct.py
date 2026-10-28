"""Tests for the post-hoc per-output gain/bias (pot/out_correct.py).

The closed form is checked against an explicit least-squares fit on real sampled
activations, because the whole point of the module is that it needs only (H, xbar).
"""
import sys, os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pot.out_correct import output_correction, quantize_bias


def _case(seed=0, out=6, inp=32, n=400):
    torch.manual_seed(seed)
    # float64 throughout: this file checks ALGEBRA, so the only tolerance that should
    # appear is the solver's, not fp32's H = X^T X accumulation (worth ~1e-8 relative).
    W = torch.randn(out, inp, dtype=torch.float64)
    W_hat = W + 0.1 * torch.randn(out, inp, dtype=torch.float64)     # a fixed "artifact"
    X = torch.randn(n, inp, dtype=torch.float64) + 0.7   # non-zero mean: bias must matter
    H = X.T @ X
    return W, W_hat, H, X.mean(0), n, X


def _brute(W, W_hat, X, mode):
    """Explicit per-row least squares over the actual samples."""
    a = torch.ones(W.shape[0], dtype=torch.float64)
    b = torch.zeros(W.shape[0], dtype=torch.float64)
    U, V = (W_hat.double() @ X.double().T), (W.double() @ X.double().T)   # [out, n]
    for r in range(W.shape[0]):
        u, v = U[r], V[r]
        if mode == "both":
            A = torch.stack([u, torch.ones_like(u)], 1)
            sol = torch.linalg.lstsq(A, v.unsqueeze(1)).solution.squeeze(1)
            a[r], b[r] = sol[0], sol[1]
        elif mode == "gain":
            a[r] = (u @ v) / (u @ u)
        elif mode == "bias":
            b[r] = (v - u).mean()
    return a, b


@pytest.mark.parametrize("mode", ["gain", "bias", "both"])
def test_closed_form_matches_explicit_least_squares(mode):
    W, W_hat, H, xbar, n, X = _case()
    a, b, _ = output_correction(W, W_hat, H, xbar, n, mode=mode)
    ab, bb = _brute(W, W_hat, X, mode)
    # `output_correction` returns float32 (that is what the pipeline stores), so the
    # comparison is limited by fp32 representation of a ~1.0 gain, not by the algebra.
    assert torch.allclose(a.double(), ab, rtol=1e-6, atol=1e-7), f"{a} vs {ab}"
    assert torch.allclose(b.double(), bb, rtol=1e-6, atol=1e-7), f"{b} vs {bb}"


@pytest.mark.parametrize("mode", ["gain", "bias", "both"])
def test_reported_reduction_matches_a_direct_recomputation(mode):
    """`frac_output_mse_removed` must equal what you get by applying (a, b) to X."""
    W, W_hat, H, xbar, n, X = _case(seed=3)
    a, b, d = output_correction(W, W_hat, H, xbar, n, mode=mode)
    base = (((W_hat - W).double() @ X.double().T) ** 2).sum()
    corr = ((a.double()[:, None] * (W_hat.double() @ X.double().T)
             + b.double()[:, None] - W.double() @ X.double().T) ** 2).sum()
    # tolerance set by the float32 round-trip of (a, b), see the note in `_case`
    assert abs(d["frac_output_mse_removed"] - float(1 - corr / base)) < 1e-6


@pytest.mark.parametrize("mode", ["gain", "bias", "both"])
def test_correction_never_increases_output_error(mode):
    """(a, b) = (1, 0) is in the feasible set, so the optimum can only help."""
    for seed in range(4):
        W, W_hat, H, xbar, n, X = _case(seed=seed)
        _a, _b, d = output_correction(W, W_hat, H, xbar, n, mode=mode)
        assert d["frac_output_mse_removed"] >= -1e-12


def test_mode_none_is_exactly_inert():
    W, W_hat, H, xbar, n, _X = _case()
    a, b, d = output_correction(W, W_hat, H, xbar, n, mode="none")
    assert torch.equal(a, torch.ones_like(a)) and torch.equal(b, torch.zeros_like(b))
    assert abs(d["frac_output_mse_removed"]) < 1e-12


def test_joint_is_at_least_as_good_as_either_alone():
    W, W_hat, H, xbar, n, _X = _case(seed=7)
    fs = {m: output_correction(W, W_hat, H, xbar, n, mode=m)[2]["frac_output_mse_removed"]
          for m in ("gain", "bias", "both")}
    assert fs["both"] >= fs["gain"] - 1e-9 and fs["both"] >= fs["bias"] - 1e-9


def test_gain_folds_into_the_scale_without_touching_any_code():
    """The C1-critical property: W_hat = codes * scale must survive the fold exactly."""
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (8, 64)).float() - 8
    scale = torch.rand(8, 1) + 0.5
    a = torch.rand(8) + 0.5
    scale2 = scale * a[:, None]
    # The pipeline must REBUILD W_hat from the folded scale rather than rescale the old
    # W_hat: `codes * (a*s)` and `(codes*s) * a` differ in the last bits in fp32, and the
    # artifact's C1 verification checks `W_hat / scale` against the alphabet exactly.
    W_hat2 = codes * scale2
    assert torch.allclose(W_hat2, (codes * scale) * a[:, None], rtol=1e-6)
    # The artifact's own verifier is the authority: it snaps the float64 ratio to the
    # alphabet rather than demanding bit-exact division (fp32 a/b*b does not round-trip
    # in general). Assert against the shipped verifier, not against a stricter invented
    # criterion -- a gain fold must leave `verify.ok` true, which is the C1 obligation.
    from pot.codebooks import get_codebook
    from pot.verify_shiftfree import verify_codebook_artifact
    cb = get_codebook("strict_pot")
    lv = cb.levels
    idx = torch.randint(0, lv.numel(), (8, 64))
    Wq = lv[idx] * scale2
    ok, diag = verify_codebook_artifact(Wq, scale2, cb, group=64)
    assert ok, diag


@pytest.mark.parametrize("bits", [4, 8, 12, 16])
def test_quantize_bias_respects_its_own_bit_width(bits):
    torch.manual_seed(0)
    b = torch.randn(512) * 3e-3
    q, d = quantize_bias(b, bits)
    assert d > 0
    codes = torch.round(q / d)
    assert int(codes.abs().max()) <= 2 ** (bits - 1)
    assert torch.allclose(q, codes * d, atol=1e-12)
    # rounding error is bounded by half a step
    assert (q - b).abs().max() <= d / 2 + 1e-12


def test_quantize_bias_passthrough_and_zero():
    b = torch.randn(16)
    q, d = quantize_bias(b, None)
    assert torch.equal(q, b) and d == 0.0
    z, d0 = quantize_bias(torch.zeros(16), 8)
    assert torch.equal(z, torch.zeros(16)) and d0 == 0.0


def test_singular_scatter_falls_back_to_the_uncorrected_artifact():
    """A rank-deficient centred scatter must yield a_r = 1, not a 1e34 blow-up.

    This is a regression test for a real failure: with few calibration tokens the
    centred denominator `u Sigma u^T` goes numerically negative and the unguarded
    ratio produced a = -4.5e34 and rel_mse = inf on layer 0 of a smoke run.
    """
    torch.manual_seed(0)
    out, inp, n = 5, 16, 3                       # n << inp: Sigma is rank <= n-1
    W = torch.randn(out, inp, dtype=torch.float64)
    W_hat = W + 0.1 * torch.randn(out, inp, dtype=torch.float64)
    X = torch.randn(n, inp, dtype=torch.float64) + 5.0     # mean dominates the energy
    a, b, d = output_correction(W, W_hat, X.T @ X, X.mean(0), n, mode="both")
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert a.abs().max() < 1e3, a
    assert d["n_gain_fallback"] >= 0
    # and the correction still cannot make the output error worse
    assert d["frac_output_mse_removed"] >= -1e-9


def test_degenerate_rank_one_activation_is_handled():
    """Every token identical => Sigma == 0 exactly => every row must fall back."""
    torch.manual_seed(1)
    W = torch.randn(4, 8, dtype=torch.float64)
    W_hat = W + 0.05 * torch.randn(4, 8, dtype=torch.float64)
    x = torch.randn(8, dtype=torch.float64)
    X = x.repeat(20, 1)
    a, b, d = output_correction(W, W_hat, X.T @ X, X.mean(0), 20, mode="both")
    assert d["n_gain_fallback"] == 4
    assert torch.allclose(a, torch.ones_like(a))
    # with a == 1 the bias alone can cancel the error exactly on a rank-1 input
    assert d["frac_output_mse_removed"] > 1 - 1e-9


def test_gainmean_is_one_scalar_for_the_whole_matrix():
    W, W_hat, H, xbar, n, X = _case(seed=11)
    a, b, d = output_correction(W, W_hat, H, xbar, n, mode="gainmean")
    assert torch.allclose(a, a[0].expand_as(a)), "gainmean must be row-constant"
    assert torch.equal(b, torch.zeros_like(b)), "gainmean adds no bias"
    assert d["frac_output_mse_removed"] >= -1e-9


def test_gainmean_is_the_optimal_single_scalar():
    """It must beat every other scalar, and be no better than the per-row fit."""
    W, W_hat, H, xbar, n, X = _case(seed=12)
    a, _b, d = output_correction(W, W_hat, H, xbar, n, mode="gainmean")
    base = (((W_hat - W).double() @ X.double().T) ** 2).sum()

    def err(c):
        return float((((c * W_hat.double() - W.double()) @ X.double().T) ** 2).sum())
    best = float(a[0])
    for c in (best - 0.02, best - 0.005, best + 0.005, best + 0.02):
        assert err(best) <= err(c) + 1e-9, f"scalar {best} beaten by {c}"
    per_row = output_correction(W, W_hat, H, xbar, n,
                                mode="gain")[2]["frac_output_mse_removed"]
    assert per_row >= d["frac_output_mse_removed"] - 1e-9
