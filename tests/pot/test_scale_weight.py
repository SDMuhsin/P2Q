"""Per-column weights for the group scale search (`hess_weighted_scale=` / --scale-weight).

The premise: the scale search minimizes `sum_j h_j (q_j - w_j)^2` over one group, but the
quantity GPTQ minimizes is `tr(E H E^T)`, which under the sweep's own LDL identity
decomposes exactly as `sum_j ||w_j - q_j||^2 / Hinv[j,j]^2`. So the additive per-column
weight GPTQ's objective implies is `1/Hinv[j,j]^2` (the CONDITIONAL precision of column j
given its successors), not `diag(H)` (the MARGINAL precision), which is what the historical
`--hess-weighted-scale` flag uses. `'ldl'` adds that weight as a third option; it costs
exactly what `'diag'` costs, since both are a [cols] vector handed to the same search.

`'invdiag'` (= `1/diag(H)`) is a fourth value that is DELIBERATELY WRONG: it is the
reciprocal of `'diag'`, so the scale search spends its accuracy on the columns the output
error cares about least. It exists only as the falsification control for the "monotone"
hypothesis and must never reach a deliverable; the tests below pin that it really does
make the objective worse, which is the only property a control needs.

The load-bearing test here is `test_default_path_is_bit_for_bit_unchanged`, and its twin
`test_bool_true_still_means_diag`: this pipeline is bit-deterministic and every run in
results/pot/pot_runs.jsonl stored a BOOLEAN for this argument, so False->'none' and
True->'diag' must keep reproducing those runs verbatim.
"""
import pytest
import torch

from pot.codebooks import get_codebook
from pot.gptq_pot import PoTGPTQ
from pot.verify_shiftfree import verify_codebook_artifact

WEIGHTS = ["none", "diag", "ldl"]
ALL_WEIGHTS = WEIGHTS + ["invdiag"]


def _fit(I=256, O=48, seed=0):
    """Correlated, heavy-tailed columns, so diag(H) and 1/Hinv[j,j]^2 really differ."""
    torch.manual_seed(seed)
    W = torch.randn(O, I) * 0.02
    W = W * torch.exp(torch.randn(1, I) * 1.2)
    A = torch.randn(4 * I, I)
    A = A @ (torch.eye(I) + 0.4 * torch.randn(I, I) / I ** 0.5)   # correlate the columns
    A = A * torch.exp(torch.randn(1, I) * 1.0)
    g = PoTGPTQ(W)
    g.add_batch(A)
    return W, g


def _cb():
    return get_codebook("strict_pot")


# ------------------------------------------------------- the anchor: nothing moved
@pytest.mark.parametrize("act_order", [False, True])
def test_default_path_is_bit_for_bit_unchanged(act_order):
    """Not passing the argument at all == passing False == passing 'none', exactly."""
    _, g = _fit()
    C = _cb()
    base = g.quantize(C, group=64, act_order=act_order)
    for spelling in (False, "none"):
        out = g.quantize(C, group=64, act_order=act_order,
                         hess_weighted_scale=spelling)
        assert torch.equal(base["W_hat"], out["W_hat"])
        assert torch.equal(base["codes"], out["codes"])
        assert torch.equal(base["scales"], out["scales"])
        assert base["gptq_loss"] == out["gptq_loss"]


@pytest.mark.parametrize("act_order", [False, True])
def test_bool_true_still_means_diag(act_order):
    """Every recorded run passed a bool; True must keep meaning diag(H) forever."""
    _, g = _fit()
    C = _cb()
    a = g.quantize(C, group=64, act_order=act_order, hess_weighted_scale=True)
    b = g.quantize(C, group=64, act_order=act_order, hess_weighted_scale="diag")
    assert torch.equal(a["W_hat"], b["W_hat"])
    assert a["scale_weight"] == b["scale_weight"] == "diag"


def test_none_diag_ldl_are_three_distinct_solutions():
    _, g = _fit()
    C = _cb()
    outs = {m: g.quantize(C, group=64, hess_weighted_scale=m) for m in WEIGHTS}
    for m in WEIGHTS:
        assert outs[m]["scale_weight"] == m
    for a, b in [("none", "diag"), ("none", "ldl"), ("diag", "ldl")]:
        assert not torch.equal(outs[a]["scales"], outs[b]["scales"]), (a, b)


def test_bad_value_rejected():
    _, g = _fit()
    with pytest.raises(ValueError):
        g.quantize(_cb(), group=64, hess_weighted_scale="hessian")


# ------------------------------------------------- the weight is the one claimed
def test_ldl_weight_is_one_over_hinv_diag_squared():
    """Reconstruct Hinv the way quantize() does and check _scale_weights returns
    1/diag(Hinv)^2 -- i.e. the claim in the docstring is the code, not a comment."""
    _, g = _fit()
    cols = g.columns
    H = g.H.clone()
    H[range(cols), range(cols)] += 0.01 * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    h_ldl = PoTGPTQ._scale_weights("ldl", H, Hinv)
    torch.testing.assert_close(h_ldl, 1.0 / torch.diagonal(Hinv) ** 2)

    h_diag = PoTGPTQ._scale_weights("diag", H, Hinv)
    torch.testing.assert_close(h_diag, torch.diag(H))
    assert PoTGPTQ._scale_weights("none", H, Hinv) is None

    # the two Hessian weights are genuinely different objects, not a rescaling of
    # each other (a pure rescale would leave every argmin of the search unchanged)
    r = h_ldl / h_diag
    assert float(r.max() / r.min()) > 1.05


def test_ldl_weight_matches_the_lower_triangular_identity():
    """1/Hinv[j,j]^2 is the conditional precision of column j given its SUCCESSORS.
    Check it directly against a Schur complement on the trailing block."""
    _, g = _fit(I=64, O=8)
    cols = g.columns
    H = g.H.clone()
    H[range(cols), range(cols)] += 0.01 * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    j = 17
    A = H[j, j]
    B = H[j, j + 1:]
    D = H[j + 1:, j + 1:]
    schur = A - B @ torch.linalg.solve(D, B)          # conditional precision of col j
    torch.testing.assert_close(1.0 / Hinv[j, j] ** 2, schur, rtol=1e-3, atol=0)


# ----------------------------------------------------- it actually helps the objective
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_hessian_weights_beat_uniform_on_gptq_own_objective(seed):
    """tr(E H E^T) is what GPTQ minimizes, so weighting the scale search by ANY sane
    Hessian weight should beat uniform weights by a wide margin (~40-70 % on this
    fixture)."""
    W, g = _fit(seed=seed)
    C = _cb()
    p = {m: g.quantize(C, group=64, hess_weighted_scale=m)["proxy_err"] for m in WEIGHTS}
    assert p["ldl"] < p["none"]
    assert p["diag"] < p["none"]


def test_ldl_vs_diag_is_not_resolvable_on_a_synthetic_fixture():
    """Deliberately asserts the WEAK thing, because the strong thing is not true here.

    On real Qwen3-1.7B layers 'ldl' beats 'diag' by 10.7 % vs 9.0 % of tr(E H E^T)
    (25/25 layers, blocks 0/7/14/20/27 x q/v/o/gate/down, g=64, strict_pot8 -- see
    FINDINGS_C). That separation comes from real inter-column correlation making the
    CONDITIONAL precision 1/Hinv[j,j]^2 differ substantially from the MARGINAL precision
    diag(H). On the synthetic fixture here the two agree to within +-0.2 % and the sign
    flips with the seed, so asserting `ldl < diag` would be asserting noise. What is
    testable in-repo is that the two are distinct, sane, and that the 'ldl' weight is
    literally the conditional precision (test_ldl_weight_matches_the_lower_triangular_identity).
    """
    ratios = []
    for seed in range(6):
        _, g = _fit(seed=seed)
        C = _cb()
        d = g.quantize(C, group=64, hess_weighted_scale="diag")["proxy_err"]
        l = g.quantize(C, group=64, hess_weighted_scale="ldl")["proxy_err"]
        ratios.append(l / d)
    assert max(ratios) < 1.01 and min(ratios) > 0.99, ratios


# ------------------------------------------------------------ constraints still hold
@pytest.mark.parametrize("mode", WEIGHTS)
def test_artifact_is_still_shift_free(mode):
    _, g = _fit()
    C = _cb()
    out = g.quantize(C, group=64, act_order=True, hess_weighted_scale=mode)
    # the artifact must still be exactly levels[code] * scale ...
    S = out["scales"][:, out["g_idx"]]
    assert torch.equal(C.levels[out["codes"].long()] * S, out["W_hat"])
    # ... and every stored weight a single signed power of two (charter C1)
    ok, rep = verify_codebook_artifact(out["W_hat"], out["scales"], C,
                                       group=out["group"], col_scale=out["col_scale"],
                                       g_idx=out["g_idx"])
    assert ok, rep


@pytest.mark.parametrize("mode", WEIGHTS)
def test_coexists_with_order_option(mode):
    """Agent D's `order=` and this option are independent; every combination runs and
    stays shift-free."""
    _, g = _fit()
    C = _cb()
    for order in PoTGPTQ.ORDERS:
        out = g.quantize(C, group=64, order=order, hess_weighted_scale=mode)
        assert out["order"] == order and out["scale_weight"] == mode
        S = out["scales"][:, out["g_idx"]]
        assert torch.equal(C.levels[out["codes"].long()] * S, out["W_hat"])


# --------------------------------------------------- the deliberately-wrong control
def test_invdiag_weight_is_the_reciprocal_of_diag():
    _, g = _fit()
    cols = g.columns
    H = g.H.clone()
    H[range(cols), range(cols)] += 0.01 * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    h_inv = PoTGPTQ._scale_weights("invdiag", H, Hinv)
    torch.testing.assert_close(h_inv, 1.0 / torch.diag(H))
    torch.testing.assert_close(h_inv * PoTGPTQ._scale_weights("diag", H, Hinv),
                               torch.ones(cols))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_invdiag_really_is_worse_a_control_must_lose(seed):
    """The whole point of the control is that it degrades the objective. If this ever
    starts passing by accident the control is not a control -- measured on 25 real
    Qwen3-1.7B layers it is +13.9 % on tr(E H E^T), worse on 25 of 25."""
    _, g = _fit(seed=seed)
    C = _cb()
    p = {m: g.quantize(C, group=64, hess_weighted_scale=m)["proxy_err"]
         for m in ("none", "diag", "ldl", "invdiag")}
    assert p["invdiag"] > p["diag"]
    assert p["invdiag"] > p["ldl"]


def test_invdiag_is_not_the_default_and_must_be_asked_for():
    _, g = _fit()
    C = _cb()
    base = g.quantize(C, group=64)
    assert base["scale_weight"] == "none"
    out = g.quantize(C, group=64, hess_weighted_scale="invdiag")
    assert out["scale_weight"] == "invdiag"
    assert not torch.equal(base["scales"], out["scales"])


@pytest.mark.parametrize("mode", ["invdiag"])
def test_control_artifact_is_still_shift_free(mode):
    """Even the deliberately-bad weight must not be able to break charter C1."""
    _, g = _fit()
    C = _cb()
    out = g.quantize(C, group=64, act_order=True, hess_weighted_scale=mode)
    ok, rep = verify_codebook_artifact(out["W_hat"], out["scales"], C,
                                       group=out["group"], col_scale=out["col_scale"],
                                       g_idx=out["g_idx"])
    assert ok, rep


# ------------------------------------------- the powdiag family (diagnostic only)
def test_powdiag_parses_and_rejects():
    assert PoTGPTQ._parse_powdiag("powdiag:-0.25") == -0.25
    assert PoTGPTQ._parse_powdiag("powdiag:1") == 1.0
    assert PoTGPTQ._parse_powdiag("diag") is None
    assert PoTGPTQ._parse_powdiag(True) is None
    with pytest.raises(ValueError):
        PoTGPTQ._parse_powdiag("powdiag:banana")


def test_powdiag_endpoints_reproduce_the_named_weights():
    """alpha=+1 is 'diag', alpha=-1 is 'invdiag', alpha=0 is uniform. The family must
    contain the named values exactly, or the x-axis it sweeps is not the same axis."""
    _, g = _fit()
    cols = g.columns
    H = g.H.clone()
    H[range(cols), range(cols)] += 0.01 * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    torch.testing.assert_close(PoTGPTQ._scale_weights("powdiag:1.0", H, Hinv),
                               PoTGPTQ._scale_weights("diag", H, Hinv))
    torch.testing.assert_close(PoTGPTQ._scale_weights("powdiag:-1.0", H, Hinv),
                               PoTGPTQ._scale_weights("invdiag", H, Hinv))
    torch.testing.assert_close(PoTGPTQ._scale_weights("powdiag:0.0", H, Hinv),
                               torch.ones(cols))


def test_powdiag_alpha_zero_is_the_default_solution():
    """Uniform weights reached through the family must give the same artifact as not
    weighting at all -- otherwise alpha=0 is not the control's x=0."""
    _, g = _fit()
    C = _cb()
    base = g.quantize(C, group=64)
    zero = g.quantize(C, group=64, hess_weighted_scale="powdiag:0.0")
    assert torch.equal(base["W_hat"], zero["W_hat"])
    assert torch.equal(base["codes"], zero["codes"])


@pytest.mark.parametrize("alpha", [-0.2, -0.5])
def test_powdiag_degradation_points_are_between_control_and_invdiag(alpha):
    """The two points used to map the degradation side must actually sit between the
    control and the full 'invdiag' control on GPTQ's own objective. Measured on 25 real
    Qwen3-1.7B layers: alpha=-0.2 -> +3.94 %, alpha=-0.5 -> +8.44 %, vs +13.93 % for
    'invdiag'."""
    _, g = _fit()
    C = _cb()
    p = {m: g.quantize(C, group=64, hess_weighted_scale=m)["proxy_err"]
         for m in ("none", "invdiag", f"powdiag:{alpha}")}
    assert p["none"] < p[f"powdiag:{alpha}"] < p["invdiag"]


@pytest.mark.parametrize("alpha", [-0.2, -0.5])
def test_powdiag_artifact_is_still_shift_free(alpha):
    _, g = _fit()
    C = _cb()
    out = g.quantize(C, group=64, act_order=True,
                     hess_weighted_scale=f"powdiag:{alpha}")
    assert out["scale_weight"] == f"powdiag:{alpha}"
    ok, rep = verify_codebook_artifact(out["W_hat"], out["scales"], C,
                                       group=out["group"], col_scale=out["col_scale"],
                                       g_idx=out["g_idx"])
    assert ok, rep
