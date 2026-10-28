"""Column PERMUTATION as a partition device (section 21.4(A)).

The premise tested here is different from test_gptq_order.py's. That file is about the
GPTQ processing ORDER. This one is about the fact -- read out of `quantize` and pinned
here -- that the permutation also decides GROUP MEMBERSHIP, because groups are contiguous
slices of the permuted column order. Under `--method rtn` (H := I) that is the ONLY thing
it decides, which has a sharp consequence: reversing the permutation is a null change.
"""
import pytest
import torch

from pot.codebooks import get_codebook
from pot.gptq_pot import PoTGPTQ
from pot.verify_shiftfree import expand_scales, verify_codebook_artifact

NEW = ["logmean", "logmeanasc", "salogmean", "rand"]
OLD = ["none", "act", "err", "errasc", "errmag"]


def _fit(I=256, O=48, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(O, I) * 0.02
    W = W * torch.exp(torch.randn(1, I) * 1.2)          # heavy-tailed column locations
    A = torch.randn(4 * I, I) * torch.exp(torch.randn(1, I) * 1.0)
    g = PoTGPTQ(W)
    g.add_batch(A)
    return W, g


def _rtn(g):
    """Exactly what run_pot.py does for --method rtn, and the calibration diag it keeps."""
    h = torch.diag(g.H).clone()
    g.H = torch.eye(g.columns)
    g.nsamples = 1
    return h


def _q(g, order, h=None, **kw):
    C = get_codebook("strict_pot")
    return g.quantize(C, group=64, percdamp=0.0, order=order, exact_scale=True,
                      perm_hess=h, **kw)


# ------------------------------------------------- the existing orders are untouched
@pytest.mark.parametrize("order", OLD)
def test_existing_orders_unchanged(order):
    """Every pre-existing order still selects exactly the permutation its own formula
    specifies. Adding branches to _column_perm must not perturb them."""
    C = get_codebook("strict_pot").levels
    W, g = _fit()
    Hd = torch.diag(g.H).clone()
    sk = dict(pot_scale=False, exact_scale=True)
    perm = g._column_perm(order, g.W, C, Hd, 64, sk, None, None)
    if order == "none":
        assert perm is None
        return
    if order == "act":
        want = torch.argsort(Hd, descending=True)
    elif order == "errmag":
        want = torch.argsort(Hd * (g.W ** 2).sum(0), descending=True)
    else:
        e = g._rtn_col_err(g.W, C, 64, sk, None, None)
        want = torch.argsort(Hd * e, descending=(order != "errasc"))
    assert torch.equal(perm, want)


def test_default_is_still_bit_for_bit_identity():
    _, g = _fit()
    a, b = _q(g, None), _q(g, "none")
    assert torch.equal(a["W_hat"], b["W_hat"]) and torch.equal(a["g_idx"], b["g_idx"])


# ---------------------------------------------------------------- the new orders
@pytest.mark.parametrize("order", NEW)
def test_is_a_true_permutation(order):
    C = get_codebook("strict_pot").levels
    W, g = _fit()
    h = _rtn(g)
    Hd = torch.diag(g.H).clone()
    perm = g._column_perm(order, g.W, C, Hd, 64, dict(pot_scale=False, exact_scale=True),
                          None, None, perm_hess=h)
    assert perm.shape == (g.columns,)
    assert torch.equal(torch.sort(perm).values, torch.arange(g.columns))


@pytest.mark.parametrize("order", NEW)
def test_g_idx_round_trips_and_partitions_evenly(order):
    _, g = _fit()
    h = _rtn(g)
    out = _q(g, order, h=h)
    gi = out["g_idx"]
    # every column is assigned a group, and the groups are exactly equal-sized
    assert torch.equal(torch.sort(gi).values,
                       torch.sort(torch.arange(g.columns) // 64).values)
    # the stored parameters rebuild the artifact through expand_scales, as an inference
    # kernel would; this is what makes g_idx's 0.002075 bpw an honest charge.
    C = get_codebook("strict_pot")
    S = expand_scales(out["scales"], g.columns, group=64, g_idx=gi)
    assert torch.equal(C.levels[out["codes"].long()] * S, out["W_hat"])


@pytest.mark.parametrize("order", NEW)
def test_artifact_is_strict_pot(order):
    _, g = _fit()
    h = _rtn(g)
    out = _q(g, order, h=h)
    ok, v = verify_codebook_artifact(out["W_hat"], out["scales"],
                                     get_codebook("strict_pot"), group=64,
                                     g_idx=out["g_idx"])
    assert ok is True, v


# ------------------------------------------- the reversal is a NULL change under rtn
def test_reversing_the_permutation_is_a_null_change_under_rtn():
    """F3 of the pre-registration. Reversing a permutation maps group g to group
    G-1-g and leaves every group SET identical, and under H := I nothing else depends
    on the order. So `logmeanasc` is a machinery test, not a falsification control --
    the real control has to be `rand`."""
    _, g = _fit()
    _rtn(g)
    a, b = _q(g, "logmean"), _q(g, "logmeanasc")
    assert torch.equal(a["W_hat"], b["W_hat"])
    assert torch.equal(a["codes"][:, torch.argsort(torch.arange(g.columns))],
                       b["codes"])
    # and it is NOT a null change when the sweep has a real Hessian (GPTQ path)
    _, g2 = _fit()
    c = g2.quantize(get_codebook("strict_pot"), group=64, order="logmean",
                    exact_scale=True)
    d = g2.quantize(get_codebook("strict_pot"), group=64, order="logmeanasc",
                    exact_scale=True)
    assert not torch.equal(c["W_hat"], d["W_hat"])


# ------------------------------------------------------------------- the keys
def test_logmean_key_is_the_log_location():
    _, g = _fit()
    C = get_codebook("strict_pot").levels
    perm = g._column_perm("logmean", g.W, C, torch.diag(g.H), 64,
                          dict(pot_scale=False, exact_scale=True), None, None)
    mu = torch.log2(g.W.abs().clamp_min(1e-30)).mean(0)
    assert torch.equal(perm, torch.argsort(mu, descending=True))


def test_salogmean_key_is_logh_plus_two_mu_and_uses_perm_hess():
    _, g = _fit()
    C = get_codebook("strict_pot").levels
    h = _rtn(g)                     # H is now I; the key must still see the real h
    mu = torch.log2(g.W.abs().clamp_min(1e-30)).mean(0)
    sk = dict(pot_scale=False, exact_scale=True)
    perm = g._column_perm("salogmean", g.W, C, torch.diag(g.H), 64, sk, None, None,
                          perm_hess=h)
    assert torch.equal(perm, torch.argsort(2 * mu + torch.log2(h.clamp_min(1e-30)),
                                           descending=True))
    # without perm_hess it would fall back to the (now constant) diag(H) -- i.e. the
    # salience term would be silently inert. That is exactly the bug run_pot.py's
    # H-capture exists to prevent, so pin the difference.
    plain = g._column_perm("salogmean", g.W, C, torch.diag(g.H), 64, sk, None, None)
    assert not torch.equal(perm, plain)


def test_rand_is_deterministic_and_seed_sensitive():
    _, g = _fit()
    _rtn(g)
    a = _q(g, "rand", perm_seed=0)
    b = _q(g, "rand", perm_seed=0)
    c = _q(g, "rand", perm_seed=7)
    assert torch.equal(a["g_idx"], b["g_idx"]) and torch.equal(a["W_hat"], b["W_hat"])
    assert not torch.equal(a["g_idx"], c["g_idx"])


def test_logmean_reduces_within_group_log_spread():
    """The mechanism claim, on a matrix built to have column-location heterogeneity."""
    _, g = _fit()
    h = _rtn(g)
    L = torch.log2(g.W.abs().clamp_min(1e-30))

    def spread(perm):
        V = L[:, perm].view(L.shape[0], L.shape[1] // 64, 64)
        return float((V.amax(-1) - V.amin(-1)).mean())

    C = get_codebook("strict_pot").levels
    sk = dict(pot_scale=False, exact_scale=True)
    p_lm = g._column_perm("logmean", g.W, C, torch.diag(g.H), 64, sk, None, None)
    p_rd = g._column_perm("rand", g.W, C, torch.diag(g.H), 64, sk, None, None)
    assert spread(p_lm) < spread(torch.arange(g.columns))
    assert spread(p_lm) < spread(p_rd)


def test_unknown_order_still_rejected():
    _, g = _fit()
    with pytest.raises(ValueError):
        _q(g, "logmeandesc")


def test_permutation_only_changes_group_membership():
    """Purity check for the `rand` arm: a permutation must be a pure relabelling of
    WHICH COLUMNS SHARE A SCALE and nothing else.

    The independent probe is group == in_features: with a single group per row there
    is no membership left to change, so under rtn every order must produce a
    BIT-IDENTICAL artifact. Anything else (a different rounding, a reordered scale
    search, a lost column) would survive this test and show up here.
    """
    _, g = _fit(I=256, O=48)
    h = _rtn(g)
    C = get_codebook("strict_pot")
    base = g.quantize(C, group=256, percdamp=0.0, order="none", exact_scale=True)
    for order in ("rand", "logmean", "errmag", "salogmean"):
        out = g.quantize(C, group=256, percdamp=0.0, order=order, exact_scale=True,
                         perm_hess=h)
        assert torch.equal(out["W_hat"], base["W_hat"]), order
        assert torch.equal(out["codes"], base["codes"]), order
        assert torch.equal(out["scales"], base["scales"]), order


def test_rand_preserves_group_count_and_sizes():
    """The bpw group_scale term must not move: same number of groups, same sizes."""
    _, g = _fit()
    _rtn(g)
    a, b = _q(g, "none"), _q(g, "rand")
    assert a["n_groups"] == b["n_groups"] == 4
    assert a["scales"].shape == b["scales"].shape
    assert torch.equal(torch.bincount(b["g_idx"]), torch.bincount(a["g_idx"]))
