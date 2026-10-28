"""Tests for the format-native log-domain distortion (src/pot/log_scale.py).

Covers, in order:
  1. defaults are bit-identical to the pre-change pipeline (both new flags OFF);
  2. the exact minimizer really attains the GLOBAL minimum (vs dense brute force);
  3. the breakpoint enumeration is COMPLETE (no assignment change is missed) and
     SUFFICIENT (assignment is constant strictly between consecutive breakpoints);
  4. the linear-vs-log decision boundary is exactly where the derivation says;
  5. stored weights stay strict PoT under every cell of the 2x2.
"""
import math

import pytest
import torch

from pot.codebooks import strict_pot, int4_sym, nearest_codeword, get_codebook
from pot.exact_scale import exact_group_scale
from pot.gptq_pot import PoTGPTQ
from pot.log_scale import (log_magnitudes, log_boundaries, log_nearest_codeword,
                           log_group_error, log_group_scale, clamp_diagnostics,
                           count_degenerate)
from pot.verify_shiftfree import verify_codebook_artifact


def _cb():
    return strict_pot(8).to(dtype=torch.float32)


# --------------------------------------------------------------- 1. bit-identity
def _linear_rtn_reference(W, C, group):
    """The pre-change RTN artifact, written with ONLY the old primitives:
    exact squared-error scale search + linear nearest-codeword."""
    rows, cols = W.shape
    n_groups = cols // group
    Q = torch.zeros_like(W)
    S = torch.zeros(rows, n_groups, device=W.device)
    for gi in range(n_groups):
        sl = slice(gi * group, (gi + 1) * group)
        s, _ = exact_group_scale(W[:, sl], C)
        S[:, gi] = s.squeeze(1)
        Q[:, sl] = C[nearest_codeword(W[:, sl] / s, C)] * s
    return Q, S


@pytest.mark.parametrize("exact", [True, False])
def test_defaults_are_bit_identical(exact):
    """With --log-scale / --log-round OFF the sweep is bit-for-bit what it was.

    Checked against an independent reimplementation that never imports log_scale,
    so this is a real equivalence, not `f(x) == f(x)`.
    """
    torch.manual_seed(0)
    rows, cols, group = 16, 128, 64
    lin = torch.nn.Linear(cols, rows, bias=False)
    cb = _cb()
    g = PoTGPTQ(lin)
    g.H = torch.eye(cols)
    g.nsamples = 1
    out = g.quantize(cb, group=group, percdamp=0.0, exact_scale=exact)
    if exact:
        Qref, Sref = _linear_rtn_reference(lin.weight.data.float(), cb.levels, group)
        assert torch.equal(out["W_hat"], Qref)
        assert torch.equal(out["scales"], Sref)
    assert out["log_scale"] is False and out["log_round"] is False
    # and the explicit-default spelling is identical to the omitted one
    g2 = PoTGPTQ(lin)
    g2.H = torch.eye(cols)
    g2.nsamples = 1
    out2 = g2.quantize(cb, group=group, percdamp=0.0, exact_scale=exact,
                       log_scale=False, log_round=False)
    assert torch.equal(out["W_hat"], out2["W_hat"])
    assert torch.equal(out["codes"], out2["codes"])


def test_log_flags_actually_change_something():
    """Guard against a no-op flag: each switch must move the artifact on real-ish data."""
    torch.manual_seed(1)
    lin = torch.nn.Linear(128, 16, bias=False)
    cb = _cb()
    arts = {}
    for ls, lr in [(False, False), (False, True), (True, False), (True, True)]:
        g = PoTGPTQ(lin)
        g.H = torch.eye(128)
        g.nsamples = 1
        kw = dict(group=64, percdamp=0.0, log_round=lr)
        kw.update(dict(log_scale=True) if ls else dict(exact_scale=True))
        arts[(ls, lr)] = g.quantize(cb, **kw)["W_hat"]
    base = arts[(False, False)]
    for k, v in arts.items():
        if k != (False, False):
            assert not torch.equal(base, v), f"cell {k} is a no-op"


# ------------------------------------------------------- 2. global optimality
def _brute_min(W, C, h=None, n=200001, span=16.0):
    ts = torch.linspace(-span, span, n, dtype=torch.float64)
    B, g = W.shape
    best = torch.full((B,), float("inf"), dtype=torch.float64)
    for i in range(0, n, 2000):
        tt = ts[i:i + 2000]
        Wr = W.unsqueeze(1).expand(B, tt.numel(), g).reshape(-1, g)
        sr = torch.exp2(tt.view(1, -1).expand(B, tt.numel()).reshape(-1, 1))
        e = log_group_error(Wr, C, sr, h).reshape(B, -1)
        best = torch.minimum(best, e.min(1).values)
    return best


@pytest.mark.parametrize("seed,scale,weighted", [(0, 1.0, False), (1, 1.0, True),
                                                 (2, 1e-4, False), (3, 30.0, True)])
def test_exact_minimizer_attains_global_minimum(seed, scale, weighted):
    torch.manual_seed(seed)
    C = strict_pot(8).levels.double()
    W = torch.randn(4, 12, dtype=torch.float64) * scale
    if seed == 3:                                    # heavy tail: forces clipping
        W = W * torch.exp(torch.randn(4, 12, dtype=torch.float64) * 2.5)
    h = (torch.rand(12, dtype=torch.float64) + 0.1) if weighted else None
    _, e = log_group_scale(W, C, hess_diag=h)
    bf = _brute_min(W, C, h)
    # exact must be <= brute force everywhere, up to the float32 return dtype
    assert torch.all(e.double() <= bf * (1 + 1e-5) + 1e-9)
    assert torch.all(e.double() >= bf * (1 - 1e-4) - 1e-9)


def test_exact_minimizer_beats_absmax_and_exact_mse_on_its_own_objective():
    torch.manual_seed(4)
    C = strict_pot(8).levels.double()
    W = torch.randn(32, 64, dtype=torch.float64)
    s_log, e_log = log_group_scale(W, C)
    s_mse, _ = exact_group_scale(W, C)
    s_am = W.abs().amax(1, keepdim=True) / float(C.abs().max())
    for s in (s_mse.double(), s_am):
        assert torch.all(e_log.double() <= log_group_error(W, C, s) + 1e-9)


def test_objective_is_continuous_in_t():
    """D(t) is C^0: at a breakpoint the two competing lattice points are equidistant
    in log-magnitude, so the residual flips sign but its square does not jump."""
    torch.manual_seed(5)
    C = strict_pot(8).levels.double()
    W = torch.randn(8, 32, dtype=torch.float64)
    u = torch.log2(W[0].abs())
    bps = torch.cat([u - m for m in
                     torch.arange(-6.5, 0.0, 1.0, dtype=torch.float64)])
    eps = 1e-9
    for t in bps[:40]:
        a = log_group_error(W[:1], C, torch.tensor([[2.0 ** (t - eps)]], dtype=torch.float64))
        b = log_group_error(W[:1], C, torch.tensor([[2.0 ** (t + eps)]], dtype=torch.float64))
        assert abs(float(a) - float(b)) < 1e-6


# ------------------------------------------------- 3. breakpoint completeness
def test_breakpoint_enumeration_is_complete_and_sufficient():
    """Every t at which the log-domain ASSIGNMENT changes is one of the enumerated
    breakpoints `u_i - (a_k + a_{k+1})/2`, and nothing changes strictly between two
    consecutive ones. Verified by a dense scan against the enumerated set."""
    torch.manual_seed(6)
    C = strict_pot(8).levels.double()
    a = torch.log2(log_magnitudes(C))
    mid = (a[:-1] + a[1:]) * 0.5
    w = torch.randn(9, dtype=torch.float64)
    u = torch.log2(w.abs())
    bps = torch.sort((u.unsqueeze(1) - mid.unsqueeze(0)).reshape(-1)).values

    lo, hi = float(bps.min()) - 2, float(bps.max()) + 2
    ts = torch.linspace(lo, hi, 400001, dtype=torch.float64)
    codes = torch.bucketize((u.unsqueeze(0) - ts.unsqueeze(1)).contiguous(),
                            mid.contiguous())                       # [n, g]
    changed = (codes[1:] != codes[:-1]).any(1)
    idx = torch.nonzero(changed).squeeze(1)
    step = float(ts[1] - ts[0])
    # COMPLETENESS: every observed change sits within one sample of an enumerated bp
    for i in idx.tolist():
        d = (bps - ts[i]).abs().min()
        assert float(d) <= 2 * step, f"unenumerated assignment change near t={ts[i]}"
    # SUFFICIENCY: every enumerated (distinct) breakpoint really is a change point
    for b in torch.unique(bps).tolist():
        c0 = torch.bucketize(u - (b - 1e-7), mid)
        c1 = torch.bucketize(u - (b + 1e-7), mid)
        assert not torch.equal(c0, c1), f"enumerated breakpoint {b} is not one"
    # COUNT: exactly g*(p-1) breakpoints, no clip-threshold extras
    assert bps.numel() == 9 * (a.numel() - 1)


def test_no_extra_breakpoint_at_the_clip_thresholds():
    """Clipping is handled by the END lattice point winning the argmin, so there is
    NO assignment change at u - t = a_1 or a_p +- 1/2."""
    C = strict_pot(8).levels.double()
    a = torch.log2(log_magnitudes(C))
    mid = (a[:-1] + a[1:]) * 0.5
    for v in (float(a.min()) - 0.5, float(a.max()) + 0.5):
        c0 = torch.bucketize(torch.tensor([v - 1e-6], dtype=torch.float64), mid)
        c1 = torch.bucketize(torch.tensor([v + 1e-6], dtype=torch.float64), mid)
        assert torch.equal(c0, c1)


# --------------------------------------------- 4. the decision-boundary claim
def test_log_boundary_is_the_geometric_midpoint():
    C = strict_pot(8).levels
    lb = log_boundaries(C)
    linb = (C[1:].double() + C[:-1].double()) * 0.5
    pos = C > 0
    n = int(pos.sum())
    # positive side: boundaries are sqrt(2)*2^(e-1) vs 1.5*2^(e-1)
    for k in range(len(C) - n, len(C) - 1):
        lo, hi = float(C[k]), float(C[k + 1])
        assert math.isclose(float(lb[k]), math.sqrt(lo * hi), rel_tol=1e-12)
        assert math.isclose(float(lb[k]), math.sqrt(2) * lo, rel_tol=1e-12)
        assert math.isclose(float(linb[k]), 1.5 * lo, rel_tol=1e-12)
    # the single sign-crossing pair keeps the arithmetic (== 0) boundary
    j = int(torch.nonzero(C > 0)[0]) - 1
    assert float(lb[j]) == 0.0 == float(linb[j])


def test_linear_and_log_rounding_differ_exactly_on_the_predicted_band():
    """x/s in (sqrt2 * 2^(e-1), 1.5 * 2^(e-1)) rounds UP under log and DOWN under
    linear, and nowhere else do the two disagree."""
    C = strict_pot(8).levels.double()
    x = torch.linspace(1e-3, 1.0, 2_000_001, dtype=torch.float64)
    kl = nearest_codeword(x, C)
    kg = log_nearest_codeword(x, C)
    diff = kl != kg
    assert bool(diff.any())
    assert torch.all(kg[diff] == kl[diff] + 1), "log rounding must round UP, never down"
    xd = x[diff]
    lo_lev = C[kl[diff]]
    assert torch.all(xd > math.sqrt(2) * lo_lev * (1 - 1e-9))
    assert torch.all(xd < 1.5 * lo_lev * (1 + 1e-9))
    # width of the disagreement band, as a fraction of the interval, is fixed:
    # (1.5 - sqrt2) / (2 - 1) * ... -> 8.58% of each octave
    frac = float(diff.double().mean())
    assert 0.05 < frac < 0.12


# ------------------------------------------------------------ 5. strict-PoT-ness
@pytest.mark.parametrize("log_scale,log_round", [(False, False), (False, True),
                                                 (True, False), (True, True)])
def test_artifact_stays_strict_pot(log_scale, log_round):
    torch.manual_seed(7)
    lin = torch.nn.Linear(128, 24, bias=False)
    cb = _cb()
    g = PoTGPTQ(lin)
    g.H = torch.eye(128)
    g.nsamples = 1
    kw = dict(group=64, percdamp=0.0, log_round=log_round)
    kw.update(dict(log_scale=True) if log_scale else dict(exact_scale=True))
    out = g.quantize(cb, **kw)
    ok, diag = verify_codebook_artifact(out["W_hat"], out["scales"], cb,
                                        group=out["group"], g_idx=out["g_idx"],
                                        col_scale=out["col_scale"])
    assert ok is True, diag
    r = (out["W_hat"] / out["scales"][:, out["g_idx"]]).abs()
    r = r[r > 0]
    assert torch.allclose(torch.log2(r), torch.round(torch.log2(r)), atol=1e-5)


# --------------------------------------------------- sign / degenerate handling
def test_sign_is_never_lost():
    torch.manual_seed(8)
    lin = torch.nn.Linear(64, 8, bias=False)
    cb = _cb()
    g = PoTGPTQ(lin)
    g.H = torch.eye(64)
    g.nsamples = 1
    out = g.quantize(cb, group=64, percdamp=0.0, log_scale=True, log_round=True)
    W = lin.weight.data.float()
    assert torch.all(torch.sign(out["W_hat"]) == torch.sign(W))
    assert torch.all(out["W_hat"] != 0)


def test_zero_weights_do_not_poison_the_fit():
    C = strict_pot(8).levels.double()
    W = torch.randn(3, 16, dtype=torch.float64)
    W[:, :4] = 0.0
    s, e = log_group_scale(W, C)
    assert torch.isfinite(s).all() and torch.isfinite(e).all()
    # the fit must equal the fit on the surviving weights alone
    s2, _ = log_group_scale(W[:, 4:], C)
    assert torch.allclose(s, s2, rtol=1e-5)
    # an all-zero group falls back to absmax rather than NaN
    s3, _ = log_group_scale(torch.zeros(2, 8, dtype=torch.float64), C)
    assert torch.isfinite(s3).all() and torch.all(s3 > 0)
    d = count_degenerate(W)
    assert d["n_zero"] == 12 and d["n_nonfinite"] == 0


def test_zero_containing_alphabets_are_refused():
    with pytest.raises(ValueError, match="undefined"):
        log_magnitudes(int4_sym().levels)
    with pytest.raises(ValueError, match="undefined"):
        log_magnitudes(get_codebook("strict_pot_zero").levels)


def test_pot_scale_returns_exact_powers_of_two():
    torch.manual_seed(9)
    C = strict_pot(8).levels.double()
    W = torch.randn(8, 32, dtype=torch.float64)
    s, _ = log_group_scale(W, C, pot_scale=True)
    l = torch.log2(s.double())
    assert torch.allclose(l, torch.round(l), atol=1e-5)
    s_free, e_free = log_group_scale(W, C)
    _, e_pot = log_group_scale(W, C, pot_scale=True)
    assert torch.all(e_pot >= e_free - 1e-6)          # constrained is never better


# ----------------------------------------------------------------- diagnostics
def test_clamp_diagnostics_sees_what_rel_mse_cannot():
    """The 4-octave collapse, in miniature: narrowing the alphabet barely moves
    weight MSE while it blows up the RELATIVE error of the bottom-rung clamp.
    (On real Qwen3-1.7B weights the 4-octave book actually WINS on rel_mse; on
    i.i.d. Gaussians it merely draws. Either way MSE is nearly blind and the log
    measure is not, which is the claim under test.)"""
    torch.manual_seed(10)
    W = torch.randn(64, 16)
    res = {}
    for nexp in (8, 4):
        cb = strict_pot(nexp).to(dtype=torch.float32)
        s, _ = exact_group_scale(W, cb.levels)
        codes = nearest_codeword(W / s, cb.levels)
        Q = cb.levels[codes] * s
        d = clamp_diagnostics(W, Q, cb.levels, codes=codes)
        d["rel_mse"] = float(((Q - W) ** 2).sum() / (W ** 2).sum())
        res[nexp] = d
    mse_ratio = res[4]["rel_mse"] / res[8]["rel_mse"]
    log_ratio = res[4]["log_mse"] / res[8]["log_mse"]
    assert mse_ratio < 1.5, mse_ratio            # MSE hardly notices
    assert log_ratio > 4.0, log_ratio            # the log measure screams
    assert log_ratio > 3 * mse_ratio
    assert res[4]["clamp_lo_frac"] > 10 * res[8]["clamp_lo_frac"]
    assert res[4]["clamp_lo_relerr"] > 2 * res[8]["clamp_lo_relerr"]
    # a clamped-up weight is not "slightly wrong": it is wrong by a FACTOR
    assert res[8]["clamp_lo_relerr"] > 1.0 and res[4]["clamp_lo_relerr"] > 1.0


# ------------------------------- salience log-domain (--log-scale-hess, §21.5)
def test_hess_weights_enter_the_log_fit_as_h_alone():
    """`sum_j h_j (log2|w_hat|-log2|w|)^2`: the per-column weight must act on the
    LOG residual with NO w^2 factor. Checked by construction -- putting all the
    weight on one column makes that column's fit exact."""
    C = strict_pot(8).levels.double()
    torch.manual_seed(11)
    W = torch.randn(1, 8, dtype=torch.float64)
    W[0, 3] *= 1e-3                                  # a small weight: w^2 would ignore it
    h = torch.full((8,), 1e-6, dtype=torch.float64)
    h[3] = 1.0                                       # salience concentrated on it
    s, _ = log_group_scale(W, C, hess_diag=h)
    u = torch.log2(W[0, 3].abs())
    t = torch.log2(s[0, 0].double())
    a = torch.log2(log_magnitudes(C))
    # the scale must place THAT weight essentially on a lattice point
    assert float(((u - t) - a).abs().min()) < 1e-4   # float32 scale return
    # and the unweighted fit must not
    s0, _ = log_group_scale(W, C)
    t0 = torch.log2(s0[0, 0].double())
    assert float(((u - t0) - a).abs().min()) > 1e-2


def test_hess_weighting_changes_the_scale_and_stays_exact():
    torch.manual_seed(12)
    C = strict_pot(8).levels.double()
    W = torch.randn(4, 32, dtype=torch.float64)
    h = torch.rand(32, dtype=torch.float64) ** 4 + 1e-3      # heterogeneous, like real h
    s_h, e_h = log_group_scale(W, C, hess_diag=h)
    s_u, _ = log_group_scale(W, C)
    assert not torch.allclose(s_h, s_u)
    # still the exact minimizer of ITS objective
    assert torch.all(e_h.double() <= log_group_error(W, C, s_u.double(), h) + 1e-9)


def test_log_scale_hess_requires_log_scale():
    """CLI guard: the salience weighting is meaningless without the log objective."""
    import subprocess, sys as _s, os
    r = subprocess.run([_s.executable, os.path.join(os.path.dirname(__file__),
                                                    "..", "..", "src", "pot",
                                                    "run_pot.py"),
                        "--log-scale-hess", "--dry-run", "--max-layers", "0"],
                       capture_output=True, text=True, timeout=300)
    assert "meaningless without --log-scale" in (r.stdout + r.stderr)


# ----------------------------------- diag(H) adequacy diagnostic (diag_adequacy.py)
def test_block_mask_is_the_scale_sharing_unit():
    from pot.diag_adequacy import block_mask
    m = block_mask(8, 4, torch.device("cpu"))
    assert m.shape == (8, 8)
    assert bool(m[0, 3]) and bool(m[4, 7])            # inside a group
    assert not bool(m[3, 4]) and not bool(m[0, 7])    # across groups
    assert int(m.sum()) == 2 * 4 * 4


def test_signflip_null_preserves_the_diagonal_term_exactly():
    """The control's whole validity rests on this: randomizing signs must leave
    `sum_j H_jj ||E[:,j]||^2` bit-identical while destroying cross-column structure."""
    from pot.diag_adequacy import ratios
    torch.manual_seed(13)
    E = torch.randn(32, 16)
    A = torch.randn(16, 16)
    H = A @ A.t()
    Hd = torch.diag(H)
    gen = torch.Generator(device=E.device); gen.manual_seed(0)
    r = ratios(E, H, Hd, gen, n_flip=4)               # asserts the invariant internally
    assert r["rho_signflip_sd"] > 0
    # a PERFECTLY uncorrelated E has rho ~ its own null; the statistic is well centred
    assert abs(r["rho"] - r["rho_signflip"]) < 6 * r["rho_signflip_sd"]
