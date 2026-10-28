"""`--refine-hessian calib`: coordinate descent on the code assignment from an RTN
start, run against the REAL calibration Hessian instead of the identity H that
`--method rtn` installs.

Covers:
  * the crux bug this flag exists for: with H = I (what `--method rtn` leaves behind),
    refinement from an RTN start is a provable no-op -- so the previously-only-possible
    invocation measured nothing;
  * defaults are bit-identical to before `trace=` existed;
  * with a real H the descent is monotone, the incremental `dL` bookkeeping agrees with
    a brute-force recomputation of tr(E H E^T), and the strict-PoT invariant survives.
"""
import torch

from pot.codebooks import strict_pot, int4_sym
from pot.code_refine import refine_codes, row_objective
from pot.gptq_pot import PoTGPTQ
from pot.verify_shiftfree import verify_codebook_artifact


def _rtn_artifact(rows=16, cols=64, group=16, cb=None, seed=0, dev="cpu"):
    """Exactly what run_pot.py --method rtn produces: H := I, percdamp 0."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(cols, rows, bias=False).to(dev)
    lin.weight.data = torch.randn(rows, cols, device=dev) * 0.05
    g = PoTGPTQ(lin, device=dev)
    X = torch.randn(96, cols, device=dev)
    g.add_batch(X)
    H_calib = g.H.clone()
    g.H = torch.eye(cols, device=dev)
    g.nsamples = 1
    out = g.quantize(cb, group=group, percdamp=0.0)
    return g, out, H_calib, torch.eye(cols, device=dev)


def test_identity_H_makes_refinement_a_noop_from_rtn():
    """THE CRUX. RTN is the per-entry weight-MSE minimiser given the scale, so with
    H = I no adjacent-level move can strictly reduce tr(E I E^T): zero moves, codes
    bit-identical. That is why `--method rtn --refine-codes` measured nothing before
    `--refine-hessian calib`."""
    cb = strict_pot(4)
    g, out, H_calib, H_eye = _rtn_artifact(cb=cb)
    r = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                     cb, H_eye, sweeps=3)
    assert r["n_moves"] == 0
    assert torch.equal(r["codes"].long(), out["codes"].long())
    assert r["obj_after"] == r["obj_before"]


def test_real_H_from_rtn_moves_and_decreases_monotonically():
    for cb in (strict_pot(4), int4_sym()):
        g, out, H_calib, _ = _rtn_artifact(cb=cb)
        r = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"],
                         out["col_scale"], cb, H_calib, sweeps=4, trace=True)
        assert r["n_moves"] > 0, f"{cb.name}: real H should find improving moves"
        assert r["obj_after"] < r["obj_before"]
        tr = r["obj_trace"]
        assert len(tr) == 4
        for a, b in zip(tr, tr[1:]):
            assert b <= a + 1e-6, f"{cb.name}: objective increased {a} -> {b}"
        # the incremental dL bookkeeping must agree with a from-scratch recomputation
        E = g.W - r["W_hat"]
        assert abs(float(row_objective(E, H_calib).sum()) - r["obj_after"]) < 1e-3 * abs(
            r["obj_before"])


def test_refined_artifact_still_verifies_strict_pot():
    cb = strict_pot(4)
    g, out, H_calib, _ = _rtn_artifact(cb=cb)
    r = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                     cb, H_calib, sweeps=2)
    ok, diag = verify_codebook_artifact(r["W_hat"], out["scales"], cb,
                                        group=out["group"], g_idx=out["g_idx"],
                                        col_scale=out["col_scale"])
    assert ok is True, diag


def test_trace_flag_does_not_change_the_result():
    """Defaults (trace=False) are bit-identical to trace=True's codes/W_hat, so adding
    the trace argument cannot have perturbed any previously-run configuration."""
    cb = strict_pot(4)
    g, out, H_calib, _ = _rtn_artifact(cb=cb)
    a = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                     cb, H_calib, sweeps=3)
    b = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                     cb, H_calib, sweeps=3, trace=True)
    assert torch.equal(a["codes"].long(), b["codes"].long())
    assert torch.equal(a["W_hat"], b["W_hat"])
    assert a["n_moves"] == b["n_moves"] and a["obj_after"] == b["obj_after"]
    assert "obj_trace" not in a


def test_zero_sweeps_is_the_anchor():
    cb = strict_pot(4)
    g, out, H_calib, _ = _rtn_artifact(cb=cb)
    r = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                     cb, H_calib, sweeps=0, trace=True)
    assert r["n_moves"] == 0 and r["obj_trace"] == []
    assert torch.equal(r["codes"].long(), out["codes"].long())


def test_alignment_ratio_is_tracked_and_norm_is_bounded():
    """The coordinator's quantity of interest: tr(EHE)/tr(EE), normalised by
    tr(H)/cols so a randomly-oriented E scores 1.0. Coordinate descent over a FIXED
    codeword set cannot inflate ||E|| the way GPTQ's retargeting does (GPTQ: +43%
    rel_mse), so err2 must stay in a narrow band while the alignment moves."""
    cb = strict_pot(4)
    g, out, H_calib, _ = _rtn_artifact(cb=cb)
    r = refine_codes(g.W, out["codes"], out["scales"], out["g_idx"], out["col_scale"],
                     cb, H_calib, sweeps=4, trace=True)
    assert len(r["err2_trace"]) == 4
    # RTN minimises ||E||^2 exactly, so any move can only increase it
    assert r["err2_after"] >= r["err2_before"]
    # ... but only within a narrow band: nothing like GPTQ's +43%
    assert r["err2_after"] < 1.6 * r["err2_before"]
    align_b = r["obj_before"] / r["err2_before"] / r["h_norm"]
    align_a = r["obj_after"] / r["err2_after"] / r["h_norm"]
    assert align_a < align_b, "descending tr(EHE) should lower the alignment ratio"
    assert 0.0 < align_b < 3.0
