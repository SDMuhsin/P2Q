"""Correctness anchors for PoT-PTQ step 2, the learned scale residual.

The load-bearing claims, in order of importance:

  1. Whatever the optimizer does to Gamma, the artifact still factors as
     `(+-2^e) * scale` -- machine-checked with the project's own verifier, not a
     hand-rolled tolerance test. This is the immutable condition; if it can be broken
     by learning, the whole result is void.
  2. Gamma = 0 is an exact no-op: with `retarget='gptq'` the finalized artifact is
     bit-identical to what GPTQ produced, so `--learn-scale` cannot silently change the
     baseline before it has learned anything.
  3. The objective actually goes down.
"""
import pytest
import torch

from pot.codebooks import get_codebook
from pot.gptq_pot import PoTGPTQ
from pot.scale_learn import ScaledPoTLinear, learn_scale_residual, rebuild_w_hat
from pot.verify_shiftfree import verify_strict_pot


def _fixture(rows=32, cols=128, group=32, n=256, seed=0, device="cpu"):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(cols, rows, bias=False).to(device)
    lin.weight.data = torch.randn(rows, cols, device=device) * 0.02
    X = torch.randn(n, cols, device=device)
    X = X @ (torch.eye(cols, device=device) + 0.3 * torch.randn(cols, cols, device=device))
    cb = get_codebook("strict_pot", device=device, dtype=torch.float32)
    g = PoTGPTQ(lin)
    g.add_batch(X)
    out = g.quantize(cb, group=group, blocksize=group, return_targets=True)
    out["W_orig"] = g.W.clone()
    out["H"] = g.H.clone()
    return lin, cb, out, X


def test_gamma_zero_is_bit_identical_to_gptq():
    lin, cb, out, _ = _fixture()
    sh = ScaledPoTLinear(out, cb.levels, retarget="gptq")
    fin = sh.finalize()
    assert torch.equal(fin["W_hat"], out["W_hat"])
    assert torch.equal(fin["codes"], out["codes"])
    assert torch.equal(fin["scales"], out["scales"])


@pytest.mark.parametrize("retarget", ["gptq", "orig", "none"])
def test_invariant_holds_for_arbitrary_gamma(retarget):
    """The point of the whole file: a RANDOM, large Gamma must still leave every weight
    on the power-of-two grid. If the invariant depended on the optimizer converging, it
    would not be an invariant."""
    lin, cb, out, _ = _fixture()
    sh = ScaledPoTLinear(out, cb.levels, retarget=retarget)
    with torch.no_grad():
        sh.gamma.copy_(torch.randn_like(sh.gamma) * 0.5)
    fin = sh.finalize()
    ok, diag = verify_strict_pot(fin["W_hat"], fin["scales"], group=out["group"],
                                 g_idx=fin["g_idx"], col_scale=fin["col_scale"])
    assert ok, diag["examples"][:2]
    assert torch.equal(rebuild_w_hat(fin, cb.levels), fin["W_hat"])
    if retarget != "none":
        # a scale that moved must actually re-assign codes, else the "learning" is inert
        assert not torch.equal(fin["codes"], out["codes"])


def test_learn_layer_scope_reduces_output_error_and_verifies():
    lin, cb, out, _ = _fixture()
    before = out["proxy_err"]
    info = learn_scale_residual(lin, quant_state=out, codebook=cb, scope="layer",
                                iters=200, lr=1e-3, device="cpu", verify=True)
    assert info["verify_ok"] is True
    e = info["linears"][""]
    assert e["err_after"] < e["err_before"], (e["err_before"], e["err_after"])
    assert info["loss_last"] <= info["loss_first"]
    # the module got the refined weight, and it is still on the grid
    ok, _ = verify_strict_pot(out["W_hat"], out["scales"], group=out["group"],
                              g_idx=out["g_idx"], col_scale=out["col_scale"])
    assert ok
    assert before > 0


def test_retarget_on_a_gptq_artifact_is_the_worse_option():
    """Pins the measured finding documented in scale_learn.py: re-rounding the codes
    under a moving scale destroys GPTQ's error feedback, so it must NOT be the default.
    If this ever starts passing the other way, the docstring is stale."""
    lin, cb, out, _ = _fixture()
    keep = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in out.items()}
    a = learn_scale_residual(lin, quant_state=out, codebook=cb, scope="layer",
                             iters=200, lr=1e-3, retarget="none", device="cpu")
    b = learn_scale_residual(lin, quant_state=keep, codebook=cb, scope="layer",
                             iters=200, lr=1e-3, retarget="gptq", device="cpu")
    assert a["linears"][""]["err_after"] < b["linears"][""]["err_after"]
    assert a["verify_ok"] is True and b["verify_ok"] is True


def test_invariant_survives_the_bf16_cast_the_driver_applies():
    """`run_pot.py` writes `W_hat.to(bf16)` into the module, so the thing actually
    evaluated is the bf16 artifact, not the fp32 one. For a strict-PoT code that is still
    exact -- multiplying by +-2^e only shifts the exponent field -- and this pins it, so a
    future codebook change cannot silently make the evaluated model unverifiable."""
    lin, cb, out, _ = _fixture()
    sh = ScaledPoTLinear(out, cb.levels, retarget="none")
    with torch.no_grad():
        sh.gamma.copy_(torch.randn_like(sh.gamma) * 0.4)
    fin = sh.finalize()
    ok, diag = verify_strict_pot(fin["W_hat"].to(torch.bfloat16),
                                 fin["scales"].to(torch.bfloat16),
                                 group=out["group"], g_idx=fin["g_idx"])
    assert ok, diag["examples"][:2]


def test_learn_col_stays_shift_free():
    """The optional per-column term multiplies the scale, so it must not disturb the
    factorization either -- the verifier is told about it via col_scale."""
    lin, cb, out, _ = _fixture()
    info = learn_scale_residual(lin, quant_state=out, codebook=cb, scope="layer",
                                iters=30, lr=1e-3, learn_col=True, device="cpu",
                                verify=True)
    assert info["verify_ok"] is True
    assert out["col_scale"] is not None
