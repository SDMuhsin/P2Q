"""Correctness anchors for the GPTQ implementation.

The single most important test in the project is `test_identity_hessian_is_rtn`: with
H = I there is no correlation between input channels, so GPTQ's feedback term is zero and
the sweep MUST degenerate to plain round-to-nearest against the same codebook and scale.
If that does not hold, nothing measured with this quantizer means anything.
"""
import pytest
import torch

from pot.codebooks import get_codebook, nearest_codeword
from pot.gptq_pot import PoTGPTQ
from pot.scale_search import search_group_scale
from pot.verify_shiftfree import verify_codebook_artifact


def rtn(W, C, group=128, **kw):
    """Plain round-to-nearest with the same per-group scale search."""
    O, I = W.shape
    Wg = W.reshape(-1, group)
    s, _ = search_group_scale(Wg, C, **kw)
    Wh = (C.levels[nearest_codeword(Wg / s, C.levels)] * s).reshape(O, I)
    return Wh, s.reshape(O, I // group)


def _fit_identity(W, C, group=128, **kw):
    """Feed activations whose Gram is exactly the identity."""
    O, I = W.shape
    g = PoTGPTQ(W)
    g.H = torch.eye(I, device=W.device)         # exactly I, no sampling noise
    g.nsamples = 1
    return g.quantize(C, group=group, percdamp=0.0, **kw)


@pytest.mark.parametrize("name", ["int4_asym", "int4_sym", "strict_pot", "learned_add2"])
def test_identity_hessian_is_rtn(name):
    """THE anchor: H = I  =>  GPTQ == RTN, bit-for-bit up to float noise."""
    torch.manual_seed(0)
    W = torch.randn(64, 512) * 0.02
    C = get_codebook(name)
    out = _fit_identity(W, C)
    Wh_rtn, s_rtn = rtn(W, C)
    assert torch.equal(out["scales"], s_rtn), "scale search disagrees"
    assert torch.equal(out["W_hat"], Wh_rtn), "GPTQ with H=I is not RTN"


def test_identity_hessian_is_rtn_various_shapes():
    torch.manual_seed(1)
    C = get_codebook("strict_pot")
    for O, I, g, bs in [(17, 128, 128, 128), (8, 384, 128, 128),
                        (32, 256, 64, 128), (5, 512, 512, 512)]:
        W = torch.randn(O, I) * 0.05
        out = _fit_identity(W, C, group=g, blocksize=bs)
        Wh_rtn, _ = rtn(W, C, group=g)
        assert torch.equal(out["W_hat"], Wh_rtn), (O, I, g, bs)


def test_rescale_in_loop_irrelevant_under_identity_hessian():
    """With H = I nothing is compensated, so the in-loop rescale has nothing to see and
    both settings must give the identical answer. A difference here means the rescale is
    reading stale/garbage state rather than compensated weights."""
    torch.manual_seed(2)
    W = torch.randn(48, 256) * 0.02
    C = get_codebook("strict_pot")
    a = _fit_identity(W, C, rescale_in_loop=True)
    b = _fit_identity(W, C, rescale_in_loop=False)
    assert torch.equal(a["W_hat"], b["W_hat"])


def test_gptq_beats_rtn_on_a_correlated_hessian():
    """On a genuinely correlated Hessian, compensation must reduce the proxy error
    tr(dW H dW^T) -- that is the objective GPTQ minimizes."""
    torch.manual_seed(3)
    O, I = 64, 256
    W = torch.randn(O, I) * 0.02
    A = torch.randn(4096, I) @ (torch.randn(I, I) / I ** 0.5)
    C = get_codebook("strict_pot")
    g = PoTGPTQ(W)
    g.add_batch(A)
    out = g.quantize(C, group=128)
    Wh_rtn, _ = rtn(W, C)
    def proxy(Wh):
        d = Wh - W
        return float(torch.einsum("oi,ij,oj->", d, g.H, d))
    assert proxy(out["W_hat"]) < 0.75 * proxy(Wh_rtn)   # ~1.8x here; ~4x on real acts
    assert out["proxy_err"] == pytest.approx(proxy(out["W_hat"]), rel=1e-4)


def test_act_order_roundtrip_and_verification():
    torch.manual_seed(4)
    W = torch.randn(32, 256) * 0.02
    A = torch.randn(2048, 256) * torch.linspace(0.1, 3.0, 256)
    C = get_codebook("strict_pot")
    g = PoTGPTQ(W)
    g.add_batch(A)
    out = g.quantize(C, group=128, act_order=True)
    ok, d = verify_codebook_artifact(out["W_hat"], out["scales"], C,
                                     group=out["group"], g_idx=out["g_idx"],
                                     col_scale=out["col_scale"])
    assert ok, d["examples"][:3]
    assert out["rel_mse"] < 0.2


def test_output_is_exactly_code_times_scale():
    """The stored artifact must be reproducible from (codes, scales) alone."""
    torch.manual_seed(5)
    W = torch.randn(32, 256) * 0.02
    A = torch.randn(1024, 256)
    C = get_codebook("learned_add2")
    g = PoTGPTQ(W); g.add_batch(A)
    for cs in (None, torch.rand(256) + 0.5):
        out = g.quantize(C, group=128, col_scale=cs)
        S = out["scales"][:, out["g_idx"]]
        if out["col_scale"] is not None:
            S = S * out["col_scale"].view(1, -1)
        assert torch.equal(C.levels[out["codes"].long()] * S, out["W_hat"])
        ok, d = verify_codebook_artifact(out["W_hat"], out["scales"], C,
                                         group=128, g_idx=out["g_idx"],
                                         col_scale=out["col_scale"])
        assert ok, d["examples"][:2]


def test_no_zero_point_is_stored():
    """Symmetric codebooks need no zero-point; a vestigial one would cost 0.0625 bpw."""
    torch.manual_seed(6)
    W = torch.randn(16, 128) * 0.02
    g = PoTGPTQ(W); g.add_batch(torch.randn(512, 128))
    out = g.quantize(get_codebook("strict_pot"), group=128)
    assert "zeros" not in out and "zero_point" not in out


def test_col_scale_equals_transformed_coordinates():
    """The derivation in gptq_pot.py's docstring, checked numerically.

    Folding a per-column scale c into the activation (x' = c*x, W' = W/c, so that
    H' = diag(c) H diag(c)) and running plain GPTQ there is the same algorithm as running
    GPTQ in the ORIGINAL coordinates with c folded into the rounding oracle's per-column
    scale. Reason: Hinv' = diag(c)^-1 Hinv diag(c)^-1, whose upper Cholesky factor is
    U' = U diag(c)^-1, so d'_i = U_ii / c_i and err'_i = (w'_i - q'_i)/d'_i = err_i, and
    the update to every later column j is -err_i * U[i, j] in original coordinates either
    way.

    Two caveats, both real and both checked here:
      * exact only at percdamp=0 -- the damping percdamp * mean(diag H) * I is not
        scale-covariant, since mean(c_i^2 H_ii) != c_i^2 mean(H_ii);
      * the equivalence is in EXACT arithmetic. In float32 the two orderings
        ((w/c)/r vs w/(r*c)) differ by an ulp, and on a log grid whose levels are a
        factor of two apart an ulp at a decision boundary flips a whole exponent, which
        then changes the error fed forward. So individual codes diverge (~3% of them
        here) while the achieved error is the same to ~0.2%. We therefore assert the
        invariant that means something -- the objective GPTQ minimizes -- rather than
        bitwise equality, and we run in original coordinates because that is what makes
        the stored artifact exactly code * (r*c) for the verifier.
    """
    torch.manual_seed(7)
    O, I = 32, 256
    W = torch.randn(O, I) * 0.02
    A = torch.randn(4096, I) @ (torch.randn(I, I) / I ** 0.5)
    c = torch.rand(I) * 1.5 + 0.5
    C = get_codebook("strict_pot")

    g1 = PoTGPTQ(W); g1.add_batch(A)
    a = g1.quantize(C, group=128, col_scale=c, percdamp=0.0)

    g2 = PoTGPTQ(W / c)                      # transformed coordinates
    g2.add_batch(A * c)                      # x' = c * x  =>  H' = diag(c) H diag(c)
    b = g2.quantize(C, group=128, percdamp=0.0)
    assert torch.allclose(g2.H, torch.diag(c) @ g1.H @ torch.diag(c), rtol=1e-4, atol=1e-6)

    def proxy(Wh):
        d = Wh - W
        return float(torch.einsum("oi,ij,oj->", d, g1.H, d))
    pa, pb = proxy(a["W_hat"]), proxy(b["W_hat"] * c)
    assert abs(pa - pb) / pb < 0.02, (pa, pb)


def test_quantize_does_not_mutate_the_accumulated_hessian():
    """Calling quantize() twice on the same object must give identical results; the
    damping and dead-column fixups are in-place and must operate on a copy."""
    torch.manual_seed(8)
    W = torch.randn(32, 256) * 0.02
    g = PoTGPTQ(W); g.add_batch(torch.randn(2048, 256))
    H0 = g.H.clone()
    C = get_codebook("strict_pot")
    a = g.quantize(C, group=128)
    assert torch.equal(g.H, H0), "quantize() mutated the accumulated Hessian"
    b = g.quantize(C, group=128)
    assert torch.equal(a["W_hat"], b["W_hat"])
