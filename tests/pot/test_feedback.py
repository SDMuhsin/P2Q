"""The feedback controls must be inert at their defaults, and must reach RTN at lambda=0.

Motivation for the feature itself: the measured 2x2 at identical bpw (4.246867) is

                strict_pot            int4_sym
    GPTQ        0.6012 (9 draws)      0.6603 (5 draws)
    RTN         0.6233 (exact)        0.6481 (exact)

i.e. GPTQ's error feedback is worth +1.21 pp on a uniform grid and -2.21 pp on a log grid.
`feedback_damp` / `feedback_clip` expose that as a knob. These tests guard the only thing
that could silently invalidate every earlier run: that leaving them alone changes nothing.
"""
import sys, os, torch, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pot.gptq_pot import PoTGPTQ
from pot.codebooks import strict_pot, int4_sym


def _mk(seed=0, rows=64, cols=128, n=8):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(cols, rows, bias=False)
    g = PoTGPTQ(lin)
    for _ in range(n):
        g.add_batch(torch.randn(16, cols))
    return g


@pytest.mark.parametrize("cb", [strict_pot(8), int4_sym()])
@pytest.mark.parametrize("kw", [dict(), dict(act_order=True), dict(group=32),
                                dict(rescale_in_loop=False)])
def test_defaults_are_bit_identical(cb, kw):
    """damp=1.0 / clip=None must reproduce the stock sweep exactly, bit for bit."""
    a = _mk().quantize(cb, **kw)
    b = _mk().quantize(cb, feedback_damp=1.0, feedback_clip=None, **kw)
    assert torch.equal(a["W_hat"], b["W_hat"])
    assert torch.equal(a["codes"], b["codes"])
    assert torch.equal(a["scales"], b["scales"])


def test_damp_zero_kills_all_propagation():
    """lambda=0 must make every column independent of every other: identical to
    quantizing with a Hessian that carries no cross terms."""
    cb = strict_pot(8)
    g = _mk()
    out = g.quantize(cb, feedback_damp=0.0)
    # with no propagation the stored codes must round the ORIGINAL weights
    from pot.codebooks import nearest_codeword
    S = out["scales"][:, out["g_idx"]]
    expect = nearest_codeword(g.W / S, cb.levels.to(g.W.device, torch.float32))
    assert torch.equal(out["codes"], expect.to(torch.uint8))


def test_damp_is_monotone_in_weight_error():
    """Less propagation must mean the artifact stays closer to the original weights:
    GPTQ trades weight error for output error, so damping it recovers weight error."""
    cb = strict_pot(8)
    errs = []
    for lam in (0.0, 0.5, 1.0):
        out = _mk().quantize(cb, feedback_damp=lam)
        g = _mk()
        errs.append(((out["W_hat"] - g.W) ** 2).sum().item())
    assert errs[0] < errs[1] < errs[2], errs


def test_clip_is_inert_when_huge_and_biting_when_tiny():
    cb = strict_pot(8)
    base = _mk().quantize(cb)
    loose = _mk().quantize(cb, feedback_clip=1e9)
    tight = _mk().quantize(cb, feedback_clip=0.0)
    assert torch.equal(base["W_hat"], loose["W_hat"])          # cap never binds
    zero = _mk().quantize(cb, feedback_damp=0.0)
    assert torch.equal(tight["W_hat"], zero["W_hat"])          # cap always binds


def test_clip_lands_between_the_endpoints():
    cb = strict_pot(8)
    g0 = _mk()
    def werr(**kw):
        return ((_mk().quantize(cb, **kw)["W_hat"] - g0.W) ** 2).sum().item()
    lo, mid, hi = werr(feedback_damp=0.0), werr(feedback_clip=0.5), werr()
    assert lo <= mid <= hi, (lo, mid, hi)
