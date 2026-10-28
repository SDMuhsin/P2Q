"""Tests for `--scale-offset-log2` / `--scale-round-log2` (PoTGPTQ._post_scale).

The load-bearing property is DEFAULT INERTNESS: the incumbent path must be
bit-identical with the options absent and with their default values supplied, because
every historical number in results/pot/pot_runs.jsonl was produced without them.
"""
import sys, os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pot.gptq_pot import PoTGPTQ
from pot.codebooks import get_codebook


def _s():
    torch.manual_seed(0)
    return torch.rand(32, 1) * 3 + 0.1


def test_defaults_are_bit_identical():
    s = _s()
    assert torch.equal(PoTGPTQ._post_scale(s), s)
    assert torch.equal(PoTGPTQ._post_scale(s, 0.0, None), s)
    assert torch.equal(PoTGPTQ._post_scale(s, 0.0, 0.0), s)      # Delta=0 is falsy => off


def test_offset_is_exact_doubling():
    s = _s()
    assert torch.allclose(PoTGPTQ._post_scale(s, 1.0), s * 2, rtol=0, atol=0)
    assert torch.allclose(PoTGPTQ._post_scale(s, -1.0), s / 2, rtol=0, atol=0)
    # offsets compose additively in log space
    a = PoTGPTQ._post_scale(PoTGPTQ._post_scale(s, 0.25), 0.75)
    assert torch.allclose(a, PoTGPTQ._post_scale(s, 1.0), rtol=1e-6)


def test_round_lands_on_the_grid_and_is_bounded():
    s = _s()
    for D in (0.01, 0.1, 0.5, 1.0):
        r = PoTGPTQ._post_scale(s, 0.0, D)
        k = torch.log2(r) / D
        assert torch.allclose(k, k.round(), atol=1e-5), f"off-grid at Delta={D}"
        # the perturbation is at most half a step, by construction
        assert (torch.log2(r) - torch.log2(s)).abs().max() <= D / 2 + 1e-5


def test_round_delta_one_gives_a_power_of_two_scale():
    r = PoTGPTQ._post_scale(_s(), 0.0, 1.0)
    e = torch.log2(r)
    assert torch.allclose(e, e.round(), atol=1e-5)


def test_offset_applied_before_round():
    """Ordering matters and is fixed: offset shifts the window, THEN storage rounds."""
    s = _s()
    got = PoTGPTQ._post_scale(s, 0.3, 0.25)
    want = PoTGPTQ._post_scale(PoTGPTQ._post_scale(s, 0.3), 0.0, 0.25)
    assert torch.equal(got, want)


@pytest.mark.parametrize("exact", [False, True])
def test_group_scale_path_is_inert_by_default(exact):
    """The option must be inert through `_group_scale`, not merely in `_post_scale`."""
    torch.manual_seed(1)
    W = torch.randn(8, 64)
    C = get_codebook("strict_pot")
    kw = dict(pot_scale=False)
    kw.update(dict(exact_scale=True) if exact else dict(n_grid=200, lo=0.3, hi=1.2))
    obj = PoTGPTQ.__new__(PoTGPTQ)
    base = PoTGPTQ._group_scale(obj, W, C, None, **dict(kw))
    withopt = PoTGPTQ._group_scale(obj, W, C, None,
                                   **dict(kw, scale_offset_log2=0.0,
                                          scale_round_log2=None))
    assert torch.equal(base, withopt)
    moved = PoTGPTQ._group_scale(obj, W, C, None,
                                 **dict(kw, scale_offset_log2=1.0,
                                        scale_round_log2=None))
    assert torch.allclose(moved, base * 2)


def test_fp16_rounding_is_idempotent_and_bounded():
    """`--scale-fp16` makes the stored scale match what account_bpw charges for it."""
    s = _s()
    r = PoTGPTQ._post_scale(s, 0.0, None, True)
    assert torch.equal(r, r.half().float())                       # on the fp16 lattice
    assert torch.equal(r, PoTGPTQ._post_scale(r, 0.0, None, True))    # idempotent
    # fp16 has 10 explicit mantissa bits, so the relative move is <= 2^-11
    assert ((r - s).abs() / s).max() <= 2 ** -11 + 1e-9
    # and in log2 terms that is ~0.0007 octaves, far finer than any --scale-round arm
    assert (torch.log2(r) - torch.log2(s)).abs().max() < 1e-3


def test_fp16_defaults_off():
    s = _s()
    assert torch.equal(PoTGPTQ._post_scale(s), s)
    assert torch.equal(PoTGPTQ._post_scale(s, 0.0, None, False), s)
