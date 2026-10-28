import pytest
import torch

from pot.codebooks import get_codebook, nearest_codeword
from pot.scale_search import (search_group_scale, search_col_scale, pot_round,
                              group_quant_error)


@pytest.mark.parametrize("name", ["strict_pot", "learned_add2", "int4_asym", "nf4"])
def test_never_worse_than_absmax(name):
    torch.manual_seed(0)
    C = get_codebook(name)
    Wg = torch.randn(500, 128) * 0.02
    Wg[3] *= 20                                            # a heavy-tailed group
    Wg[7] = 0.0                                            # a dead group
    s, e = search_group_scale(Wg, C)
    base = (Wg.abs().amax(1, keepdim=True) / C.absmax).clamp_min(1e-12)
    e0 = group_quant_error(Wg, C.levels, base)
    assert bool((e <= e0 + 1e-12).all()), float((e - e0).max())
    assert torch.isfinite(s).all() and bool((s > 0).all())


def test_hessian_weighting_lowers_weighted_error():
    torch.manual_seed(1)
    C = get_codebook("strict_pot")
    Wg = torch.randn(200, 128) * 0.02
    h = torch.rand(128) * 10
    s_plain, _ = search_group_scale(Wg, C)
    s_w, e_w = search_group_scale(Wg, C, hess_diag=h)
    assert bool((e_w <= group_quant_error(Wg, C.levels, s_plain, h) + 1e-9).all())


def test_pot_round():
    s = torch.tensor([1.0, 0.7, 0.26, 3.9, 1e-8])
    r = pot_round(s)
    assert torch.allclose(r, torch.tensor([1.0, 0.5, 0.25, 4.0, 2.0 ** -27]))
    assert torch.all(torch.frexp(r).mantissa == 0.5)


def test_pot_scale_search_returns_powers_of_two():
    torch.manual_seed(2)
    C = get_codebook("strict_pot")
    s, _ = search_group_scale(torch.randn(64, 128) * 0.02, C, pot_scale=True)
    assert torch.all(torch.frexp(s).mantissa == 0.5)


def test_col_scale_monotone_and_not_worse():
    torch.manual_seed(3)
    C = get_codebook("strict_pot")
    W = torch.randn(64, 256) * 0.02 * torch.linspace(0.2, 5.0, 256)
    r0, c0, rel0 = search_col_scale(W, C, iters=0)          # absmax, c == 1
    r1, c1, rel1 = search_col_scale(W, C, iters=4)
    assert rel1 <= rel0
    assert c1.shape == (256,) and bool((c1 > 0).all())
