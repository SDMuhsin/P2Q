"""Codebook levels are what the spec says, and every shift-free book passes its verifier."""
import math
import pytest
import torch

from pot.codebooks import (get_codebook, strict_pot, strict_pot_zero, apot_k2_b4,
                           learned_add2, fit_add2_codebook, int4_asym, int4_sym, nf4,
                           nearest_codeword, shift_alphabet, CODEBOOKS)
from pot.scale_search import search_group_scale
from pot.verify_shiftfree import verify_codebook_artifact, verify_strict_pot, verify_add2


def test_strict_pot_levels():
    C = strict_pot(8)
    assert len(C) == 16 and C.shift_free == "strict_pot"
    mags = sorted({abs(float(v)) for v in C.levels})
    assert mags == sorted(2.0 ** -i for i in range(8))
    assert 0.0 not in [float(v) for v in C.levels]          # no zero in the strict format
    assert torch.allclose(C.levels, -C.levels.flip(0))      # symmetric -> no zero-point


def test_strict_pot_zero_levels():
    C = strict_pot_zero(7)
    v = [float(x) for x in C.levels]
    assert len(C) == 15 and 0.0 in v
    assert sorted({abs(x) for x in v if x != 0}) == sorted(2.0 ** -i for i in range(7))


def test_apot_k2_b4():
    U = apot_k2_b4(signed=False)
    assert len(U) == 16 and U.shift_free == "add2"
    exp = sorted({a + b for a in (0., 1., 2**-2, 2**-4) for b in (0., 2**-1, 2**-3, 2**-5)})
    assert [float(x) for x in U.levels] == pytest.approx(exp)
    S = apot_k2_b4(signed=True)                              # 8 magnitudes + sign
    assert len({abs(float(x)) for x in S.levels}) == 8
    assert len(S) == 15                                      # +0 == -0


def test_learned_add2_levels():
    C = learned_add2()
    v = [0.875, 0.5625, 0.375, 0.28125, 0.21875, 0.140625, 0.078125, 0.02734375]
    assert sorted({abs(float(x)) for x in C.levels}) == sorted(v)
    # every level really is +-2^a +- 2^b
    A = shift_alphabet(-12, 1)
    for x in C.levels:
        assert float((A - float(x)).abs().min()) == 0.0


def test_reference_books_not_shift_free():
    for f in (int4_asym, int4_sym, nf4):
        assert f().shift_free == "none"
    assert len(int4_asym()) == 16 and len(int4_sym()) == 15 and len(nf4()) == 16


def test_all_sorted_and_unique():
    for name in CODEBOOKS:
        L = get_codebook(name).levels
        assert L.dim() == 1 and bool((L[1:] > L[:-1]).all())


@pytest.mark.parametrize("name", ["strict_pot", "strict_pot_zero", "apot_k2_b4",
                                  "learned_add2"])
def test_shift_free_books_pass_their_verifier(name):
    torch.manual_seed(0)
    C = get_codebook(name)
    W = torch.randn(64, 256) * 0.03
    Wg = W.reshape(-1, 128)
    s, _ = search_group_scale(Wg, C)
    W_hat = (C.levels[nearest_codeword(Wg / s, C.levels)] * s).reshape(W.shape)
    ok, diag = verify_codebook_artifact(W_hat, s.reshape(64, 2), C, group=128)
    assert ok is True, diag["examples"][:3]
    assert diag["n_violations"] == 0


def test_fit_add2_reproduces_a_two_shift_book():
    torch.manual_seed(0)
    x = torch.randn(200000).double()
    x = x / x.abs().max()
    C = fit_add2_codebook(x, n=16)
    assert len(C) == 16 and C.shift_free == "add2"
    A = shift_alphabet(-12, 1)
    for v in C.levels:
        assert float((A - float(v)).abs().min()) == 0.0, f"{float(v)} not two-shift"
    # a fitted book must beat the strict PoT book in MSE on the data it was fitted to
    def mse(cb):
        return float(((cb.levels[nearest_codeword(x.float(), cb.levels)] - x.float()) ** 2).mean())
    assert mse(C) < mse(strict_pot(8))


def test_nearest_codeword_matches_bruteforce():
    torch.manual_seed(0)
    C = get_codebook("learned_add2").levels
    x = torch.randn(5000) * 0.6
    assert torch.equal(nearest_codeword(x, C), (x[:, None] - C).abs().argmin(1))
