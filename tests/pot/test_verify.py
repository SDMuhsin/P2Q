"""The verifier must be a real proof: it has to FAIL on corrupted artifacts."""
import pytest
import torch

from pot.codebooks import get_codebook, nearest_codeword
from pot.scale_search import search_group_scale
from pot.verify_shiftfree import verify_strict_pot, verify_add2, expand_scales


def _artifact(name, shape=(32, 256), group=128, seed=0):
    torch.manual_seed(seed)
    C = get_codebook(name)
    W = torch.randn(*shape) * 0.02
    Wg = W.reshape(-1, group)
    s, _ = search_group_scale(Wg, C)
    W_hat = (C.levels[nearest_codeword(Wg / s, C.levels)] * s).reshape(shape)
    return C, W_hat, s.reshape(shape[0], shape[1] // group)


def test_strict_pot_passes_clean():
    _, W_hat, s = _artifact("strict_pot")
    ok, d = verify_strict_pot(W_hat, s, group=128)
    assert ok and d["n_violations"] == 0 and d["n_weights"] == W_hat.numel()


@pytest.mark.parametrize("delta", [1e-3, 1e-6, 1.0000001, -0.37])
def test_verifier_catches_corruption(delta):
    _, W_hat, s = _artifact("strict_pot")
    W_bad = W_hat.clone()
    W_bad[5, 9] = W_bad[5, 9] * 1.3 + delta
    ok, d = verify_strict_pot(W_bad, s, group=128)
    assert ok is False and d["n_violations"] >= 1
    assert d["examples"][0]["row"] == 5 and d["examples"][0]["col"] == 9


def test_verifier_catches_one_ulp():
    """Even a single-ulp perturbation is not a power of two times the scale."""
    _, W_hat, s = _artifact("strict_pot")
    W_bad = W_hat.clone()
    W_bad[0, 0] = torch.nextafter(W_bad[0, 0], torch.tensor(1e9))
    ok, d = verify_strict_pot(W_bad, s, group=128)
    assert ok is False and d["n_violations"] == 1


def test_verifier_catches_add2_weight_under_strict_rule():
    """An add2 codebook must NOT pass the strict-PoT verifier."""
    _, W_hat, s = _artifact("learned_add2")
    ok, d = verify_strict_pot(W_hat, s, group=128)
    assert ok is False and d["frac_violations"] > 0.5


def test_add2_verifier_accepts_add2_and_strict():
    for name in ("learned_add2", "apot_k2_b4", "strict_pot"):
        _, W_hat, s = _artifact(name)
        ok, d = verify_add2(W_hat, s, group=128)
        assert ok, (name, d["examples"][:2])


def test_add2_verifier_rejects_a_three_shift_value():
    """0.7 = 2^-1+2^-3+2^-5+... is not expressible as two shifts."""
    _, W_hat, s = _artifact("learned_add2")
    W_bad = W_hat.clone()
    W_bad[1, 1] = 0.7 * float(s[1, 0])
    ok, d = verify_add2(W_bad, s, group=128)
    assert ok is False and d["n_violations"] == 1


def test_nonfinite_and_zero_scale():
    _, W_hat, s = _artifact("strict_pot")
    W_bad = W_hat.clone(); W_bad[2, 3] = float("nan")
    assert verify_strict_pot(W_bad, s, group=128)[0] is False
    s_bad = s.clone(); s_bad[0, 0] = 0.0
    ok, d = verify_strict_pot(W_hat, s_bad, group=128)
    assert ok is False and d["n_zero_or_nonfinite_scales"] > 0


def test_expand_scales_with_col_scale_and_gidx():
    torch.manual_seed(0)
    s = torch.rand(4, 2) + 0.5
    c = torch.rand(256) + 0.5
    g_idx = torch.arange(256) // 128
    S = expand_scales(s, 256, 128, col_scale=c, g_idx=g_idx)
    assert S.shape == (4, 256)
    assert torch.allclose(S[:, 200], s[:, 1] * c[200])
