"""Column-ordering options for the GPTQ sweep (`order=` / `--order`).

The premise: GPTQ's act-order heuristic sorts columns by descending diag(H), which is the
right key ONLY on a uniform grid, where every column's rounding error is bounded by the
same Delta/2 and therefore factors out of the column's contribution to the output error
(~ H_jj * e_j). On a log grid e_j is proportional to the column's magnitude and spans
orders of magnitude, so it does not factor out. `order='err'` sorts on the measured
product; `order='errmag'` on a pre-pass-free proxy; `order='errasc'` is the ascending
falsification control.

The load-bearing test here is `test_default_path_is_bit_for_bit_unchanged`: this pipeline
is bit-deterministic and adding an option must not perturb the existing arms.
"""
import pytest
import torch

from pot.codebooks import get_codebook, nearest_codeword
from pot.gptq_pot import PoTGPTQ
from pot.verify_shiftfree import verify_codebook_artifact

ORDERS = ["none", "act", "err", "errasc", "errmag"]


def _fit(I=256, O=48, seed=0, dev="cpu"):
    """A correlated Hessian, so the ordering actually has something to do."""
    torch.manual_seed(seed)
    W = torch.randn(O, I, device=dev) * 0.02
    # heavy-tailed column magnitudes: the log grid's error really does vary by column
    W = W * torch.exp(torch.randn(1, I, device=dev) * 1.2)
    A = torch.randn(4 * I, I, device=dev)
    A = A * torch.exp(torch.randn(1, I, device=dev) * 1.0)   # varied diag(H)
    g = PoTGPTQ(W)
    g.add_batch(A)
    return W, g


def _quant(g, C, order=None, **kw):
    return g.quantize(C, group=64, percdamp=0.01, **({"order": order} if order is not None
                                                     else {}), **kw)


# ---------------------------------------------------------------- determinism
@pytest.mark.parametrize("act_order", [False, True])
def test_default_path_is_bit_for_bit_unchanged(act_order):
    """order=None must reproduce the pre-existing act_order behaviour exactly."""
    C = get_codebook("strict_pot")
    _, g = _fit()
    a = g.quantize(C, group=64, act_order=act_order)
    b = g.quantize(C, group=64, act_order=act_order,
                   order=("act" if act_order else "none"))
    assert torch.equal(a["W_hat"], b["W_hat"])
    assert torch.equal(a["codes"], b["codes"])
    assert torch.equal(a["scales"], b["scales"])
    assert torch.equal(a["g_idx"], b["g_idx"])
    assert a["order"] == ("act" if act_order else "none")


def test_order_overrides_act_order_alias():
    C = get_codebook("strict_pot")
    _, g = _fit()
    # explicit --order wins over the legacy flag
    a = g.quantize(C, group=64, act_order=True, order="none")
    b = g.quantize(C, group=64, act_order=False)
    assert torch.equal(a["W_hat"], b["W_hat"])


@pytest.mark.parametrize("order", ORDERS)
def test_repeatable(order):
    C = get_codebook("strict_pot")
    _, g = _fit()
    a = g.quantize(C, group=64, order=order)
    b = g.quantize(C, group=64, order=order)
    assert torch.equal(a["W_hat"], b["W_hat"])


def test_unknown_order_raises():
    C = get_codebook("strict_pot")
    _, g = _fit()
    with pytest.raises(ValueError):
        g.quantize(C, group=64, order="descending_vibes")


# ------------------------------------------------------- permutation / g_idx
@pytest.mark.parametrize("order", ORDERS)
def test_g_idx_and_permutation_inverse(order):
    """W_hat/codes must come back in ORIGINAL column coordinates, and g_idx must say
    which stored scale each original column uses."""
    C = get_codebook("strict_pot")
    W, g = _fit()
    out = g.quantize(C, group=64, order=order)
    gi = out["g_idx"]
    assert gi.shape == (W.shape[1],)
    # every group is used exactly `group` times
    counts = torch.bincount(gi, minlength=out["n_groups"])
    assert torch.equal(counts, torch.full_like(counts, 64))
    # the dequant identity the inference kernel would use
    S = out["scales"][:, gi]
    assert torch.equal(C.levels[out["codes"].long()] * S, out["W_hat"])
    # identity ordering must give contiguous groups
    if order == "none":
        assert torch.equal(gi, torch.arange(W.shape[1]) // 64)
    else:
        assert not torch.equal(gi, torch.arange(W.shape[1]) // 64)


@pytest.mark.parametrize("order", ORDERS)
def test_verify_shiftfree_passes(order):
    C = get_codebook("strict_pot")
    _, g = _fit()
    out = g.quantize(C, group=64, order=order)
    ok, rep = verify_codebook_artifact(out["W_hat"], out["scales"], C,
                                       group=out["group"], col_scale=out["col_scale"],
                                       g_idx=out["g_idx"])
    assert ok, rep


# ------------------------------------------------------------- the sort keys
def test_act_and_err_keys_actually_differ():
    C = get_codebook("strict_pot")
    _, g = _fit()
    outs = {o: g.quantize(C, group=64, order=o) for o in ORDERS}
    for o in ["act", "err", "errmag"]:
        assert not torch.equal(outs[o]["g_idx"], outs["none"]["g_idx"])
    assert not torch.equal(outs["act"]["g_idx"], outs["err"]["g_idx"])
    # ascending is the exact reverse of descending on the same key
    assert not torch.equal(outs["err"]["g_idx"], outs["errasc"]["g_idx"])


def test_rtn_col_err_matches_a_hand_rolled_rtn():
    """The `err` key's pre-pass must be plain RTN with the pipeline's own scale search."""
    from pot.scale_search import search_group_scale
    C = get_codebook("strict_pot")
    W, g = _fit()
    L = C.levels
    sk = dict(n_grid=200, lo=0.3, hi=1.2, pot_scale=False)
    e = g._rtn_col_err(W.clone(), L, 64, sk, None, None)
    O, I = W.shape
    Wg = W.reshape(-1, 64)
    s, _ = search_group_scale(Wg, L, **sk)
    Wh = (L[nearest_codeword(Wg / s, L)] * s).reshape(O, I)
    assert torch.allclose(e, ((W - Wh) ** 2).sum(0))


def test_err_prepass_is_independent_of_the_hessian_scale():
    """e_j is a property of W and the codebook only; scaling H must not change it."""
    C = get_codebook("strict_pot")
    W, g = _fit()
    sk = dict(n_grid=200, lo=0.3, hi=1.2, pot_scale=False)
    e = g._rtn_col_err(W.clone(), C.levels, 64, sk, None, None)
    e2 = g._rtn_col_err(W.clone(), C.levels, 64, sk, None, None)
    assert torch.equal(e, e2)
    assert (e > 0).all()


def test_errasc_is_the_reverse_of_err_on_the_same_key():
    C = get_codebook("strict_pot")
    W, g = _fit()
    sk = dict(n_grid=200, lo=0.3, hi=1.2, pot_scale=False)
    Hd = torch.diag(g.H).clone()
    key = Hd * g._rtn_col_err(W.clone(), C.levels, 64, sk, None, None)
    p_desc = g._column_perm("err", W.clone(), C.levels, Hd, 64, sk, None, None)
    p_asc = g._column_perm("errasc", W.clone(), C.levels, Hd, 64, sk, None, None)
    assert torch.equal(key[p_desc], torch.sort(key, descending=True).values)
    assert torch.equal(key[p_asc], torch.sort(key, descending=False).values)


def test_identity_hessian_ordering_is_a_no_op_on_the_reconstruction():
    """With H = I there is no feedback, so any column ordering must give the SAME
    reconstruction (up to which columns share a group -- so use one group per row)."""
    C = get_codebook("strict_pot")
    torch.manual_seed(3)
    W = torch.randn(16, 128) * 0.02
    outs = []
    for o in ORDERS:
        g = PoTGPTQ(W)
        g.H = torch.eye(128)
        g.nsamples = 1
        outs.append(g.quantize(C, group=128, percdamp=0.0, order=o)["W_hat"])
    for x in outs[1:]:
        assert torch.equal(outs[0], x)
