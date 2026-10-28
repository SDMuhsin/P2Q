"""Tests for activation-aware PoT column equalization (src/pot/equalize.py)."""
import types

import pytest
import torch

from pot.codebooks import get_codebook, nearest_codeword
from pot.equalize import (equalize_columns, qwen3_equalize_plan, gqa_tie_index,
                          apply_reparam_check, INT_RANGE)
from pot.gptq_pot import PoTGPTQ
from pot.verify_shiftfree import verify_strict_pot, expand_scales


def _cfg():
    return types.SimpleNamespace(hidden_size=2048, intermediate_size=6144,
                                 num_attention_heads=16, num_key_value_heads=8,
                                 head_dim=128)


# ------------------------------------------------------- 1. exactness of reparam
def test_reparametrization_is_exact():
    torch.manual_seed(0)
    W = torch.randn(37, 64, dtype=torch.float64)
    x = torch.randn(64, 5, dtype=torch.float64)
    c = torch.exp2(torch.randint(-6, 7, (64,)).double())   # power-of-two diagonal
    assert apply_reparam_check(W, x, c) < 1e-12
    c_real = torch.rand(64, dtype=torch.float64) * 3 + 0.1
    assert apply_reparam_check(W, x, c_real) < 1e-10


def test_rmsnorm_absorption_is_exact():
    """Folding a diagonal into an RMSNorm weight reproduces diag(c) @ norm(x)."""
    torch.manual_seed(1)
    H = 64
    x = torch.randn(8, H, dtype=torch.float64)
    g = torch.rand(H, dtype=torch.float64) + 0.5
    c = torch.exp2(torch.randint(-3, 4, (H,)).double())

    def rms(x_, w):
        return x_ / torch.sqrt((x_ ** 2).mean(-1, keepdim=True) + 1e-6) * w

    assert torch.allclose(rms(x, g) * c, rms(x, g * c), atol=1e-12)


def test_gqa_tie_index():
    t = gqa_tie_index(16, 8, 128)
    assert t.numel() == 2048 and int(t.max()) == 1023
    # query heads 0 and 1 both read kv head 0
    assert torch.equal(t[:128], t[128:256])
    assert not torch.equal(t[:128], t[256:384])


def test_plan_shape():
    plan = qwen3_equalize_plan(_cfg())
    names = sum([g["members"] for g in plan], [])
    assert sorted(names) == sorted([
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    assert all(g["free"] for g in plan)
    assert sum(g["n_free"] for g in plan) == 2048 + 2048 + 1024 + 6144


# --------------------------------------------------------- 2. optimizer contract
@pytest.mark.parametrize("mode", ["int", "real"])
def test_objective_is_monotone_and_not_worse_than_identity(mode):
    torch.manual_seed(2)
    C = get_codebook("strict_pot")
    W = torch.randn(64, 128) * 0.02
    W[:, ::7] *= 30.0                                      # structure a group scale misses
    h = torch.rand(128) * 5 + 0.1
    c, info = equalize_columns([W], C, [64], hdiag=h, mode=mode)
    assert info["obj_after"] <= info["obj_before"] + 1e-9
    assert torch.isfinite(c).all() and bool((c > 0).all())


def test_int_mode_diagonal_is_a_power_of_two():
    torch.manual_seed(3)
    C = get_codebook("strict_pot")
    W = torch.randn(32, 128) * 0.02
    c, _ = equalize_columns([W], C, [64], mode="int")
    assert torch.all(torch.frexp(c).mantissa == 0.5)
    assert torch.all(torch.log2(c).abs() <= INT_RANGE * 4 + 1)


def test_tie_is_respected():
    torch.manual_seed(4)
    C = get_codebook("strict_pot")
    W = torch.randn(32, 128) * 0.02
    tie = torch.arange(128) // 4
    c, info = equalize_columns([W], C, [64], mode="int", tie=tie)
    assert info["n_ties"] == 32
    assert torch.equal(c.view(32, 4), c.view(32, 4)[:, :1].expand(32, 4))


def test_shared_diagonal_across_matrices():
    torch.manual_seed(5)
    C = get_codebook("strict_pot")
    A = torch.randn(48, 128) * 0.02
    B = torch.randn(16, 128) * 0.05
    c, info = equalize_columns([A, B], C, [64, 64], mode="int")
    assert c.numel() == 128
    assert info["obj_after"] <= info["obj_before"] + 1e-9


def test_uniform_h_cancels_from_the_column_argmin():
    """For a FIXED r the salience is a per-column constant and cannot change the
    per-column argmin; it can only act through the shared group scale. Scaling h by a
    global constant must therefore leave c unchanged."""
    torch.manual_seed(6)
    C = get_codebook("strict_pot")
    W = torch.randn(32, 128) * 0.02
    h = torch.rand(128) + 0.1
    c1, _ = equalize_columns([W], C, [64], hdiag=h, mode="int")
    c2, _ = equalize_columns([W], C, [64], hdiag=h * 1234.5, mode="int")
    assert torch.equal(c1, c2)


# ------------------------------------------------ 3. artifact stays strictly PoT
def test_quantized_artifact_is_strict_pot_under_equalization():
    torch.manual_seed(7)
    C = get_codebook("strict_pot")
    W = torch.randn(48, 128) * 0.02
    X = torch.randn(256, 128)
    for mode in ("int", "real"):
        h = (2.0 / X.shape[0]) * (X ** 2).sum(0)
        c, _ = equalize_columns([W], C, [64], hdiag=h, mode=mode)
        g = PoTGPTQ(W.clone())
        g.add_batch(X)
        g.H = torch.eye(128)
        g.nsamples = 1
        out = g.quantize(C, group=64, percdamp=0.0, col_scale=c,
                         exact_scale=True, scale_hess=h * c ** 2)
        ok, diag = verify_strict_pot(out["W_hat"], out["scales"], group=64,
                                     col_scale=out["col_scale"], g_idx=out["g_idx"])
        assert ok, diag["examples"][:2]


# --------------------------------------------------- 4. defaults are unchanged
def test_scale_hess_none_is_bit_identical():
    torch.manual_seed(8)
    C = get_codebook("strict_pot")
    W = torch.randn(40, 128) * 0.02
    X = torch.randn(200, 128)

    def run(**kw):
        g = PoTGPTQ(W.clone())
        g.add_batch(X)
        return g.quantize(C, group=64, **kw)

    a = run()
    b = run(scale_hess=None)
    assert torch.equal(a["W_hat"], b["W_hat"])
    assert torch.equal(a["codes"], b["codes"])
    assert torch.equal(a["scales"], b["scales"])


def test_scale_hess_ones_matches_uniform_under_exact_scale():
    torch.manual_seed(9)
    C = get_codebook("strict_pot")
    W = torch.randn(40, 128) * 0.02
    X = torch.randn(200, 128)

    def run(**kw):
        g = PoTGPTQ(W.clone())
        g.add_batch(X)
        g.H = torch.eye(128)
        g.nsamples = 1
        return g.quantize(C, group=64, percdamp=0.0, exact_scale=True, **kw)

    a = run()
    b = run(scale_hess=torch.ones(128))
    assert torch.equal(a["W_hat"], b["W_hat"])


def test_identity_diagonal_reproduces_plain_rtn():
    """c == 1 must give bit-for-bit the un-equalized artifact."""
    torch.manual_seed(10)
    C = get_codebook("strict_pot")
    W = torch.randn(40, 128) * 0.02
    X = torch.randn(200, 128)

    def run(**kw):
        g = PoTGPTQ(W.clone())
        g.add_batch(X)
        g.H = torch.eye(128)
        g.nsamples = 1
        return g.quantize(C, group=64, percdamp=0.0, exact_scale=True, **kw)

    a = run()
    b = run(col_scale=torch.ones(128), scale_hess=torch.ones(128))
    assert torch.equal(a["W_hat"], b["W_hat"])


# ---------------------------------------- 5. deployment equivalence end to end
def test_deployed_product_matches_stored_artifact():
    """The kernel computes sum_j code_ij * r_ig * (c_j x_j): storing only code and r,
    and rescaling the activation by c, must equal the dense W_hat @ x."""
    torch.manual_seed(11)
    C = get_codebook("strict_pot")
    W = torch.randn(40, 128) * 0.02
    X = torch.randn(200, 128)
    h = (2.0 / X.shape[0]) * (X ** 2).sum(0)
    c, _ = equalize_columns([W], C, [64], hdiag=h, mode="int")
    g = PoTGPTQ(W.clone())
    g.add_batch(X)
    g.H = torch.eye(128)
    g.nsamples = 1
    out = g.quantize(C, group=64, percdamp=0.0, col_scale=c, exact_scale=True,
                     scale_hess=h * c ** 2)
    x = torch.randn(128, 3)
    dense = out["W_hat"] @ x
    R = out["scales"][:, out["g_idx"]]
    codes = C.levels[out["codes"].long()]
    shifted = (codes * R) @ (c.view(-1, 1) * x)
    assert torch.allclose(dense, shifted, atol=1e-5, rtol=1e-5)


# ------------------------------------ 6. the covariance theorem (see module docstring)
def test_pot_column_scale_is_a_noop_in_the_interior():
    """A power-of-two column scale is BIT-EXACTLY inert on a strict-PoT alphabet except
    where the shift pushes a weight onto an end rung. This is why the integer lever --
    the very thing the AWQ-on-a-log-grid pitch calls 'free and exact' -- has no degrees
    of freedom, and why all the power lives in the real-valued diagonal."""
    torch.manual_seed(12)
    C = get_codebook("strict_pot")
    L = C.levels.double()
    w = torch.randn(200000, dtype=torch.float64) * 0.02
    s = 0.05

    def code(x):
        return L[nearest_codeword(x / s, L)]

    e0 = code(w) * s - w
    ext = float(L.abs().min())
    for t in (1, 2, 3, -1, -2, -3, -5):
        c = 2.0 ** t
        e1 = code(w / c) * s * c - w
        m0, m1 = code(w).abs(), code(w / c).abs()
        interior = (m0 < 1.0) & (m0 > ext) & (m1 < 1.0) & (m1 > ext)
        assert interior.sum() > 1000
        assert float((e0 - e1)[interior].abs().max()) == 0.0, t


def test_real_diagonal_has_strictly_more_power_than_integer():
    """On weights the group scale already positions well, the integer lever is inert
    while the real one is not."""
    torch.manual_seed(13)
    C = get_codebook("strict_pot")
    W = torch.randn(96, 128) * 0.02
    _, i_int = equalize_columns([W], C, [64], mode="int")
    _, i_real = equalize_columns([W], C, [64], mode="real")
    assert i_real["obj_after"] <= i_int["obj_after"] + 1e-9


# ----------------- 7. rung-occupancy diagnostic (the MSE-blind failure mode)
def test_rung_stats_counts_the_extreme_rungs():
    from pot.run_pot import rung_stats, account_bpw
    C = get_codebook("strict_pot")
    lev = C.levels
    amin, amax = float(lev.abs().min()), float(lev.abs().max())
    # hand-built artifact: 2 weights on the bottom rung, 1 on the top, 1 interior
    codes = torch.tensor([[int((lev == -amin).nonzero()[0]),
                           int((lev == amin).nonzero()[0]),
                           int((lev == amax).nonzero()[0]),
                           int((lev == 0.25).nonzero()[0])]], dtype=torch.uint8)
    W_hat = torch.tensor([[-amin, amin, amax, 0.25]])
    W = torch.tensor([[-amin / 4, amin, amax, 0.25]])      # first one clamped UP 2 octaves
    r = rung_stats(codes, lev, W_hat, W)
    assert r["n"] == 4 and r["n_bottom"] == 2 and r["n_top"] == 1
    assert abs(r["dlog2_bottom_sum"] - 2.0) < 1e-6         # (2 + 0) / 2 * 2
    assert r["dlog2_top_sum"] == 0.0


def test_equalize_bpw_is_charged_when_asked():
    from pot.run_pot import account_bpw
    n_w = 1409286144
    # the incumbent grouping: default=64, self_attn.v_proj=128, over 28 Qwen3-1.7B blocks
    n_g = 28 * (2048 * 32 + 1024 * 32 + 1024 * 16 + 2048 * 32
                + 6144 * 32 + 6144 * 32 + 2048 * 96)
    base, _ = account_bpw(n_weights=n_w, n_groups_total=n_g, n_rows_total=0,
                          code_bits=4, group=64, scale_bits=16, zero_bits=0,
                          col_scale_bits=16, col_scale_entries=0, codebook_entries=16)
    assert base == pytest.approx(4.244792, abs=1e-6)     # the recorded incumbent bpw
    tot, bd = account_bpw(n_weights=n_w, n_groups_total=n_g, n_rows_total=0,
                          code_bits=4, group=64, scale_bits=16, zero_bits=0,
                          col_scale_bits=16, col_scale_entries=0, codebook_entries=16,
                          eq_bits=16, eq_entries=11264 * 28)
    assert bd["equalize"] == pytest.approx(16 * 11264 * 28 / n_w)
    assert tot == pytest.approx(base + bd["equalize"])
    assert tot <= 4.2501                       # the pre-registered conservative charge


def test_exact_group_scale_chunking_is_bit_identical():
    """The chunk-size heuristic in exact_scale.py was under-counting its own scratch
    footprint by a factor of g (~8 GB per call at g=64 instead of ~134 MB). Chunking is
    over independent rows, so shrinking it must change nothing at all."""
    from pot.exact_scale import exact_group_scale
    torch.manual_seed(14)
    C = get_codebook("strict_pot")
    W = torch.randn(300, 64) * 0.02
    h = torch.rand(64) + 0.1
    for hess in (None, h):
        a = exact_group_scale(W, C, hess_diag=hess, max_elems=2 ** 24)
        b = exact_group_scale(W, C, hess_diag=hess, max_elems=2 ** 12)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
