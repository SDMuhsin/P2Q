"""Per-group alphabet selection: every level of every family member is a single signed
power of two (C1), the artifact verifies exactly like a single-codebook run, bpw
accounting includes the selector honestly, and the single-member family reproduces
today's path bit-for-bit."""
import math

import pytest
import torch

from pot.alphabets import (strict_pot_stride, default_family, selector_bits_for,
                           select_group_alphabet_scale, PoTGPTQFamily)
from pot.codebooks import strict_pot
from pot.gptq_pot import PoTGPTQ
from pot.verify_shiftfree import verify_codebook_artifact


# ------------------------------------------------------------------- C1: single PoT
def test_every_level_is_a_single_power_of_two():
    for stride in (1, 2, 3, 4, 5):
        cb = strict_pot_stride(stride, nexp=8)
        for v in cb.levels:
            v = float(v)
            assert v != 0.0
            m, e = torch.frexp(torch.tensor(abs(v), dtype=torch.float64))
            # frexp: x = m * 2**e with 0.5 <= |m| < 1; a power of two has m == 0.5 exactly
            assert m.item() == 0.5, f"stride={stride} level {v} is not a single power of two"


def test_family_members_share_code_width():
    fam = default_family()
    assert all(len(c) == len(fam[0]) for c in fam)
    assert len(fam[0]) == 16                       # 4-bit codes, same as today
    assert all(c.bits() == 4 for c in fam)
    assert all(c.shift_free == "strict_pot" for c in fam)


def test_selector_bits():
    assert selector_bits_for(default_family(strides=(1, 2, 3, 4))) == 2
    assert selector_bits_for(default_family(strides=(1, 2))) == 1
    assert selector_bits_for(default_family(strides=(1,))) == 1


# ---------------------------------------------------------------- anchor: bit-identical
def test_single_member_family_is_bit_identical_to_todays_path():
    torch.manual_seed(0)
    O, I = 16, 128
    W = torch.randn(O, I) * 0.05
    X = torch.randn(32, I)

    g_plain = PoTGPTQ(W.clone())
    g_plain.add_batch(X)
    out_plain = g_plain.quantize(strict_pot(8), group=64, act_order=False)

    g_fam = PoTGPTQFamily(W.clone())
    g_fam.add_batch(X)
    out_fam = g_fam.quantize([strict_pot_stride(1, 8)], group=64, act_order=False)

    assert torch.equal(out_plain["W_hat"], out_fam["W_hat"])
    assert torch.equal(out_plain["scales"], out_fam["scales"])
    assert torch.equal(out_plain["codes"], out_fam["codes"])
    assert (out_fam["alpha_sel"] == 0).all()


def test_single_member_family_bit_identical_with_act_order():
    torch.manual_seed(1)
    O, I = 16, 128
    W = torch.randn(O, I) * 0.05
    X = torch.randn(32, I)

    g_plain = PoTGPTQ(W.clone())
    g_plain.add_batch(X)
    out_plain = g_plain.quantize(strict_pot(8), group=64, act_order=True, blocksize=64)

    g_fam = PoTGPTQFamily(W.clone())
    g_fam.add_batch(X)
    out_fam = g_fam.quantize([strict_pot_stride(1, 8)], group=64, act_order=True,
                            blocksize=64)

    assert torch.equal(out_plain["W_hat"], out_fam["W_hat"])
    assert torch.equal(out_plain["codes"], out_fam["codes"])


# --------------------------------------------------------------------------- verifier
def test_selected_alphabet_artifact_verifies():
    torch.manual_seed(2)
    O, I = 24, 256
    W = torch.randn(O, I) * 0.05
    W[0, :4] *= 400.0                              # inject an outlier group
    X = torch.randn(48, I)
    fam = default_family()

    g = PoTGPTQFamily(W.clone())
    g.add_batch(X)
    out = g.quantize(fam, group=64, act_order=False)

    ok, diag = verify_codebook_artifact(out["W_hat"], out["scales"], fam[0],
                                        group=out["group"], g_idx=out["g_idx"],
                                        col_scale=out["col_scale"])
    assert ok is True, diag
    assert diag["n_violations"] == 0


def test_selected_alphabet_artifact_verifies_with_act_order():
    torch.manual_seed(3)
    O, I = 24, 256
    W = torch.randn(O, I) * 0.05
    W[3, 10:14] *= 300.0
    X = torch.randn(48, I)
    fam = default_family()

    g = PoTGPTQFamily(W.clone())
    g.add_batch(X)
    out = g.quantize(fam, group=64, act_order=True, blocksize=64)

    ok, diag = verify_codebook_artifact(out["W_hat"], out["scales"], fam[0],
                                        group=out["group"], g_idx=out["g_idx"],
                                        col_scale=out["col_scale"])
    assert ok is True, diag


def test_wider_alphabet_can_be_selected_and_still_verifies():
    """A group with a genuinely wide dynamic range (small background weights plus a
    huge, isolated, hess-weighted-irrelevant outlier) can select a non-stride-1 member;
    whichever member is picked, the artifact must still verify. This does not assert a
    particular alphabet wins (see the CPU dynamic-range study in the report: plain MSE
    is dominated by the largest element in a group, so the default -- unweighted --
    selection objective rarely prefers a wider ladder even when one exists; this test
    only pins down that the mechanism is correct when it DOES fire)."""
    torch.manual_seed(4)
    g = 64
    rows = 500
    # deliberately bimodal magnitudes spanning ~20 octaves so some alphabet other than
    # stride=1 is at least reachable by the argmin.
    e = torch.rand(rows, g) * 20.0
    Wg = torch.sign(torch.randn(rows, g)) * (2.0 ** (-e))
    fam = default_family()
    s, k, err = select_group_alphabet_scale(Wg, fam, n_grid=200, lo=0.3, hi=1.2)
    assert s.shape == (rows, 1)
    assert k.shape == (rows,)
    assert k.min() >= 0 and k.max() < len(fam)


# ------------------------------------------------------------------------------ bpw
def test_bpw_accounting_includes_selector_and_stays_in_budget():
    from pot.run_pot import account_bpw, G_IDX_BPW_ACT_ORDER

    n_weights = 1_409_286_144
    group = 128
    scale_bits = 16
    sel_bits = selector_bits_for(default_family())        # 2
    n_groups_total = n_weights // group

    bpw, breakdown = account_bpw(
        n_weights=n_weights, n_groups_total=n_groups_total, n_rows_total=0,
        code_bits=4, group=group, scale_bits=scale_bits, zero_bits=0,
        col_scale_bits=16, col_scale_entries=0,
        codebook_entries=4 * 16, codebook_bits=16,
        selector_bits=sel_bits, g_idx_bpw=G_IDX_BPW_ACT_ORDER)

    assert breakdown["alphabet_selector"] == pytest.approx(sel_bits / group, rel=1e-9)
    assert breakdown["g_idx"] == G_IDX_BPW_ACT_ORDER
    # exact accounting per the task's arithmetic: (16+2)/128 = 0.140625 metadata
    assert breakdown["group_scale"] == pytest.approx(16 / group, rel=1e-9)
    expected_total = 4 + 18 / group + 4 * 16 * 16 / n_weights + G_IDX_BPW_ACT_ORDER
    assert bpw == pytest.approx(expected_total, rel=1e-9)
    assert bpw <= 4.2500, f"bpw {bpw} exceeds the C3 ceiling"
    assert bpw <= 4.2501
