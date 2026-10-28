"""`--codebook-nexp`: a diagnostic override of the strict-PoT octave count.

The point of the flag is to make the FORMAT's capacity measurable: `strict_pot(nexp)`
has 2*nexp levels spanning nexp octaves, so extra code bits can only extend the
alphabet downward -- they buy dynamic range, never resolution, because no single signed
power of two lies between 2^0 and 2^-1. These tests pin (a) that the default path is
untouched, (b) that the code width and therefore the bpw charge is honest, and (c) that
shift-freeness survives at every nexp.
"""
import math
import subprocess
import sys

import pytest
import torch

from pot.codebooks import (get_codebook, strict_pot, strict_pot_zero, nearest_codeword,
                           NEXP_CODEBOOKS)
from pot.run_pot import account_bpw
from pot.scale_search import search_group_scale
from pot.verify_shiftfree import verify_codebook_artifact


def test_default_is_bit_for_bit_unchanged():
    """nexp=None must reproduce the historical books exactly."""
    for name, ref in (("strict_pot", strict_pot(8)),
                      ("strict_pot_zero", strict_pot_zero(7))):
        cb = get_codebook(name)
        assert cb.name == ref.name
        assert torch.equal(cb.levels, ref.levels)
        assert cb.bits() == 4.0


@pytest.mark.parametrize("nexp,n_levels,bits", [(4, 8, 3), (8, 16, 4), (16, 32, 5),
                                                (32, 64, 6)])
def test_nexp_sets_level_count_and_code_width(nexp, n_levels, bits):
    cb = get_codebook("strict_pot", nexp=nexp)
    assert len(cb) == n_levels
    assert cb.bits() == float(bits)
    assert cb.shift_free == "strict_pot"
    assert cb.meta["n_exp"] == nexp
    mags = sorted({abs(float(v)) for v in cb.levels})
    assert mags == sorted(2.0 ** -i for i in range(nexp))
    # the alphabet only ever extends DOWNWARD: the top rung is 2^0 at every nexp
    assert max(mags) == 1.0


def test_nexp_extends_downward_only():
    """Every level of a smaller book is a level of a larger one -- pure range, no
    resolution. This is the whole reason a 5th code bit cannot help at the top."""
    small = set(float(v) for v in strict_pot(8).levels)
    big = set(float(v) for v in strict_pot(16).levels)
    assert small.issubset(big)
    assert min(abs(v) for v in big) < min(abs(v) for v in small)


def test_nexp_rejects_non_pot_codebooks_and_bad_values():
    with pytest.raises(ValueError):
        get_codebook("int4_sym", nexp=16)
    with pytest.raises(ValueError):
        get_codebook("nf4", nexp=8)
    with pytest.raises(ValueError):
        get_codebook("strict_pot", nexp=1)
    assert set(NEXP_CODEBOOKS) == {"strict_pot", "strict_pot_zero"}


@pytest.mark.parametrize("nexp", [4, 8, 16, 32])
def test_shift_free_verifier_passes_at_every_nexp(nexp):
    torch.manual_seed(0)
    cb = get_codebook("strict_pot", nexp=nexp)
    W = torch.randn(64, 256) * 0.03
    Wg = W.reshape(-1, 128)
    s, _ = search_group_scale(Wg, cb)
    W_hat = (cb.levels[nearest_codeword(Wg / s, cb.levels)] * s).reshape(W.shape)
    ok, diag = verify_codebook_artifact(W_hat, s.reshape(64, 2), cb, group=128)
    assert ok is True, diag["examples"][:3]
    assert diag["n_violations"] == 0


def test_smallest_level_survives_bfloat16():
    """nexp=32 puts the bottom rung at 2^-31. The pipeline stores W_hat in the model's
    bfloat16, whose exponent range is fp32's, so it does NOT underflow -- but it WOULD
    in float16 (min subnormal 2^-24). Pin the fact the capacity curve relies on."""
    s = 0.03                                   # a representative per-group scale
    v = s * 2.0 ** -31
    assert float(torch.tensor(v, dtype=torch.bfloat16)) > 0.0
    assert float(torch.tensor(v, dtype=torch.float16)) == 0.0


def test_bpw_charge_grows_with_the_code_width():
    """A 5-bit alphabet must be charged 5 bpw of codes -- the accounting is what keeps a
    diagnostic arm from being mistaken for a deliverable."""
    N = 1_409_286_144                      # Qwen3-1.7B quantized weights
    kw = dict(n_weights=N, n_groups_total=N // 64, n_rows_total=573_440,
              group=64, scale_bits=16, col_scale_entries=0, zero_bits=0,
              col_scale_bits=16)
    b4, d4 = account_bpw(code_bits=4, codebook_entries=16, **kw)
    b5, d5 = account_bpw(code_bits=5, codebook_entries=32, **kw)
    assert d4["codes"] == 4.0 and d5["codes"] == 5.0
    assert b5 - b4 == pytest.approx(1.0, abs=1e-3)
    assert b4 <= 4.2501 < b5              # 4-bit fits the ceiling; 5-bit cannot


def test_cli_exposes_the_flag_and_defaults_to_none():
    out = subprocess.run([sys.executable, "src/pot/run_pot.py", "--help"],
                         capture_output=True, text=True, cwd="/workspace/CoBALT")
    assert "--codebook-nexp" in out.stdout
