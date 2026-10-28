"""
Tests for the higher-power ARC-Easy readouts (`src/pot/arc_stats.py`) and for the
per-item score dump they are computed from (`pot.eval_arc.dump_per_item`).

Two things must hold, and the first is the important one:

  1. THE DUMP IS STRICTLY ADDITIVE. `eval_arc_easy(..., dump_path=...)` must return a
     dict bit-identical to `eval_arc_easy(...)` without it, and the dump must be a
     faithful enough record that `acc`, `acc_norm` and `acc_norm_token` can be
     recomputed from it exactly. `acc_norm` is the CHARTER acceptance metric; nothing
     in this instrument work is allowed to perturb it.

  2. The statistics behave: they are per-item, orientable, they agree with `acc` on the
     argmax, and the draw-count arithmetic is right.

CPU only, no GPU, no network for the synthetic tests. The end-to-end test builds a tiny
randomly-initialised model and needs the cached tokenizer + ARC snapshot, in the same
style as tests/pot/test_calib_arce.py.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pytest
import torch

import pot.arc_stats as S           # noqa: E402
import pot.eval_arc as EA           # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"


# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------

def _synth_per_doc(rng, n=64):
    """
    Build a per_doc table with exactly the layout eval_arc_easy produces:
    (raw_ll, raw_ll/nchar, raw_ll/ntok, nchar, ntok), ragged over 3/4/5 choices.
    """
    per_doc, golds = [], []
    for _ in range(n):
        C = int(rng.integers(3, 6))
        doc = []
        for _ in range(C):
            nchar = int(rng.integers(3, 40))
            ntok = int(rng.integers(1, 10))
            sc = float(-rng.gamma(3.0, 3.0))
            doc.append((sc, sc / nchar, sc / ntok, nchar, ntok))
        per_doc.append(doc)
        golds.append(int(rng.integers(0, C)))
    return per_doc, golds


def _acc_reference(per_doc, golds, k):
    """The exact reduction eval_arc_easy._acc performs, copied verbatim."""
    return sum(max(range(len(d)), key=lambda i: d[i][k]) == g
               for d, g in zip(per_doc, golds)) / len(per_doc)


# ------------------------------------------------------------------------------------
# 1. the dump is a faithful, lossless record of the three shipped statistics
# ------------------------------------------------------------------------------------

def test_dump_roundtrips_ragged_choices(tmp_path):
    rng = np.random.default_rng(0)
    per_doc, golds = _synth_per_doc(rng)
    p = EA.dump_per_item(str(tmp_path / "d.npz"), per_doc, golds)
    d = S.load_dump(p)

    assert d["raw"].shape == (len(per_doc), 5)          # C = max over docs
    assert list(d["nch"]) == [len(x) for x in per_doc]
    assert list(d["gold"]) == golds
    for i, doc in enumerate(per_doc):
        for j, entry in enumerate(doc):
            assert d["raw"][i, j] == entry[0]
            assert d["nchar"][i, j] == entry[3]
            assert d["ntok"][i, j] == entry[4]
        assert np.isnan(d["raw"][i, len(doc):]).all()   # padded slots are NaN


def test_shipped_accuracies_recomputed_from_dump_are_bit_identical(tmp_path):
    rng = np.random.default_rng(1)
    per_doc, golds = _synth_per_doc(rng, n=500)
    d = S.load_dump(EA.dump_per_item(str(tmp_path / "d.npz"), per_doc, golds))
    got = S.shipped_accuracies(d)
    assert got["acc"] == _acc_reference(per_doc, golds, 0)
    assert got["acc_norm"] == _acc_reference(per_doc, golds, 1)
    assert got["acc_norm_token"] == _acc_reference(per_doc, golds, 2)


def test_padded_choices_never_win_the_argmax(tmp_path):
    """A 3-choice item must never be scored against the 4th/5th padded slot."""
    per_doc = [[(-1.0, -1.0 / 5, -1.0 / 2, 5, 2),
                (-9.0, -9.0 / 5, -9.0 / 2, 5, 2),
                (-9.0, -9.0 / 5, -9.0 / 2, 5, 2)],
               [(-9.0, -9.0 / 5, -9.0 / 2, 5, 2)] * 4 +
               [(-0.5, -0.5 / 5, -0.5 / 2, 5, 2)]]
    golds = [0, 4]
    d = S.load_dump(EA.dump_per_item(str(tmp_path / "d.npz"), per_doc, golds))
    for b in S.BASES:
        assert S.per_item(d, b)["acc"].tolist() == [1.0, 1.0]
    assert S.shipped_accuracies(d) == {"acc": 1.0, "acc_norm": 1.0,
                                       "acc_norm_token": 1.0}


# ------------------------------------------------------------------------------------
# 2. the statistics behave
# ------------------------------------------------------------------------------------

def test_all_statistics_are_per_item_and_finite(tmp_path):
    rng = np.random.default_rng(2)
    per_doc, golds = _synth_per_doc(rng, n=300)
    d = S.load_dump(EA.dump_per_item(str(tmp_path / "d.npz"), per_doc, golds))
    tab = S.stat_table(d)
    assert sorted(tab) == sorted(S.stat_names())
    for k, v in tab.items():
        assert v.shape == (300,), k
        assert np.isfinite(v).all(), k


def test_margin_sign_agrees_with_acc(tmp_path):
    """margin > 0 iff the argmax picked gold -- the continuous statistic must be a
    refinement of the 0/1 one, not a different question."""
    rng = np.random.default_rng(3)
    per_doc, golds = _synth_per_doc(rng, n=400)
    d = S.load_dump(EA.dump_per_item(str(tmp_path / "d.npz"), per_doc, golds))
    for b in S.BASES:
        pi = S.per_item(d, b)
        assert np.array_equal(pi["margin"] > 0, pi["acc"] > 0.5)


def test_pgold_and_brier_are_bounded_and_consistent(tmp_path):
    rng = np.random.default_rng(4)
    per_doc, golds = _synth_per_doc(rng, n=200)
    d = S.load_dump(EA.dump_per_item(str(tmp_path / "d.npz"), per_doc, golds))
    for b in S.BASES:
        pi = S.per_item(d, b)
        assert ((pi["pgold"] >= 0) & (pi["pgold"] <= 1)).all()
        assert ((pi["brier"] >= 0) & (pi["brier"] <= 2)).all()
        assert np.allclose(pi["nll"], -np.log(pi["pgold"]))
        assert ((pi["tanhmargin"] >= -1) & (pi["tanhmargin"] <= 1)).all()


def test_mrr_is_a_refinement_of_acc(tmp_path):
    """MRR == 1 exactly on the items acc scores 1, and 1/rank otherwise."""
    rng = np.random.default_rng(5)
    per_doc, golds = _synth_per_doc(rng, n=300)
    d = S.load_dump(EA.dump_per_item(str(tmp_path / "d.npz"), per_doc, golds))
    for b in S.BASES:
        pi = S.per_item(d, b)
        assert np.array_equal(pi["mrr"] == 1.0, pi["acc"] > 0.5)
        assert set(np.round(1.0 / pi["mrr"], 9)).issubset({1.0, 2.0, 3.0, 4.0, 5.0})
        # a padded slot must never push gold down the ranking
        assert (1.0 / pi["mrr"] <= d["nch"]).all()


def test_orientation_flips_only_loss_like_statistics():
    v = np.array([1.0, 2.0])
    for b in S.BASES:
        assert S.oriented(f"{b}_pgold", v)[0] == 1.0
        assert S.oriented(f"{b}_nll", v)[0] == -1.0
        assert S.oriented(f"{b}_brier", v)[0] == -1.0


def test_tanh_scale_is_a_fixed_constant_not_a_fitted_one():
    """Refitting the squash per arm would make arms incomparable; pin the constants."""
    assert S.TANH_SCALE == {"raw": 3.0, "char": 0.2, "tok": 1.0}


# ------------------------------------------------------------------------------------
# 3. the draw-count arithmetic
# ------------------------------------------------------------------------------------

def test_paired_t_matches_the_known_invdiag_result():
    """
    The session's one resolvable effect: `--scale-weight invdiag` vs `PI_legal`, five
    matched offsets, paired Delta = -8.16 pp at t = -3.53 (SESSION section 4b).
    Recomputing it from the logged acc_norm values pins the estimator.
    """
    ctl = [0.5744949494949495, 0.601010101010101, 0.6464646464646465,
           0.6586700336700336, 0.5900673400673401]
    arm = [0.5420875420875421, 0.5652356902356902, 0.48737373737373735,
           0.5698653198653199, 0.4983164983164983]
    r = S.paired_t(arm, ctl)
    assert r["delta"] == pytest.approx(-0.0816, abs=5e-4)
    assert r["t"] == pytest.approx(-3.53, abs=0.02)
    assert r["n_draws"] == 5 and r["n_negative"] == 5


def test_draws_for_effect_scales_as_the_square_of_sd_over_delta():
    n1 = S.draws_for_effect(0.04, 0.015)
    assert n1 == pytest.approx((2.801585 * 0.04 / 0.015) ** 2, rel=1e-6)
    assert S.draws_for_effect(0.08, 0.015) == pytest.approx(4 * n1, rel=1e-9)
    assert S.draws_for_effect(0.04, 0.030) == pytest.approx(n1 / 4, rel=1e-9)
    assert np.isnan(S.draws_for_effect(0.04, 0.0))


# ------------------------------------------------------------------------------------
# 4. end-to-end: dump_path does not perturb a single reported number
# ------------------------------------------------------------------------------------

def test_eval_arc_easy_is_bit_identical_with_and_without_dump(tmp_path):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()

    kw = dict(device="cpu", limit=24, batch_size=8, verbose=False)
    a = EA.eval_arc_easy(model, tok, **kw)
    p = str(tmp_path / "d.npz")
    b = EA.eval_arc_easy(model, tok, dump_path=p, **kw)
    assert a == b                                   # bit-identical, not just close

    d = S.load_dump(p)
    got = S.shipped_accuracies(d)
    assert got["acc"] == a["acc"]
    assert got["acc_norm"] == a["acc_norm"]
    assert got["acc_norm_token"] == a["acc_norm_token"]
    assert int(d["nch"].shape[0]) == a["n"]
