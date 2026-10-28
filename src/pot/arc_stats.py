"""
Higher-power readouts of an ARC-Easy evaluation, computed from the raw per-item,
per-choice log-probability table dumped by `pot.eval_arc.dump_per_item`.

WHY THIS EXISTS -- and what it is NOT
-------------------------------------
`eval_arc.py` computes a continuous score for every (item, choice) pair and then
reduces it to a 0/1 argmax. The margins -- the confidence structure -- are discarded.
Over 2376 items that costs power: the project's decisions are limited by a
draw-to-draw sd of ~4 pp on `acc_norm`, and nine hypotheses in a row died with the
verdict "not resolvable at 5 draws".

This module is an **INSTRUMENT, NOT A TARGET.** CHARTER's acceptance criterion is and
remains ARC-Easy `acc_norm >= 0.6400`. Nothing here may be reported as a headline, and
no arm may be presented as hitting the target on a statistic defined in this file. The
only job of these statistics is to *rank* interventions with more power than `acc_norm`
can; the deliverable is still reported in `acc_norm`.

A low-variance statistic is worthless if it is low-variance because it is insensitive,
so every statistic here must be validated against BOTH a known-positive control
(`--scale-weight invdiag`, a real -8.2 pp degradation at t = -3.53) and known nulls
(`ldl`, `powdiag:-0.5`, `strict_pot_zero`, `--codebook-nexp 16`). A statistic that
"resolves" a null is broken, not insightful.

THE SCORING BASES
-----------------
Each item has C in {3,4,5} choices with a summed continuation log-probability s_ij.
Three bases, matching the three shipped accuracies:
  raw   : s_ij                      -> `acc`
  char  : s_ij / len(cont_string)   -> `acc_norm`        (the CHARTER metric)
  tok   : s_ij / n_cont_tokens      -> `acc_norm_token`
The normalized bases are per-character / per-token rates, so before exponentiating
them into a choice distribution we multiply by a FIXED reference length L0 (the
dataset-mean divisor). L0 depends only on the item set, which is identical across every
arm, so it is a constant of the instrument and not a fitted parameter.

THE STATISTICS (all per-item, so any of them can also be paired item-by-item)
----------------------------------------------------------------------------
  acc        0/1 argmax. The incumbent readout; baseline for everything else.
  margin     s_gold - max_{j != gold} s_j. Continuous, signed, unbounded. Directly
             measures how close the argmax was to flipping -- exactly the information
             the 0/1 reduction throws away.
  tanhmargin tanh(margin / TANH_SCALE). Same ordering, but bounded, so a handful of
             items with enormous margins cannot dominate the mean.
  pgold      softmax over the choices, probability assigned to gold. Bounded in [0,1],
             and `acc` is its hard-argmax degenerate case, so it measures the same
             quantity with the ties resolved smoothly. (Linear/probability score.)
  nll        -log pgold. A strictly proper scoring rule -- but unbounded below-left,
             so a single confidently-wrong item can move the mean a long way. Reported
             for completeness; expect it to be noisier than pgold.
  brier      sum_j (p_j - onehot_j)^2. The other textbook proper scoring rule,
             bounded in [0, 2].
  mrr        1 / (rank of gold among the choices by score). Ordinal: it uses the whole
             ranking rather than just "was gold first", but makes no assumption about
             the scale of the scores, so unlike margin/pgold it cannot be distorted by
             the fat-tailed log-probability distribution. `acc` is the indicator that
             this equals 1.
Signs are normalized by `HIGHER_IS_BETTER` so every statistic can be read "up = better".
"""

from __future__ import annotations

import numpy as np

# Fixed squashing scale for `tanhmargin`, in the units of each basis. Set to a round
# number near the median |margin| measured on the control arm (raw 3.32, char 0.198,
# tok 1.05 on G_ctl_off0), so that typical items sit on the roughly-linear part of tanh
# and only genuine outliers are compressed. These are FIXED constants of the instrument,
# never fitted per arm -- refitting per arm would make arms incomparable.
TANH_SCALE = {"raw": 3.0, "char": 0.2, "tok": 1.0}

BASES = ("raw", "char", "tok")

HIGHER_IS_BETTER = {"acc": True, "margin": True, "tanhmargin": True,
                    "pgold": True, "nll": False, "brier": False, "mrr": True}


def load_dump(path):
    """Load a per-item dump written by `pot.eval_arc.dump_per_item`."""
    with np.load(path) as z:
        return {k: z[k] for k in ("raw", "nchar", "ntok", "nch", "gold")}


def _valid_mask(d):
    """(n, C) boolean: True where the choice slot is a real choice."""
    C = d["raw"].shape[1]
    return np.arange(C)[None, :] < d["nch"][:, None]


def basis_scores(d, basis, l0=None):
    """
    (n, C) score array in the requested basis, -inf in padded slots.

    `l0` is the fixed reference length used to put the normalized bases back on a
    log-probability scale before softmax; None means "dataset mean divisor", which is
    a constant of the item set (identical for every arm).
    """
    m = _valid_mask(d)
    raw = np.where(m, np.nan_to_num(d["raw"], nan=0.0), 0.0)
    if basis == "raw":
        s = raw
        scale = 1.0
    elif basis == "char":
        div = np.where(m, d["nchar"], 1).astype(np.float64)
        s = raw / div
        scale = float(div[m].mean()) if l0 is None else float(l0)
    elif basis == "tok":
        div = np.where(m, d["ntok"], 1).astype(np.float64)
        s = raw / div
        scale = float(div[m].mean()) if l0 is None else float(l0)
    else:
        raise ValueError(f"unknown basis {basis!r}")
    return np.where(m, s, -np.inf), scale


def _softmax(s):
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(s)                       # -inf slots -> exactly 0
    return e / e.sum(axis=1, keepdims=True)


def per_item(d, basis):
    """
    dict of per-item (n,) arrays for one scoring basis. Keys are the bare statistic
    names ("acc", "margin", ...); `stat_table` prefixes them with the basis.
    """
    s, scale = basis_scores(d, basis)
    n, C = s.shape
    gold = d["gold"].astype(int)
    rows = np.arange(n)

    s_gold = s[rows, gold]
    s_other = s.copy()
    s_other[rows, gold] = -np.inf
    s_best_other = s_other.max(axis=1)

    margin = s_gold - s_best_other
    # argmax with the harness's tie-break (first index wins), matching `_acc`.
    acc = (s.argmax(axis=1) == gold).astype(np.float64)

    p = _softmax(s * scale)
    p_gold = p[rows, gold]
    onehot = np.zeros_like(p)
    onehot[rows, gold] = 1.0
    brier = ((p - onehot) ** 2).sum(axis=1)

    return {
        "acc": acc,
        "margin": margin,
        "tanhmargin": np.tanh(margin / TANH_SCALE[basis]),
        "pgold": p_gold,
        "nll": -np.log(np.clip(p_gold, 1e-300, None)),
        "brier": brier,
        "mrr": 1.0 / (1.0 + (s > s_gold[:, None]).sum(axis=1)),
    }


def stat_table(d):
    """All statistics x all bases for one run: {"<basis>_<stat>": (n,) array}."""
    out = {}
    for b in BASES:
        for k, v in per_item(d, b).items():
            out[f"{b}_{k}"] = v
    return out


def stat_names():
    return [f"{b}_{k}" for b in BASES for k in HIGHER_IS_BETTER]


def oriented(name, values):
    """Flip sign so that larger is always better, for uniform reporting of Delta/t."""
    stat = name.split("_", 1)[1]
    return values if HIGHER_IS_BETTER[stat] else -values


def shipped_accuracies(d):
    """
    Reproduce eval_arc_easy's three reported numbers from the dump alone.
    Used to assert the dump is a faithful record of what was scored.
    """
    return {
        "acc": float(per_item(d, "raw")["acc"].mean()),
        "acc_norm": float(per_item(d, "char")["acc"].mean()),
        "acc_norm_token": float(per_item(d, "tok")["acc"].mean()),
    }


# ------------------------------------------------------------------------------------
# Draw-level comparison machinery
# ------------------------------------------------------------------------------------

def paired_t(arm_means, ctl_means):
    """
    Paired-by-offset Delta, sd, sem and t. `arm_means`/`ctl_means` are equal-length
    sequences of per-draw arm-level means on MATCHED calibration offsets.
    """
    a = np.asarray(arm_means, dtype=np.float64)
    c = np.asarray(ctl_means, dtype=np.float64)
    assert a.shape == c.shape and a.ndim == 1
    dif = a - c
    k = len(dif)
    sd = float(dif.std(ddof=1))
    sem = sd / np.sqrt(k)
    return {"delta": float(dif.mean()), "sd": sd, "sem": sem,
            "t": float(dif.mean() / sem) if sem > 0 else float("nan"),
            "n_draws": k, "n_negative": int((dif < 0).sum())}


# Two-sided alpha = 0.05, power = 0.80. The exact normal-approximation constant is
# (z_{0.975} + z_{0.80}) = 1.95996 + 0.84162 = 2.80158; the t-distribution correction at
# these tiny n makes the true requirement larger, so treat the result as a LOWER BOUND
# on the number of draws.
_Z80 = 1.959964 + 0.841621


def draws_for_effect(paired_sd, delta_target):
    """Draws needed to resolve `delta_target` at 80% power, two-sided alpha=0.05."""
    if delta_target == 0 or not np.isfinite(paired_sd) or paired_sd == 0:
        return float("nan")
    return float((_Z80 * paired_sd / abs(delta_target)) ** 2)
