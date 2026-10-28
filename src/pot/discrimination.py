"""
Separate DISCRIMINATION from CALIBRATION for every multiple-choice arm.

Why this exists
---------------
Accuracy on a multiple-choice benchmark is a thresholded statistic, and a quantized
model can move it in two completely different ways:

  * discrimination -- it really does rank the gold answer above the distractors more
    often. This is capability.
  * calibration / sharpness -- the ordering is unchanged, but the score distribution is
    stretched, shifted, or re-tilted with respect to continuation LENGTH, so items near
    the decision boundary change sides. This is not capability, and on a
    length-normalised metric like `acc_norm` a uniform per-token log-probability shift
    is enough to do it.

The panel study needs the distinction because the first four arms produced a perfect
rank correlation between ARC-Easy `acc_norm`, the mean per-token log-probability of the
chosen answer, and BoolQ's yes/no margin -- three quantities that would all move together
under a pure sharpness change.

The statistics
--------------
`auc_raw`    P(gold's RAW summed log-prob > a random distractor's). Invariant to any
             per-item monotone transform of the scores that is applied equally to every
             choice, so a global sharpening cannot move it. This is the cleanest
             available discrimination measure.
`auc_norm`   the same on the length-normalised score `logprob / n_chars`, i.e. on the
             quantity `acc_norm` actually argmaxes. It is NOT sharpness-invariant: a
             per-token shift re-tilts it against length. `auc_norm - auc_raw` is
             therefore a direct readout of how much length-tilt the artifact carries.
`lp_per_tok` mean per-token log-probability over ALL choices -- the sharpness itself.
`len_pref`   mean n_chars of the chosen continuation, the length preference `acc_norm`
             expresses.

A difference that shows up in `acc`/`acc_norm` but not in `auc_raw` is a calibration
change wearing capability's clothes.
"""

from __future__ import annotations

import os

import numpy as np


def _load(path):
    d = np.load(path)
    raw = np.where(np.isnan(d["raw"]), -np.inf, d["raw"])
    return raw, d["nchar"], d["ntok"], d["gold"]


def item_stats(path):
    """All statistics for one dumped task. Returns a dict."""
    raw, nchar, ntok, gold = _load(path)
    n, C = raw.shape
    rows = np.arange(n)
    valid = np.isfinite(raw)
    normed = np.where(nchar > 0, raw / np.maximum(nchar, 1), -np.inf)

    def auc(M):
        """P(gold > a uniformly random distractor), averaged over items.

        Ties count as 1/2, which is the standard convention and matters because two
        choices can share a score exactly at low precision.
        """
        tot = 0.0
        for i in range(n):
            others = [j for j in range(C) if valid[i, j] and j != gold[i]]
            if not others:
                continue
            g = M[i, gold[i]]
            o = M[i, others]
            tot += float(np.mean((g > o) + 0.5 * (g == o)))
        return tot / n

    pick_raw = np.argmax(raw, 1)
    pick_norm = np.argmax(normed, 1)
    return {
        "n": int(n),
        "acc": float((pick_raw == gold).mean()),
        "acc_norm": float((pick_norm == gold).mean()),
        "auc_raw": auc(raw),
        "auc_norm": auc(normed),
        "lp_per_tok": float(np.mean(raw[valid] / np.maximum(ntok, 1)[valid])),
        "len_pref": float(nchar[rows, pick_norm].mean()),
    }


def report(dump_root, tags, tasks):
    """Print the discrimination/calibration split for every tag x task available."""
    out = {}
    for tag in tags:
        per = {}
        for t in tasks:
            p = (os.path.join(dump_root, f"{tag}_arce.npz") if t == "arc_easy"
                 else os.path.join(dump_root, tag, f"{t}.npz"))
            if os.path.exists(p):
                per[t] = item_stats(p)
        if per:
            out[tag] = per

    for t in tasks:
        have = [g for g in out if t in out[g]]
        if len(have) < 2:
            continue
        print(f"\n--- {t} ---")
        print(f"  {'arm':<18} {'acc':>7} {'acc_norm':>9} {'auc_raw':>8} {'auc_norm':>9} "
              f"{'tilt':>7} {'lp/tok':>8} {'len':>7}")
        for g in have:
            s = out[g][t]
            print(f"  {g:<18} {s['acc']:>7.4f} {s['acc_norm']:>9.4f} {s['auc_raw']:>8.4f} "
                  f"{s['auc_norm']:>9.4f} {s['auc_norm'] - s['auc_raw']:>+7.4f} "
                  f"{s['lp_per_tok']:>8.4f} {s['len_pref']:>7.2f}")
        # how much of the accuracy spread survives in the threshold-free statistic
        a = [out[g][t]["acc_norm"] for g in have]
        r = [out[g][t]["auc_raw"] for g in have]
        print(f"  spread: acc_norm {100*(max(a)-min(a)):.2f} pp   "
              f"auc_raw {100*(max(r)-min(r)):.2f} pp")
    return out
