"""
Read out the panel study: does the artifact-to-artifact ARC-Easy swing survive on
seven other benchmarks?

Run: python3 src/pot/panel_analysis.py [tag_prefix]   (default prefix "PN_")

Everything here is descriptive. The pre-registered reading lives in
`scripts/pot_panel_study.sh` and is not restated (or quietly amended) here.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pot.eval_mc import damage_fraction, BF16_ARCE_ACC_NORM, CHANCE_ARCE  # noqa: E402

RUNS = "results/pot/pot_runs.jsonl"
DUMPS = "results/pot/mcdump"


def load(prefix="PN_"):
    out = {}
    with open(RUNS) as f:
        for line in f:
            r = json.loads(line)
            if r["tag"].startswith(prefix) and r.get("mc"):
                out[r["tag"]] = r                       # later runs win
    return out


def _rank(x):
    order = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    return order.astype(float)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    p = float(np.corrcoef(x, y)[0, 1])
    s = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return p, s


def binom_se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-12) / n)


def main(prefix="PN_"):
    runs = load(prefix)
    if not runs:
        print(f"no runs with mc results under prefix {prefix!r}")
        return
    tags = sorted(runs, key=lambda t: -runs[t]["arc"]["acc_norm"])
    tasks = list(next(iter(runs.values()))["mc"]["tasks"])

    print(f"{'tag':<16} {'bpw':>9} {'ARC-E':>7} {'ARC-E':>7} | {'panel':>7} {'panel':>7} | "
          + " ".join(f"{t[:8]:>8}" for t in tasks))
    print(f"{'':<16} {'':>9} {'accnorm':>7} {'acc':>7} | {'macro':>7} {'micro':>7} |")
    for t in tags:
        r = runs[t]
        m = r["mc"]
        print(f"{t:<16} {r['bpw']:>9.6f} {r['arc']['acc_norm']:>7.4f} {r['arc']['acc']:>7.4f} | "
              f"{m['panel_macro']:>7.4f} {m['panel_micro']:>7.4f} | "
              + " ".join(f"{m['tasks'][k]['headline']:>8.4f}" for k in tasks))

    arce = [runs[t]["arc"]["acc_norm"] for t in tags]
    arce_acc = [runs[t]["arc"]["acc"] for t in tags]
    macro = [runs[t]["mc"]["panel_macro"] for t in tags]
    micro = [runs[t]["mc"]["panel_micro"] for t in tags]

    print(f"\n=== agreement across {len(tags)} arms ===")
    for name, v in (("panel_macro", macro), ("panel_micro", micro), ("ARC-E acc", arce_acc)):
        p, s = corr(arce, v)
        print(f"  ARC-E acc_norm vs {name:<12}: pearson {p:+.3f}  spearman {s:+.3f}")
    print("  per-task vs ARC-E acc_norm:")
    for k in tasks:
        v = [runs[t]["mc"]["tasks"][k]["headline"] for t in tags]
        p, s = corr(arce, v)
        print(f"    {k:<11} pearson {p:+.3f}  spearman {s:+.3f}")

    # Spread comparison, in units of each metric's own item-sampling se. This is the
    # question the study exists for: ARC-E moves 12.5 pp between two random
    # permutations -- how many se is that on ARC-E, and how many on the panel?
    nll = [gold_nll(t, tasks) for t in tags]
    if not any(math.isnan(v) for v in nll):
        print("\n=== continuous readout: mean per-token gold NLL over the panel ===")
        for t, v in zip(tags, nll):
            print(f"  {t:<16} {v:.5f}")
        for name, v in (("ARC-E acc_norm", arce), ("panel_macro", macro)):
            pe, sp = corr(nll, v)
            print(f"  gold_nll vs {name:<15}: pearson {pe:+.3f}  spearman {sp:+.3f} "
                  f"(negative = lower NLL goes with higher accuracy, i.e. sane)")

    # ---------------------------------------------------------------- robustness
    # An unweighted macro lets the single most volatile task dominate the SPREAD even
    # when it does not dominate the LEVEL. On the first four arms BoolQ moved 11.07 pp
    # (8.6 item-sampling se) against 0.5-3.1 se for every other task, and BoolQ alone
    # correlated -0.975 with ARC-Easy -- so the headline correlation has to be shown to
    # survive dropping any one task, or it is a claim about BoolQ and not about the
    # panel. Reported whether or not it survives.
    print("\n=== leave-one-task-out: does the panel's verdict depend on one task? ===")
    print(f"  {'dropped':<12} {'macro range':>12} {'r vs ARC-E':>12} {'rho vs ARC-E':>13}")
    base_p, base_s = corr(arce, macro)
    print(f"  {'(none)':<12} {100*(max(macro)-min(macro)):>11.2f}p {base_p:>+12.3f} "
          f"{base_s:>+13.3f}")
    for drop in tasks:
        keep = [k for k in tasks if k != drop]
        m2 = [float(np.mean([runs[t]["mc"]["tasks"][k]["headline"] for k in keep]))
              for t in tags]
        p2, s2 = corr(arce, m2)
        print(f"  {drop:<12} {100*(max(m2)-min(m2)):>11.2f}p {p2:>+12.3f} {s2:>+13.3f}")

    # Headroom-normalised damage: 0.0 = bf16, 1.0 = chance. This is the only scale on
    # which "ARC-E moved X pp" and "the panel moved Y pp" are comparable quantities.
    print("\n=== damage as a fraction of the above-chance bf16 headroom ===")
    print(f"  {'tag':<16} {'ARC-E':>8} {'panel':>8}")
    dmg_a, dmg_p = [], []
    for t in tags:
        da = ((BF16_ARCE_ACC_NORM - runs[t]["arc"]["acc_norm"])
              / (BF16_ARCE_ACC_NORM - CHANCE_ARCE))
        dp = damage_fraction({k: runs[t]["mc"]["tasks"][k]["headline"] for k in tasks})
        dmg_a.append(da); dmg_p.append(dp)
        print(f"  {t:<16} {da:>8.4f} {dp:>8.4f}")
    pe, sp = corr(dmg_a, dmg_p)
    print(f"  agreement: pearson {pe:+.3f}  spearman {sp:+.3f}")
    print(f"  spread    ARC-E {max(dmg_a)-min(dmg_a):.4f}   panel {max(dmg_p)-min(dmg_p):.4f}"
          f"   ratio {(max(dmg_a)-min(dmg_a))/max(max(dmg_p)-min(dmg_p), 1e-9):.2f}x")

    print("\n=== spread across arms, in units of item-sampling se ===")
    n_arce = runs[tags[0]]["arc"]["n"]
    se_arce = binom_se(float(np.mean(arce)), n_arce)
    se_macro = math.sqrt(sum(binom_se(np.mean([runs[t]["mc"]["tasks"][k]["headline"]
                                               for t in tags]),
                                      runs[tags[0]]["mc"]["tasks"][k]["n"]) ** 2
                             for k in tasks)) / len(tasks)
    print(f"  ARC-E acc_norm : range {100*(max(arce)-min(arce)):6.2f} pp "
          f"= {(max(arce)-min(arce))/se_arce:6.2f} se   (se {100*se_arce:.2f} pp, n={n_arce})")
    print(f"  panel_macro    : range {100*(max(macro)-min(macro)):6.2f} pp "
          f"= {(max(macro)-min(macro))/se_macro:6.2f} se   (se {100*se_macro:.2f} pp, "
          f"n={runs[tags[0]]['mc']['n_items']})")

    # ------------------------------------------------- output-bias drift, via BoolQ
    # BoolQ's two choices are the fixed strings " no" and " yes", so logP(yes)-logP(no)
    # is a direct readout of the model's yes/no output bias -- a quantity that has
    # nothing to do with reading comprehension. It is reported because on the first four
    # arms it accounted for essentially all of `panel_macro`'s spread: an artifact that
    # answers "yes" to 93% of items scores near the 0.6153 always-yes baseline while
    # looking like a capability change.
    rows = []
    for t in tags:
        p = os.path.join(DUMPS, t, "boolq.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        raw, gold = d["raw"], d["gold"]
        pred = np.argmax(np.where(np.isnan(raw), -np.inf, raw), 1)
        rows.append((t, float((pred == gold).mean()), float(pred.mean()),
                     float(np.mean(raw[:, 1] - raw[:, 0]))))
    if rows:
        print("\n=== output-bias drift (BoolQ: choices are the constants ' no'/' yes') ===")
        print(f"  gold rate of 'yes' = {float(np.mean(np.load(os.path.join(DUMPS, rows[0][0], 'boolq.npz'))['gold'])):.4f}"
              "   (an always-yes model scores exactly this)")
        print(f"  {'tag':<18} {'acc':>7} {'pred-yes':>9} {'mean margin':>12}")
        for t, acc, py, mg in rows:
            print(f"  {t:<18} {acc:>7.4f} {py:>9.4f} {mg:>12.4f}")

    # The load-bearing pair, if both arms are present.
    a, b = f"{prefix}rand0", f"{prefix}rand2"
    if a in runs and b in runs:
        print(f"\n=== the load-bearing pair: {a} vs {b} "
              f"(identical recipe, different random column permutation) ===")
        d_arce = runs[a]["arc"]["acc_norm"] - runs[b]["arc"]["acc_norm"]
        d_mac = runs[a]["mc"]["panel_macro"] - runs[b]["mc"]["panel_macro"]
        print(f"  ARC-E acc_norm delta {100*d_arce:+6.2f} pp = {d_arce/(se_arce*math.sqrt(2)):+5.2f} se")
        print(f"  panel_macro   delta {100*d_mac:+6.2f} pp = {d_mac/(se_macro*math.sqrt(2)):+5.2f} se")
        print("  per task:")
        for k in tasks:
            x = runs[a]["mc"]["tasks"][k]["headline"]
            y = runs[b]["mc"]["tasks"][k]["headline"]
            se = binom_se((x + y) / 2, runs[a]["mc"]["tasks"][k]["n"]) * math.sqrt(2)
            print(f"    {k:<11} {x:.4f} vs {y:.4f}  delta {100*(x-y):+6.2f} pp "
                  f"= {(x-y)/se:+5.2f} se")
        _paired_items(a, b, tasks)


def gold_nll(tag, tasks):
    """Mean per-token NLL of the GOLD continuation over the whole panel.

    Free from the dumps, and a continuous readout rather than an argmax: it uses the
    magnitude of the log-probability the model assigns to the right answer, not merely
    whether it beat the distractors. SESSION 10.2 found continuous readouts do not beat
    `acc` on ARC-Easy alone (the artifact shifts the whole per-item distribution
    coherently, so a continuous statistic carries the same RELATIVE noise). Whether that
    still holds when pooled over 7 tasks and ~34k choices is a different question, and
    this costs nothing to answer.
    """
    tot, n = 0.0, 0
    for k in tasks:
        p = os.path.join(DUMPS, tag, f"{k}.npz")
        if not os.path.exists(p):
            return float("nan")
        d = np.load(p)
        raw, ntok, gold = d["raw"], d["ntok"], d["gold"]
        rows = np.arange(len(gold))
        tot += float(np.sum(-raw[rows, gold] / np.maximum(ntok[rows, gold], 1)))
        n += len(gold)
    return tot / max(n, 1)


def _paired_items(a, b, tasks):
    """Item-level pairing from the dumps: McNemar over every panel item at once.

    A paired test removes item difficulty entirely, which is the only variance
    component an unpaired comparison of two deterministic artifacts can remove.
    """
    tot = [0, 0, 0]                                    # both, a-only, b-only
    for k in list(tasks) + ["_arce"]:
        pa = (os.path.join(DUMPS, a, f"{k}.npz") if k != "_arce"
              else os.path.join(DUMPS, f"{a}_arce.npz"))
        pb = (os.path.join(DUMPS, b, f"{k}.npz") if k != "_arce"
              else os.path.join(DUMPS, f"{b}_arce.npz"))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        da, db = np.load(pa), np.load(pb)
        # headline for the panel is acc_norm for some tasks and acc for others; the
        # paired test below uses raw acc, which is the statistic SESSION 10.2 measured
        # as the least noisy of the three.
        ca = np.nanargmax(np.where(np.isnan(da["raw"]), -np.inf, da["raw"]), 1) == da["gold"]
        cb = np.nanargmax(np.where(np.isnan(db["raw"]), -np.inf, db["raw"]), 1) == db["gold"]
        lbl = "ARC-E (held out)" if k == "_arce" else k
        n01 = int((~ca & cb).sum())
        n10 = int((ca & ~cb).sum())
        if k != "_arce":
            tot[1] += n10
            tot[2] += n01
        z = (n10 - n01) / math.sqrt(max(n10 + n01, 1))
        print(f"    [paired] {lbl:<18} a-only {n10:5d}  b-only {n01:5d}  "
              f"McNemar z {z:+6.2f}")
    if tot[1] + tot[2]:
        z = (tot[1] - tot[2]) / math.sqrt(tot[1] + tot[2])
        print(f"    [paired] {'PANEL POOLED':<18} a-only {tot[1]:5d}  b-only {tot[2]:5d}  "
              f"McNemar z {z:+6.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PN_")
