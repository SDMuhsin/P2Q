"""Read out the --solve-scale validation against its PRE-REGISTERED criteria.

Run: python3 src/pot/ss_validate_report.py

The criteria are stated in scripts/pot_multimodel_validate.sh and in the earlier
scripts/pot_ss_validate.sh, both written before any arm ran. This module only
EVALUATES them; it does not restate, relax or amend them, and it prints every arm it
finds including the ones that make the candidate look worse.

Two statistics, both fixed by earlier sessions:
    panel_macro  unweighted mean of the 7 headline accuracies      (PRIMARY)
    core_6       the same over the six tasks EXCLUDING BoolQ.  BoolQ is a binary task
                 whose accuracy sits near a degenerate all-one-class solution, so it
                 moves for reasons unrelated to capability; core_6 is the statistic
                 SESSION 4 used to overturn 11.3 and to withdraw strict_pot_zero.
ARC-Easy is reported but never decides: it is the held-out test set, and its
artifact-to-artifact swing is ~17x the panel's on provably-equivalent pairs.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pot.eval_mc import PANEL_TASKS, _HEADLINE  # noqa: E402

RUNS = "results/pot/pot_runs.jsonl"
BASELINES = "results/pot/baselines.jsonl"
CORE_6 = tuple(t for t in PANEL_TASKS if t != "boolq")


def load(path, need_mc=True):
    """tag -> record, last write wins (a re-run supersedes an earlier one)."""
    out = {}
    try:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tag = r.get("tag")
            if not tag:
                continue
            if need_mc and not r.get("mc"):
                continue
            out[tag] = r
    except FileNotFoundError:
        pass
    return out


def heads(rec):
    return {t: rec["mc"]["tasks"][t]["headline"] for t in PANEL_TASKS}


def stats(rec):
    h = heads(rec)
    arc = rec.get("arc") or {}
    return {
        "arce": arc.get("acc_norm"),
        "panel": rec["mc"]["panel_macro"],
        "core6": float(np.mean([h[t] for t in CORE_6])),
        "ppl": rec.get("ppl"),
        "bpw": rec.get("bpw"),
        "heads": h,
    }


def _fmt(v, w=7, p=4):
    return " " * w if v is None else f"{v:{w}.{p}f}"


def row(label, s, ref=None):
    d = ""
    if ref is not None:
        def dd(k, scale=100.0, unit="pp"):
            if s.get(k) is None or ref.get(k) is None:
                return "     --"
            return f"{scale * (s[k] - ref[k]):+6.2f}"
        d = f"  |  dARC {dd('arce')}  dpanel {dd('panel')}  dcore6 {dd('core6')}"
    print(f"  {label:<18s} ARC-E {_fmt(s['arce'])}  panel {_fmt(s['panel'])}  "
          f"core6 {_fmt(s['core6'])}  ppl {_fmt(s['ppl'], 8, 3)}  "
          f"bpw {_fmt(s['bpw'], 9, 6)}{d}")


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def main():
    R = load(RUNS)
    B = load(BASELINES)

    missing = []

    def get(tag, store=None):
        rec = (store or R).get(tag)
        if rec is None:
            missing.append(tag)
        return rec

    # ------------------------------------------------------------------ STAGE 1
    section("STAGE 1  Qwen3-1.7B -- replication of SS_mu1 (pre-registered in "
            "pot_ss_validate.sh)")

    inc = get("PN_incumbent")
    if inc:
        print("\n[incumbent: RTN + exact scale, deterministic -- reads NO calibration data]")
        row("PN_incumbent", stats(inc))

    print("\n[axis B: uniform g=128. PASS iff positive on core_6.]")
    b_base, b_ss = get("ZA3_g128_pot"), get("SS_g128_mu1")
    if b_base and b_ss:
        row("ZA3_g128_pot", stats(b_base))
        row("SS_g128_mu1", stats(b_ss), ref=stats(b_base))
        verdict = "POSITIVE" if stats(b_ss)["core6"] > stats(b_base)["core6"] else "NEGATIVE"
        print(f"   -> axis B core_6 {verdict}")

    print("\n[draw spread: --solve-scale reads calibration data, the incumbent does not.\n"
          " PASS iff the 3-draw ARC-E mean > 0.6364 AND the 3-draw core_6 mean > 0.4973.\n"
          " A spread > ~4 pp on ARC-E is a COST (a lottery artifact replacing an exact one).]")
    draws = [(t, get(t)) for t in ("SS_mu1", "SS_mu1_o128", "SS_mu1_o256")]
    got = [(t, r) for t, r in draws if r]
    for t, r in got:
        row(t, stats(r), ref=stats(inc) if inc else None)
    if len(got) >= 2:
        arce = [stats(r)["arce"] for _, r in got]
        core = [stats(r)["core6"] for _, r in got]
        pan = [stats(r)["panel"] for _, r in got]
        n = len(got)
        print(f"   -> over {n} draw(s): ARC-E mean {np.mean(arce):.4f} "
              f"spread {100 * (max(arce) - min(arce)):.2f} pp   |   "
              f"core_6 mean {np.mean(core):.4f} spread {100 * (max(core) - min(core)):.2f} pp"
              f"   |   panel mean {np.mean(pan):.4f}")
        if n == 3:
            ok_arce, ok_core = np.mean(arce) > 0.6364, np.mean(core) > 0.4973
            print(f"   -> PRE-REGISTERED: ARC-E mean > 0.6364 ? {ok_arce}   "
                  f"core_6 mean > 0.4973 ? {ok_core}   "
                  f"=> {'PASS' if (ok_arce and ok_core) else 'FAIL'}")
            if 100 * (max(arce) - min(arce)) > 4.0:
                print("   -> COST: the draw spread exceeds 4 pp on ARC-Easy. The deterministic "
                      "incumbent has no such spread.")
        else:
            print(f"   -> INCOMPLETE: {n} of 3 draws present; the pre-registered test needs 3.")

    print("\n[second codebook: strict_pot_zero]")
    z_base, z_ss = get("PN_zero"), get("SS_zero_mu1")
    if z_base and z_ss:
        row("PN_zero", stats(z_base))
        row("SS_zero_mu1", stats(z_ss), ref=stats(z_base))

    # ------------------------------------------------------------------ STAGE 2
    section("STAGE 2  generalisation -- SS_mu1 minus incumbent, per model "
            "(pre-registered in pot_multimodel_validate.sh)")
    print("\nThe panel is the selection instrument; ARC-Easy is held out and never decides.")

    models = [("Qwen3-1.7B", "PN_incumbent", "SS_mu1", None),
              ("Qwen3-0.6B", "MM_q06_rtn", "MM_q06_ss1", "MM_q06_bf16"),
              ("gemma-2b", "MM_g2b_rtn", "MM_g2b_ss1", "MM_g2b_bf16")]
    deltas = {}
    for name, rtn_tag, ss_tag, bf_tag in models:
        print(f"\n-- {name}")
        if bf_tag:
            bf = B.get(bf_tag) or R.get(bf_tag)
            if bf and bf.get("mc"):
                b = stats(bf) if bf.get("arc") else None
                m = bf["mc"]
                arce = bf.get("arc_easy_acc_norm")
                h = {t: m["tasks"][t]["headline"] for t in PANEL_TASKS}
                print(f"  {'bf16 reference':<18s} ARC-E {_fmt(arce)}  "
                      f"panel {_fmt(m['panel_macro'])}  "
                      f"core6 {_fmt(float(np.mean([h[t] for t in CORE_6])))}  "
                      f"ppl {_fmt(bf.get('wikitext2_ppl'), 8, 3)}")
            else:
                missing.append(bf_tag)
        a, c = R.get(rtn_tag), R.get(ss_tag)
        if not (a and c):
            for t in (rtn_tag, ss_tag):
                if t not in R:
                    missing.append(t)
            print("  (incomplete)")
            continue
        sa, sc = stats(a), stats(c)
        row(f"{rtn_tag} (RTN)", sa)
        row(f"{ss_tag} (SS)", sc, ref=sa)
        deltas[name] = (sc["panel"] - sa["panel"], sc["core6"] - sa["core6"],
                        (sc["arce"] - sa["arce"]) if None not in (sc["arce"], sa["arce"]) else None)

    print("\n  NOTE: Qwen3-0.6B is scored at --mc-token-budget 1024, not 4096, so its "
          "ABSOLUTE\n  panel numbers are not comparable to Qwen3-1.7B's. Both arms of "
          "its pair share the\n  budget, so the within-model delta above IS exact. Read "
          "the deltas, not the levels.")
    print("  NOTE: gemma-2b needs ~20 GB and this box has had that for 0.5% of the last "
          "15 h.\n  It is verified working (CPU smoke, bpw 4.249407) and is DEFERRED, "
          "not refuted. It is the\n  only architecture axis; run it first on any free "
          "GPU.")

    new = [m for m in ("Qwen3-0.6B", "gemma-2b") if m in deltas]
    if len(new) == 2:
        signs = [deltas[m][0] > 0 for m in new]
        if all(signs):
            verdict = "GENERALISES (panel_macro positive on both new models)"
        elif not any(signs):
            verdict = ("FORMAT-SPECIFIC -- panel_macro is at or below zero on both new "
                       "models, so the mechanism is a Qwen3-1.7B fact and SS_mu1 must "
                       "not be promoted as a property of strict PoT")
        else:
            verdict = ("SPLIT -- positive on one model, negative on the other. Promotes "
                       "nothing, and is NOT resolved by running more arms on the winner")
        print(f"\n  => SECONDARY VERDICT: {verdict}")
        for m in new:
            dp, dc, da = deltas[m]
            print(f"       {m:<12s} dpanel {100 * dp:+.2f} pp   dcore6 {100 * dc:+.2f} pp"
                  + (f"   dARC-E {100 * da:+.2f} pp" if da is not None else ""))
    elif new:
        print(f"\n  => SECONDARY VERDICT NOT EVALUABLE: {len(new)} of 2 new models "
              f"reachable on this box.")
        print("     GENERALISES / FORMAT-SPECIFIC / SPLIT all require two models. One "
              "model settles\n     none of them and must not be reported as if it did. "
              "What the single model gives\n     is a SIZE-axis data point within the "
              "same family, not an architecture axis:")
        for m in new:
            dp, dc, da = deltas[m]
            print(f"       {m:<12s} dpanel {100 * dp:+.2f} pp   dcore6 {100 * dc:+.2f} pp"
                  + (f"   dARC-E {100 * da:+.2f} pp" if da is not None else ""))

    if missing:
        print(f"\n[not yet in the logs: {', '.join(sorted(set(missing)))}]")


if __name__ == "__main__":
    main()
