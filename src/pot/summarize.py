"""Group results/pot/pot_runs.jsonl by config-excluding-calib_offset and report
n / mean / sd / sem of arc.acc_norm, plus bpw and verify.ok, per arm.

Two runs are "the same arm" iff their config dicts are equal after dropping
calib_offset, seed, tag, out, and dry_run (the fields a confirmation sweep is
expected to vary run-to-run without changing what is being measured).

Usage:
    python3 src/pot/summarize.py [path/to/pot_runs.jsonl] [--tag-prefix PREFIX]

Robust to older records that have no "arc" key at all (arc=None), or an "arc"
dict missing "acc_norm" -- those runs are counted (n) but excluded from the
acc_norm stats, and flagged as missing in the printed table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

DEFAULT_PATH = "results/pot/pot_runs.jsonl"

# Fields that vary across a confirmation sweep without changing the arm.
# bpw_budget is an enforcement ceiling, not part of the produced artifact --
# ignoring it lets runs from before/after the C3 revision (4.5 -> 4.25) group
# together as long as everything that actually affects the weights matches.
IGNORE_KEYS = {"calib_offset", "seed", "tag", "out", "dry_run", "bpw_budget"}


def load_records(path: str):
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping malformed line {lineno}", file=sys.stderr)
    return records


def arm_key(config: dict) -> str:
    filtered = {k: v for k, v in (config or {}).items() if k not in IGNORE_KEYS}
    return json.dumps(filtered, sort_keys=True)


def mean_sd_sem(xs):
    n = len(xs)
    if n == 0:
        return None, None, None
    mean = sum(xs) / n
    if n < 2:
        return mean, None, None
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    return mean, sd, sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT_PATH)
    ap.add_argument("--tag-prefix", default=None,
                     help="only include runs whose tag starts with this prefix")
    args = ap.parse_args()

    records = load_records(args.path)
    if args.tag_prefix:
        records = [r for r in records if str(r.get("tag", "")).startswith(args.tag_prefix)]

    groups = defaultdict(list)
    for r in records:
        groups[arm_key(r.get("config") or {})].append(r)

    rows = []
    for key, runs in groups.items():
        tags = [r.get("tag", "?") for r in runs]
        acc_norms = []
        missing_arc = 0
        for r in runs:
            arc = r.get("arc")
            if not arc or "acc_norm" not in arc:
                missing_arc += 1
                continue
            acc_norms.append(arc["acc_norm"])
        bpws = sorted(set(round(r.get("bpw", float("nan")), 4) for r in runs
                          if r.get("bpw") is not None))
        verify_oks = sorted(set(
            (r.get("verify") or {}).get("ok") for r in runs
        ))
        mean, sd, sem = mean_sd_sem(acc_norms)
        rows.append({
            "tags": tags,
            "n_runs": len(runs),
            "n_acc_norm": len(acc_norms),
            "missing_arc": missing_arc,
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "bpw": bpws,
            "verify_ok": verify_oks,
        })

    # widest arm (by acc_norm sample count) first, tag alphabetical as tiebreak
    rows.sort(key=lambda r: (-r["n_acc_norm"], r["tags"][0]))

    fmt = "{:<28} {:>3} {:>10} {:>10} {:>10} {:>16} {:>10}"
    print(fmt.format("arm (first tag)", "n", "mean", "sd", "sem", "bpw", "verify.ok"))
    for r in rows:
        mean_s = f"{r['mean']:.4f}" if r["mean"] is not None else "n/a"
        sd_s = f"{r['sd']:.4f}" if r["sd"] is not None else "n/a"
        sem_s = f"{r['sem']:.4f}" if r["sem"] is not None else "n/a"
        bpw_s = ",".join(f"{b:.4f}" for b in r["bpw"]) if r["bpw"] else "n/a"
        verify_s = ",".join(str(v) for v in r["verify_ok"]) if r["verify_ok"] else "n/a"
        print(fmt.format(r["tags"][0], r["n_acc_norm"], mean_s, sd_s, sem_s, bpw_s, verify_s))
        if r["missing_arc"]:
            print(f"    ({r['missing_arc']} of {r['n_runs']} runs in this arm have no arc.acc_norm, excluded from stats)")
        if r["n_runs"] > 1:
            print(f"    tags: {', '.join(r['tags'])}")


if __name__ == "__main__":
    main()
