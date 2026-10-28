"""Block until a GPU has enough free memory, then print its index.

Why this exists
---------------
This box's two A40s are shared with other tenants, and on 2026-09-12 every arm of the
pre-registered `--solve-scale` validation died of `torch.OutOfMemoryError` -- not of a
negative result. Three arms died inside `pot.scale_solve` and one inside the panel
evaluator, with the OOM messages naming five or six foreign PIDs holding 28-30 GB.

Two separate fixes were needed and this is the second:

  * `pot.scale_solve` now plans its row block from free VRAM and halves on OOM. That
    makes the SOLVE fit in whatever is available, and it is exact at any block size.
  * The EVALUATOR must not be made to fit. `pot.eval_mc` is a ranking instrument whose
    batch packing is deliberately a constant of the code -- shrinking a batch to dodge
    an OOM would perturb scores by ~0.4 pp (the measured batching noise floor), which
    is larger than the effects being ranked. So the evaluator is never adapted; the
    JOB is held back until the memory it needs actually exists.

Hence a gate, not a retry. A job admitted here still has no exclusive claim -- another
tenant can allocate at any moment -- so `--headroom` asks for materially more than the
job needs rather than the bare minimum.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def gpu_free_mb():
    """[(index, free_mb)] for every visible device, via nvidia-smi.

    nvidia-smi is used rather than torch so that the gate costs no CUDA context: a
    context is ~300 MB per process, which on a full GPU is the difference between
    admitting a job and not.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    res = []
    for line in out.strip().splitlines():
        idx, total, used = (int(x.strip()) for x in line.split(","))
        res.append((idx, total - used))
    return res


def wait_for_gpu(need_mb, timeout_s=None, poll_s=60, allow=None, stable=2, quiet=False):
    """Return the index of a GPU with >= need_mb free, or None on timeout.

    `stable` consecutive polls must agree before a device is admitted, so a job is not
    launched into the brief trough between two of a neighbour's allocations.
    """
    t0 = time.time()
    streak = {}
    while True:
        free = gpu_free_mb()
        if allow is not None:
            free = [(i, f) for i, f in free if i in allow]
        best = max(free, key=lambda t: t[1], default=None)
        for i, f in free:
            streak[i] = streak.get(i, 0) + 1 if f >= need_mb else 0
            if streak[i] >= stable:
                return i
        if timeout_s is not None and time.time() - t0 > timeout_s:
            return None
        if not quiet:
            waited = int(time.time() - t0)
            have = f"{best[1]} MB free on GPU{best[0]}" if best else "no visible GPU"
            print(f"[gpu-gate] want {need_mb} MB, {have}; waited {waited}s",
                  file=sys.stderr, flush=True)
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--need-mb", type=int, required=True,
                    help="free VRAM the job needs, INCLUDING headroom for the other "
                         "tenants on a shared device")
    ap.add_argument("--timeout-s", type=int, default=None)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--stable", type=int, default=2,
                    help="consecutive polls a device must stay free before admission")
    ap.add_argument("--allow", default=None,
                    help="comma-separated GPU indices to consider (default: all)")
    args = ap.parse_args()

    allow = None
    if args.allow:
        allow = {int(x) for x in args.allow.split(",") if x.strip() != ""}
    idx = wait_for_gpu(args.need_mb, timeout_s=args.timeout_s, poll_s=args.poll_s,
                       allow=allow, stable=args.stable)
    if idx is None:
        print("[gpu-gate] TIMEOUT: no device ever had enough free memory",
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    print(idx)


if __name__ == "__main__":
    main()
