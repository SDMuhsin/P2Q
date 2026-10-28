#!/usr/bin/env bash
# Random-permutation seed distribution. Companion to scripts/pot_salience_grouping.sh;
# read that file's pre-registration first.
set -uo pipefail
cd /workspace/CoBALT
COMMON=(--method rtn --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy
        --verify-layers all --codebook strict_pot --group 128
        --group-map default=64,self_attn.v_proj=128 --exact-scale)

# =============================================================================
# APPENDED AFTER THE FIRST FIVE ARMS LANDED (pre-registration above is untouched).
#
# The falsification control WON: `rand` (seed 0) beat the incumbent by +3.54 pp
# acc / +4.59 pp acc_norm, and all three derived keys lost to BOTH the incumbent
# and random. The derived-key hypothesis is dead (see results at the bottom).
# One random draw is a lottery ticket until its distribution is known, so the
# only thing worth running now is more seeds. Decision rule, fixed BEFORE the
# numbers exist:
#   * mean(acc) >= ~0.69 with modest sd  -> real mechanism, new candidate
#   * mean(acc) ~ 0.6654 with sd ~2 pp   -> SG_rand is a lottery ticket, withdraw
# Report mean and sd, never the max.
#
# This runner does NOT abort the queue when one arm dies: GPU 1 was filled by
# another agent mid-run and killed SG_salog_o1, which `set -e` then turned into
# a dead queue. Each arm is retried once.
# =============================================================================
runseed () {
  local s="$1" tag="SG_rand_s$1"
  [ -f "results/pot/logs/${tag}.done" ] && return 0
  local try
  for try in 1 2; do
    CUDA_VISIBLE_DEVICES="${GPU:-1}" \
    python3 -u src/pot/run_pot.py "${COMMON[@]}" --calib-offset 0 \
      --order rand --perm-seed "$s" --tag "$tag" \
      > "results/pot/logs/${tag}.log" 2>&1 && { touch "results/pot/logs/${tag}.done"; return 0; }
    echo "arm $tag failed (attempt $try)"; sleep 60
  done
  return 0
}
for s in 1 2 3 4 5 6 7 8; do runseed "$s"; done
echo ALL_SEEDS_DONE
