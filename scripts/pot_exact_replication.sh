#!/usr/bin/env bash
# Is exact_scale a MECHANISM or another lucky ticket?
#
# The bracket neighbourhood scan showed config-to-config scatter of sd 1.83 pp among
# reasonable scale searches -- larger than exact_scale's +1.30 pp gain over the [0.3,1.2]
# grid. So a single contrast cannot establish it, and `narrow`'s +1.56 pp has already been
# killed by exactly this reasoning (its neighbours scattered 3.2 pp around it).
#
# A deterministic method has no draws to average, so the only way to separate mechanism
# from lottery is REPLICATION ON INDEPENDENT AXES. exact_scale is principled (the exact
# global minimiser of the stated per-group objective, versus a 200-point grid over a
# bracket that has never been varied in 104 runs), so if it is a real mechanism it should
# win WHEREVER it is applied -- on a different grouping, and on a different alphabet.
#
# Two independent replications, both deterministic (--method rtn => H := I => zero draw
# variance, so every number below is exact and the contrasts have no error bars):
#   AXIS 1  uniform g=64, no group-map, no act-order  -> bpw 4.250000, still legal
#   AXIS 2  int4_sym reference baseline (C1: reference only, never a deliverable)
#
# PRE-REGISTERED: exact_scale is a mechanism iff it wins on BOTH new axes. 2 of 3 is not
# enough given sd 1.83 pp scatter -- that is a coin flip. If it loses on either, it is a
# principled default with no demonstrated benefit, and must be reported that way.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-1}
run () { local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --method rtn --nsamples 128 --bpw-budget 4.2501 \
    --tasks ppl,arc_easy --calib-offset 0 --verify-layers all --tag $tag "$@" \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_rep.out
}
# AXIS 1: uniform g=64, no group map, no act-order (16/64 = 0.2500 exactly)
run REP_g64u_grid   --codebook strict_pot --group 64
run REP_g64u_exact  --codebook strict_pot --group 64 --exact-scale
# AXIS 2: int4_sym, incumbent grouping  (reference baseline only)
run REP_int4_grid   --codebook int4_sym --group 128 --group-map default=64,self_attn.v_proj=128
run REP_int4_exact  --codebook int4_sym --group 128 --group-map default=64,self_attn.v_proj=128 --exact-scale
# and the deliverable candidate itself, act-order dropped (provably a no-op under H := I)
run REP_deliverable --codebook strict_pot --group 128 --group-map default=64,self_attn.v_proj=128 --exact-scale
echo REP_COMPLETE >> $D/run_rep.out
