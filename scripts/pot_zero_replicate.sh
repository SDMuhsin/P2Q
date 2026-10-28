#!/usr/bin/env bash
# =============================================================================
# Is the ZERO level a MECHANISM, or another lucky ticket?
#
# ZP_zero_exact beat the incumbent at IDENTICAL bpw, deterministically:
#   incumbent  strict_pot        acc 0.6654  acc_norm 0.6364  ppl 19.92  bpw 4.244792
#   ZP_zero    strict_pot_zero   acc 0.6696  acc_norm 0.6452  ppl 19.72  bpw 4.244792
# i.e. +0.42 pp acc, +0.88 pp acc_norm, verify.ok on both.
#
# That is INSIDE the project's measured config-to-config scatter of sd 1.83 pp, so a
# single contrast cannot establish it -- exactly the reasoning that killed `narrow`
# (+1.56 pp, withdrawn after its neighbourhood scattered 3.2 pp) and that validated
# `--exact-scale` (6/6 positive across three independent axes).
#
# It is, however, a DERIVED prediction rather than a found config: §20 says upward
# clamping of small weights is the dominant failure mode, and a zero level replaces a
# 7.46x inflation with an exact zero. So it earns a replication, on the same three-axis
# protocol used for --exact-scale.
#
# PRE-REGISTERED: the zero level is a mechanism iff it wins on BOTH new axes, on acc.
# Two of three is a coin flip at sd 1.83 pp and must be reported as such.
#   AXIS 1  uniform g=64, no group-map          (bpw 4.250000 both arms)
#   AXIS 2  group 32 via group-map              (a third, independent grouping)
# All arms --method rtn => H := I => draw-independent and exactly reproducible; one run
# per arm IS the answer.
#
# NOTE ON THE TRADE, so the result is read correctly either way: strict_pot_zero at 4
# bits is 15 levels = 7 octaves + zero, versus strict_pot's 16 levels = 8 octaves and no
# zero. Both charge ceil(log2 n) = 4.000000 bits. So this BUYS a zero and PAYS an octave
# of range -- and §20 says range is load-bearing. The two effects oppose, which is why
# the sign is not obvious and why replication is the only way to tell.
# =============================================================================
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run () { local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --method rtn --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy \
    --calib-offset 0 --verify-layers all --exact-scale --tag $tag "$@" \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_zr.out
}
# AXIS 1: uniform g=64, no group map
run ZR_a1_pot   --codebook strict_pot      --group 64
run ZR_a1_zero  --codebook strict_pot_zero --group 64
# AXIS 2: uniform g=32 (independent grouping; 16/32 = 0.5 bpw scale, still legal at 4.5?
# no -- g=32 costs 0.5 bpw => 4.5 total, OVER budget. Use g=128 instead: 0.125 bpw.)
run ZR_a2_pot   --codebook strict_pot      --group 128
run ZR_a2_zero  --codebook strict_pot_zero --group 128
echo ZR_COMPLETE >> $D/run_zr.out
