#!/usr/bin/env bash
# THE DIRECT TEST OF THE OVERFITTING CLAIM -- which I have been asserting without testing.
#
# CLAIM (from the 2x2): GPTQ hurts strict_pot (-2.21 pp) and helps int4_sym (+1.21 pp)
# because it trades weight-space fidelity for fit to a Hessian estimated from only 128
# sequences, and the size of that trade scales with grid coarseness (delta rel_mse ratio
# 2.39, draw-variance ratio 2.39).
#
# If that is STATISTICAL overfitting, it is fixable: H's sampling error falls as 1/nsamples,
# so more calibration data should shrink the harm and may flip GPTQ's sign for PoT --
# recovering the +1.2 pp that H is demonstrably worth on a uniform grid.
# If it is STRUCTURAL -- the retargeting itself being unabsorbable by a log grid, because
# moving a weight to a larger magnitude moves it somewhere coarser -- then no amount of data
# helps and RTN-like is genuinely right.
#
# These give opposite guidance for every future PoT quantizer, and everything else I ran
# this session probed AROUND this question rather than AT it.
#
# The baseline needs no runs: RTN does not read calibration data at all, so its score is
# constant in nsamples -- 0.6233 (grid) / 0.6364 (exact), both exact, zero variance.
# GPTQ-RTN can therefore be read off directly at each nsamples.
#
# wikitext103 is used because wikitext2 holds only 1229 sequences (nsamples 1024 needs
# 1024+offset). It is charter-legal and measured indistinguishable from wikitext2
# (0.6103 vs 0.6012, t = 0.58). Two offsets per setting: this arm reads calibration data, so
# one draw cannot rank it, and 2 draws at least bounds the trend.
#
# PRE-REGISTERED:
#   STATISTICAL  => GPTQ - RTN rises monotonically with nsamples and approaches/crosses 0.
#   STRUCTURAL   => GPTQ - RTN stays flat and negative regardless of nsamples.
#   The int4 arm at nsamples 1024 is the control: it should stay positive throughout, and if
#   it too degrades then something about large-nsamples runs is confounded and neither
#   reading holds.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-0}
run () { # $1=codebook $2=nsamples $3=offset
  local tag=NS_$1_n$2_off$3
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook $1 --group 128 --group-map default=64,self_attn.v_proj=128 \
    --method gptq --nsamples $2 --bpw-budget 4.2501 \
    --calib-dataset wikitext103 --calib-offset $3 \
    --tasks ppl,arc_easy --verify-layers all --tag $tag \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_ns.out
}
for off in 20000 22000; do
  run strict_pot 128  $off
  run strict_pot 512  $off
  run strict_pot 1024 $off
done
run int4_sym 1024 20000     # control
echo NS_COMPLETE >> $D/run_ns.out
