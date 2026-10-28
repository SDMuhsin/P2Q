#!/usr/bin/env bash
# THE COMPLETION OF THE DIAGNOSIS: use the Hessian where it helps, not where it hurts.
#
# Established this session:
#   GPTQ uses H for RETARGETING          -> -2.21 pp for strict_pot (+1.21 pp for int4_sym)
#   RTN  discards H ENTIRELY             -> 0.6233, and 0.6364 with an exact scale search
# Neither uses H for the SCALE SEARCH with the feedback switched off. That cell has never
# been run: `hess_weighted_scale` was tested four ways (diag/ldl/invdiag/powdiag) but ALWAYS
# with full-strength GPTQ feedback, where the -2.21 pp of retargeting damage was being done
# on top of it and could easily have masked the scale effect.
#
# WHY IT SHOULD WORK. The scale picks which octave window a group of 64 weights lands in.
# Which columns MATTER for the output is real information and it lives in diag(H). Using it
# to place the window is a robust, low-dimensional use of H -- one statistic per group of 64
# -- whereas retargeting is a per-element use that a log grid cannot absorb. The replication
# result supports this: exact_scale (a better scale, nothing else) is worth +1.30/+1.68 pp on
# PoT vs +0.59 pp on int4, i.e. scale placement matters ~2.5x more on a log grid.
#
# --feedback-damp 0.0 gives zero propagation while leaving H available to the scale search.
# 'ldl' weights by 1/Hinv[j,j]^2, which is what GPTQ's own objective decomposition implies;
# 'diag' weights by diag(H).
#
# THREE OFFSETS, not one. This arm READS calibration data, so unlike the deterministic
# battery it has draw variance and a single screen cannot rank it -- and the ridge sweep
# showed one-draw screening is actively invalid on a log grid (int4 responded smoothly to
# percdamp, PoT scattered). Three draws bound the mean well enough to decide whether a
# full confirm is warranted against the deterministic 0.6364, which has NO error bars.
#
# PRE-REGISTERED: to justify a full confirm, the 3-draw mean must exceed 0.6364 -- the
# deterministic incumbent it has to beat. Anything less and H-in-the-scale is not worth
# reintroducing calibration dependence for, and RTN+exact stands as the deliverable.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-1}
run () { # $1=weight $2=offset
  local tag=HS_$1_off$2
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook strict_pot --group 128 --group-map default=64,self_attn.v_proj=128 \
    --method gptq --feedback-damp 0.0 --scale-weight $1 --exact-scale \
    --nsamples 128 --bpw-budget 4.2501 \
    --tasks ppl,arc_easy --calib-offset $2 --verify-layers all --tag $tag \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_hs.out
}
for off in 0 128 256; do run ldl  $off; done
for off in 0 128 256; do run diag $off; done
echo HS_COMPLETE >> $D/run_hs.out
