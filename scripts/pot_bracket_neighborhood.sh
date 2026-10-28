#!/usr/bin/env bash
# LOTTERY-vs-MECHANISM CONTROL for the deterministic scale-search results.
#
# The deterministic battery returned (all exact, zero draw variance, RTN base 0.6233):
#     exact    0.6364  (+1.30 pp)  rel_mse 0.028963  <- LOWER mse than base
#     narrow   0.6389  (+1.56 pp)  rel_mse 0.029433  <- HIGHER mse than base
#     wide     0.6288  (+0.55 pp)  rel_mse 0.028967
#     fine     0.6208  (-0.25 pp)  rel_mse 0.028979
#     potscale 0.5248  (-9.85 pp)  rel_mse 0.038235
#
# "Exact" means no MEASUREMENT noise. It does NOT mean the difference is a mechanism.
# Each variant is one artifact drawn from a distribution that this project has repeatedly
# measured as chaotic: artifacts with weight error agreeing to 3-4 significant figures
# routinely sit 3-15 pp apart on ARC-E. Picking the best of six such artifacts is a
# winner's curse, and the deterministic setting removes the noise that would normally
# expose it -- there are no draws to average.
#
# `narrow` is the specific worry. Narrowing the multiplier bracket can only RESTRICT the
# search, so it can only raise MSE -- and it did (0.029433 vs 0.028981) while scoring 1.56
# pp HIGHER. A mechanism that improves the metric by doing a strictly worse job of its own
# objective is exactly what a lottery win looks like.
#
# THE TEST. Scan the bracket neighbourhood. If bracket width is a real mechanism the scores
# vary SMOOTHLY with it and there is a locatable optimum. If it is the artifact lottery the
# scores SCATTER with no relation to width, and `narrow` was simply a lucky ticket.
# PRE-REGISTERED: mechanism <=> monotone-or-single-peaked in width AND |spread| clearly
# exceeding the scatter of neighbouring brackets. Otherwise the deterministic scale-search
# gains are NOT claimed, and only `exact` survives -- it is the one variant that improved
# the metric while also improving its own objective, and it is principled (unbracketed
# global optimum) rather than a tuned hyperparameter.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-1}
run () { # $1=lo $2=hi
  local tag=BR_${1}_${2}
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook strict_pot --group 128 --group-map default=64,self_attn.v_proj=128 \
    --method rtn --nsamples 128 --bpw-budget 4.2501 \
    --scale-lo $1 --scale-hi $2 \
    --tasks ppl,arc_easy --calib-offset 0 --verify-layers all --tag $tag \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_br.out
}
run 0.35 1.15
run 0.40 1.10
run 0.50 1.00
run 0.55 0.95
run 0.60 0.90
run 0.70 0.85
echo BR_COMPLETE >> $D/run_br.out
