#!/bin/bash
# Round 1b, strict single-power-of-two only. Re-aimed after the round-1 finding that
# WEIGHT MSE IS ANTI-CORRELATED with the acceptance criterion in both directions measured:
#
#   --group 64  : weight rel_mse 0.04574 -> 0.04212 (better), ARC-E acc_norm -2.82 pp (worse)
#   --act-order : weight rel_mse 0.04574 -> 0.04984 (worse),  ARC-E acc_norm +4.55 pp (better)
#
# Our per-group scale search minimises plain weight MSE -- exactly the anti-correlated
# quantity. So the highest-value remaining knob is retargeting that objective at OUTPUT
# error (--hess-weighted-scale), tested at our BEST config (g=128 + act-order), not at
# g=64 which we already know hurts the criterion.
#
# act-order is ON everywhere: it is free (+4.55 pp acc_norm at identical 4.1250 bpw).
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

r(){ tag=$1; shift
     python3 -u src/pot/run_pot.py --method gptq --act-order --nsamples 128 \
       --tasks ppl,arc_easy --tag "$tag" "$@" > "$S/$tag.log" 2>&1
     echo "EXIT $tag=$?"; }

# THE one to watch: scale search retargeted from weight MSE to output error
r S_g128_ao_hw   --codebook strict_pot      --group 128 --hess-weighted-scale
# does a code spent on exact zero beat the 8th magnitude level, at our best config?
r S_g128_ao_zero --codebook strict_pot_zero --group 128
# second point on the scale-side prediction, at the finest affordable grouping (4.25 bpw)
r S_g32_ao       --codebook strict_pot      --group 32 --scale-bits 8
echo PUSH1B_COMPLETE
