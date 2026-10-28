#!/bin/bash
# Round 2, strict single-power-of-two only, centred on the WINNING configuration.
#
# Round 1 result that reset the plan: g=64 and act-order INTERACT strongly and are not
# interpretable alone.
#   g=64, no act-order  -> acc_norm 0.5067  (worst)
#   g=128, act-order    -> acc_norm 0.5804
#   g=64,  act-order    -> acc_norm 0.6574  (best; +15.1 pp over g=64 alone)
# So "finer groups hurt" was conditional on act-order being off. Everything below keeps
# act-order ON and g=64 as the base, and varies one thing at a time from there.
#
# Reference points: bf16 16.72 / acc .7264 / acc_norm .7050
#                   RTN-INT4-asym (NOT shift-free, 4.188 bpw) 20.35 / .6780 / .6696
#                   best strict so far (4.250 bpw) 17.75 / .6797 / .6574
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

r(){ tag=$1; shift
     python3 -u src/pot/run_pot.py --method gptq --act-order \
       --tasks ppl,arc_easy --tag "$tag" "$@" > "$S/$tag.log" 2>&1
     echo "EXIT $tag=$?"; }

# finest affordable grouping, same 4.25 bpw as g=64 (int8 PoT scale pays for g=32)
r S2_g32_ao       --codebook strict_pot      --group 32  --scale-bits 8 --nsamples 128
# scale objective retargeted from weight MSE (measured anti-correlated with the criterion)
# to output error, now at the winning config
r S2_g64_ao_hw    --codebook strict_pot      --group 64  --nsamples 128 --hess-weighted-scale
# does a code spent on exact zero beat the 8th magnitude level, at the winning config?
r S2_g64_ao_zero  --codebook strict_pot_zero --group 64  --nsamples 128
# more calibration -> better-conditioned Hessian -> better compensation
r S2_g64_ao_ns256 --codebook strict_pot      --group 64  --nsamples 256
# damping: on a 2x-spaced grid residuals are large, so 0.01 may be over-damped
r S2_g64_ao_d003  --codebook strict_pot      --group 64  --nsamples 128 --percdamp 0.003
echo PUSH2_COMPLETE
