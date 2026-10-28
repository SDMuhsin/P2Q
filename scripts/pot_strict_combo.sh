#!/bin/bash
# TOP PRIORITY: the untested combination. Strict single-power-of-two only.
#
# Three interventions have each been measured ALONE, and two of them turned out to
# interact strongly, so no main effect here should be trusted to add:
#
#   config                          ppl     acc     acc_norm   bpw
#   g=128 baseline                 23.39   0.5901   0.5349    4.1250
#   g=128 + rotation               17.88   0.6393   0.5930    4.1250   (+5.8 pp)
#   g=128 + act-order              18.47   0.6258   0.5804    4.1250   (+4.6 pp)
#   g=64  + act-order              17.75   0.6797   0.6574    4.2500   (best so far)
#   g=64  alone                    19.99   0.5438   0.5067    4.2500   (worst; interacts)
#
# Rotation has never been run WITH act-order, nor at g=64. That is the gap.
# References: bf16 16.72 / .7264 / .7050 ; RTN-INT4 (not shift-free) 20.35 / .6780 / .6696
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

r(){ tag=$1; shift
     python3 -u src/pot/run_pot.py --codebook strict_pot --method gptq --nsamples 128 \
       --tasks ppl,arc_easy --tag "$tag" "$@" > "$S/$tag.log" 2>&1
     echo "EXIT $tag=$?"; }

# the one that matters: all three together, at the winning group size
r C_g64_ao_rot   --group 64  --act-order --rotate random_hadamard
# same at g=128, to separate the rotation x act-order interaction from the group size
r C_g128_ao_rot  --group 128 --act-order --rotate random_hadamard
# finest affordable grouping with everything on. --pot-scale is REQUIRED here, not optional:
# --scale-bits alone is accounting-only and does not constrain the stored scale, so charging
# 8 bits without it would claim 4.25 bpw for a run that is really 4.5000002 bpw (over budget).
r C_g32_ao_rot   --group 32  --scale-bits 8 --pot-scale --act-order --rotate random_hadamard
echo COMBO_COMPLETE

# Legitimate g=32: --pot-scale actually CONSTRAINS the group scale to a power of two, so
# the 8-bit charge is real (4 + 8/32 + codebook = 4.2500 bpw). The earlier S2_g32_ao
# charged 8 bits without constraining anything and was really 4.5000002 bpw -- over budget.
r G32_potscale --group 32 --scale-bits 8 --pot-scale --act-order
