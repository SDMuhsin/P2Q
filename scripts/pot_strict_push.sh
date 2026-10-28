#!/bin/bash
# STRICT single-power-of-two only. Every arm <= 4.5 bpw fully accounted.
#
# Diagnosis driving this queue: a log grid has constant RELATIVE resolution, so a weight
# carries ~4% expected squared relative error wherever it lands. The scale search is well
# inside its range (median mult 0.918, 0.01% pinned at floor) and no codes are dead
# (usage 2.7/2.7/5.3/10.4/19.1/28.7/24.7/6.4 % across |2^0|..|2^-7|). So the codebook and
# the scale are NOT the problem, and finer groups only re-centre an already-wide window.
# What is left is making the error COMPENSATION work harder, plus one codebook question.
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad
r(){ tag=$1; shift; python3 -u src/pot/run_pot.py --nsamples 128 --method gptq \
       --tasks ppl,arc_easy --tag "$tag" "$@" > "$S/$tag.log" 2>&1; echo "EXIT $tag=$?"; }

r S_g128_ao     --codebook strict_pot      --group 128 --act-order
r S_g64_ao      --codebook strict_pot      --group 64  --act-order
r S_g64_ao_hw   --codebook strict_pot      --group 64  --act-order --hess-weighted-scale
r S_g64_ao_zero --codebook strict_pot_zero --group 64  --act-order
r S_g32_ao      --codebook strict_pot      --group 32  --scale-bits 8 --act-order
echo PUSH_COMPLETE
