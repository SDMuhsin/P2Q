#!/bin/bash
# Replicate the HEADLINE configuration across calibration draws.
#
# S_g64_ao (g=64 + act-order, strict PoT, 4.2500 bpw, acc 0.6797 / acc_norm 0.6574) is the
# result that will be quoted, and it currently rests on a single calibration draw (seed 0).
# A seed replication of rot_rh_notie gave per-run SD ~2.0 pp on acc and ~0.9 pp on acc_norm
# from one pair -- comparable to the ~1.0 pp binomial evaluation SE. So the parity claim
# needs its own replication more than any further ablation does.
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

for sd in 1 2; do
  python3 -u src/pot/run_pot.py --codebook strict_pot --group 64 --act-order --method gptq \
    --nsamples 128 --seed $sd --tasks ppl,arc_easy --tag "S_g64_ao_s${sd}" \
    > "$S/S_g64_ao_s${sd}.log" 2>&1
  echo "EXIT S_g64_ao_s${sd}=$?"
done
echo REPLICATE_COMPLETE
