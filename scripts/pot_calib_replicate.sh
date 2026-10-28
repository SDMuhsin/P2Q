#!/bin/bash
# A REAL error bar for the two top arms, from disjoint calibration data.
#
# Why this and not --seed: run_pot.py draws calibration via get_calibration_data, which
# takes contiguous chunks from the head of the tokenised dataset -- deterministic, no
# shuffling, no random offset. So --seed never touches the calibration set; it only seeds
# the random Hadamard sign flips. A "--seed 1" run of a non-rotated arm returns bit-identical
# numbers (verified: S_g64_ao_s1 matched seed 0 to every printed digit) and carries no
# information about data sensitivity.
#
# --calib-offset N skips N sequences before taking --nsamples, so offsets 0/128/256 are
# disjoint 128-sequence draws from the same distribution (verified disjoint).
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

# arm A: g=64 + act-order (4.2500 bpw)
for off in 128 256; do
  python3 -u src/pot/run_pot.py --codebook strict_pot --group 64 --act-order --method gptq \
    --nsamples 128 --calib-offset $off --tasks ppl,arc_easy --tag "R_g64ao_off${off}" \
    > "$S/R_g64ao_off${off}.log" 2>&1
  echo "EXIT R_g64ao_off${off}=$?"
done

# arm B: learned scale residual at g=128 (4.1250 bpw)
for off in 128 256; do
  python3 -u src/pot/run_pot.py --codebook strict_pot --group 128 --method gptq \
    --learn-scale --learn-scope block --nsamples 128 --calib-offset $off \
    --tasks ppl,arc_easy --tag "R_ls128_off${off}" \
    > "$S/R_ls128_off${off}.log" 2>&1
  echo "EXIT R_ls128_off${off}=$?"
done
echo CALIB_REPLICATE_COMPLETE
