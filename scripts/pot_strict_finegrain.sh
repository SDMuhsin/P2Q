#!/bin/bash
# Strict power-of-two needs finer groups: g=128 costs 4.125 bpw but drops ARC-E to 53.5%.
# g=64 -> 4.25 bpw, g=32 with an int8 PoT scale -> 4.25 bpw. Both inside the 4.5 ceiling.
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad
python3 -u src/pot/run_pot.py --codebook strict_pot --group 64 --nsamples 128 --method gptq \
  --tasks ppl,arc_easy --tag HL_strict_g64 > "$S/HL_strict_g64.log" 2>&1; echo "EXIT g64=$?"
python3 -u src/pot/run_pot.py --codebook strict_pot --group 32 --nsamples 128 --method gptq \
  --scale-bits 8 --tasks ppl,arc_easy --tag HL_strict_g32 > "$S/HL_strict_g32.log" 2>&1; echo "EXIT g32=$?"
python3 -u src/pot/run_pot.py --codebook learned_add2 --group 64 --nsamples 128 --method gptq \
  --tasks ppl,arc_easy --tag HL_add2_g64 > "$S/HL_add2_g64.log" 2>&1; echo "EXIT add2g64=$?"
echo FINEGRAIN_COMPLETE
