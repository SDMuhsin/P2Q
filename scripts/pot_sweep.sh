#!/bin/bash
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad
run(){ python3 -u src/pot/run_pot.py --codebook "$1" --group 128 --nsamples 128 \
        --method "$2" --tasks ppl,arc_easy --tag "$3" > "$S/$3.log" 2>&1; echo "EXIT $3=$?"; }
run strict_pot   gptq HL_strict_pot
run learned_add2 gptq HL_learned_add2
run int4_sym     gptq HL_int4_sym
run strict_pot   rtn  RTN_strict_pot
run learned_add2 rtn  RTN_learned_add2
echo SWEEP_COMPLETE
