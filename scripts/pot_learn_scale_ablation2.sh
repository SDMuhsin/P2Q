#!/bin/bash
# Follow-ups to scripts/pot_learn_scale_ablation.sh, which established (Qwen3-1.7B,
# strict PoT g128, GPTQ, 128x2048 wikitext calib, everything else matched):
#   LSA_base   ppl 23.393  ARC-E acc_norm 0.5349
#   LSA_learn  ppl 17.834  ARC-E acc_norm 0.6709     (block scope, 200 it, lr 5e-3)
# Two questions left: does the cheaper per-linear scope get the same thing, and does the
# block scope keep improving with more Adam steps?
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad
COMMON="--codebook strict_pot --group 128 --nsamples 128 --method gptq \
        --tasks ppl,arc_easy --verify-layers all --learn-scale \
        --learn-retarget none --learn-lr 5e-3 --learn-lambda 0.0"

python3 -u src/pot/run_pot.py $COMMON --learn-scope layer --learn-iters 200 \
        --tag LSA_layer200 > "$S/LSA_layer200.log" 2>&1
echo "EXIT LSA_layer200=$?"
python3 -u src/pot/run_pot.py $COMMON --learn-scope block --learn-iters 800 \
        --tag LSA_block800 > "$S/LSA_block800.log" 2>&1
echo "EXIT LSA_block800=$?"
echo ABLATION2_COMPLETE
