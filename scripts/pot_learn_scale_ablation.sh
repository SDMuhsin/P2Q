#!/bin/bash
# PoT-PTQ step 2 (the learned scale residual, src/pot/scale_learn.py) -- matched ablation.
#
# Two arms, identical in every other respect, so the contribution of step 2 is isolated:
#   LSA_base   strict PoT + GPTQ g128                       (= the current headline)
#   LSA_learn  the same, then Gamma learned by block-output reconstruction
#
# Hyperparameters come from the single-layer probe on Qwen3-1.7B layer 0 (block relative
# output error, GPTQ baseline 0.015453):
#   scope=block  retarget=none  lr=5e-3  lam=0   200 it -> 0.8355x   (2000 it -> 0.8226x)
#   scope=layer  retarget=none                          -> 0.9676x
#   retarget=gptq/orig (re-round the codes as the scale moves)  -> 1.6x to 4.8x WORSE
#   lam=1e-2 -> 0.984x (the penalty simply switches the learning off); learn_col -> no gain
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad
COMMON="--codebook strict_pot --group 128 --nsamples 128 --method gptq \
        --tasks ppl,arc_easy --verify-layers all"

python3 -u src/pot/run_pot.py $COMMON --tag LSA_base  > "$S/LSA_base.log"  2>&1
echo "EXIT LSA_base=$?"
python3 -u src/pot/run_pot.py $COMMON --tag LSA_learn --learn-scale \
        --learn-scope block --learn-retarget none --learn-iters 200 \
        --learn-lr 5e-3 --learn-lambda 0.0 > "$S/LSA_learn.log" 2>&1
echo "EXIT LSA_learn=$?"
echo ABLATION_COMPLETE
