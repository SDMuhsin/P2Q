#!/usr/bin/env bash
# PoT-native GPTQ, arm Z: strict_pot_zero(7) -- spends one code on an exact zero,
# buying it with one octave of dynamic range. Same 4 bits/weight, same group map,
# same bpw as PI_legal: bpw-neutral by construction.
# Full 5-draw CONFIRM per CHARTER C4. Two GPUs, offsets split across them.
set -u
cd /workspace/CoBALT
run () {  # $1=gpu $2=offset
  CUDA_VISIBLE_DEVICES=$1 python3 -u src/pot/run_pot.py \
    --codebook strict_pot_zero \
    --group-map default=64,self_attn.v_proj=128 --act-order --method gptq \
    --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy \
    --calib-offset $2 --verify-layers all --tag ZC_off$2 \
    > /tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative/ZC_off$2.log 2>&1
  echo "done off$2 rc=$?"
}
( for o in 0 256 512; do run 0 $o; done ) &
( for o in 128 384;   do run 1 $o; done ) &
wait
echo ALL_DONE
