#!/usr/bin/env bash
# H10 / P3 -- DECISIVE DIRECTION TEST: is the 8-15 pp draw-to-draw ARC-E swing
# PoT-specific, or a property of this model + benchmark?
#
# INT4-symmetric at EXACTLY the incumbent's configuration, five disjoint draws.
# CHARTER C1: int4_sym is run here ONLY as a LABELLED REFERENCE BASELINE. It is
# ILLEGAL as a deliverable and must never be quoted as a result. `cb.shift_free ==
# "none"` sets verify_ok=None, so the shift-free check is correctly skipped rather
# than spuriously failed.
#
# Context: P1b measured PoT as ~0.88x INT4's weight-space flip fragility under BOTH
# a multiplicative and an additive insult -- i.e. PoT is LESS fragile in weight space.
# P3 asks the downstream question that weight space cannot answer.
#
# PRE-REGISTERED:
#   if INT4 within-arm sd << PoT's 0.0366  -> the chaos IS PoT-specific; a variance-
#       reducing, PoT-native change to GPTQ is the right target, despite P1b.
#   if INT4 within-arm sd ~= 0.0366        -> the chaos is a model/benchmark property,
#       NOT the codebook. The quantizer-variance line is dead and draw SELECTION
#       (Agent H) is the only live route to the +4.45 pp oracle headroom.
#   Either outcome is decisive for direction, which is why it is worth 5 runs.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-0}
for off in 0 128 256 384 512; do
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook int4_sym \
    --group 128 \
    --group-map default=64,self_attn.v_proj=128 \
    --act-order \
    --method gptq \
    --nsamples 128 \
    --bpw-budget 4.2501 \
    --tasks ppl,arc_easy \
    --calib-offset $off \
    --verify-layers all \
    --tag R_int4ref_off$off \
    > $D/R_int4ref_off$off.log 2>&1
  echo "$(date +%H:%M:%S) done R_int4ref_off$off rc=$?" >> $D/run_p3.out
done
echo P3_ALL_COMPLETE >> $D/run_p3.out
