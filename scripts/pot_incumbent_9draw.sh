#!/usr/bin/env bash
# Exhaust wikitext2's calibration supply for the incumbent.
#
# WHY. PI_legal's headline is 0.6141 +- 0.0164 (sem) from FIVE draws. Its 95% CI is
# [0.5820, 0.6462] -- the CHARTER target 0.6400 lies INSIDE it. We do not currently know
# whether the incumbent already meets the target. Agent G's instrument study puts the
# 5-draw detection limit at 5.72 pp on `acc`, far coarser than the 2.59 pp gap to target.
#
# wikitext2 yields 1229 sequences of 2048, so at --nsamples 128 it supplies exactly
# floor(1229/128) = 9 fully disjoint draws. Five are spent; these are the remaining four.
# Same corpus, same config, no method change -- this only measures what we already have,
# more precisely. Nine draws narrow the CI half-width from 0.0321 to 0.0239.
#
# NOT a target-hunt: the pre-registered reading is symmetric. If the 9-draw mean lands
# below 0.6400 that is the answer and it gets reported; the point is that the current
# protocol cannot distinguish 0.582 from 0.646 and every comparison in this project has
# been made inside that interval.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-0}
for off in 640 768 896 1024; do
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook strict_pot \
    --group 128 \
    --group-map default=64,self_attn.v_proj=128 \
    --act-order \
    --method gptq \
    --nsamples 128 \
    --bpw-budget 4.2501 \
    --tasks ppl,arc_easy \
    --calib-offset $off \
    --verify-layers all \
    --tag PI_legal_off$off \
    > $D/PI_legal_off$off.log 2>&1
  echo "$(date +%H:%M:%S) done PI_legal_off$off rc=$?" >> $D/run_9draw.out
done
echo NINE_DRAW_COMPLETE >> $D/run_9draw.out
