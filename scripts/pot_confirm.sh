#!/bin/bash
# 5-draw CONFIRM harness. Per CHARTER C4, a finding only counts once it has been
# measured on all 5 disjoint calibration draws (offsets 0,128,256,384,512) and
# reported as mean/sd/sem -- never on a single screening draw.
#
# Usage:
#   scripts/pot_confirm.sh <tag_prefix> [extra run_pot.py flags...]
#
# Example (the incumbent control):
#   scripts/pot_confirm.sh A_g64ao --codebook strict_pot --group 64 --act-order \
#       --method gptq --nsamples 128
#
# Runs are serial on the assigned GPU (set CUDA_VISIBLE_DEVICES before calling,
# or edit the export below). Each run is tagged "<tag_prefix>_off<N>" and appends
# to results/pot/pot_runs.jsonl. --verify-layers all and --tasks ppl,arc_easy are
# always added; do not pass those flags yourself.
#
# After it finishes: python3 src/pot/summarize.py --tag-prefix <tag_prefix>
set -u
cd /workspace/CoBALT

if [ $# -lt 1 ]; then
  echo "usage: $0 <tag_prefix> [extra run_pot.py flags...]" >&2
  exit 1
fi
TAG_PREFIX="$1"; shift
EXTRA_FLAGS=("$@")

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES before calling this script}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR="${POT_CONFIRM_LOGDIR:-/tmp/pot_confirm_logs}"
mkdir -p "$LOGDIR"

for off in 0 128 256 384 512; do
  tag="${TAG_PREFIX}_off${off}"
  echo "=== $tag ==="
  python3 -u src/pot/run_pot.py "${EXTRA_FLAGS[@]}" \
    --calib-offset "$off" --tasks ppl,arc_easy --verify-layers all --bpw-budget 4.25 \
    --tag "$tag" \
    > "$LOGDIR/${tag}.log" 2>&1
  echo "EXIT $tag=$?"
done
echo "CONFIRM_COMPLETE ${TAG_PREFIX}"
