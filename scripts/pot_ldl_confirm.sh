#!/bin/bash
# 5-draw CONFIRM of --scale-weight ldl on the current best legal arm (PI_legal's config).
#
# Identical to scripts/pot_confirm.sh except that it passes --bpw-budget 4.2501 rather
# than 4.2500 (CHARTER C3 epsilon note: a hard 4.25 fails EVERY arm including the control,
# because the shared 16-entry codebook costs 1.8e-7 bpw). pot_confirm.sh hardcodes 4.25,
# and argparse would silently take the hardcoded value over one passed in EXTRA_FLAGS,
# so this cannot be done by passing a flag to that script.
set -u
cd /workspace/CoBALT
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES before calling this script}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR="${POT_CONFIRM_LOGDIR:-/tmp/pot_confirm_logs}"
mkdir -p "$LOGDIR"
TAG_PREFIX="$1"; shift
for off in 0 128 256 384 512; do
  tag="${TAG_PREFIX}_off${off}"
  echo "=== $tag ==="
  python3 -u src/pot/run_pot.py "$@" \
    --calib-offset "$off" --tasks ppl,arc_easy --verify-layers all \
    --bpw-budget 4.2501 --tag "$tag" \
    > "$LOGDIR/${tag}.log" 2>&1
  echo "EXIT $tag=$?"
done
echo "CONFIRM_COMPLETE ${TAG_PREFIX}"
