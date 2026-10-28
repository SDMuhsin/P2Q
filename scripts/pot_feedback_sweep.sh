#!/usr/bin/env bash
# PoT-native GPTQ: sweep the error-feedback strength.
#
# Measured 2x2 at identical bpw 4.246867:
#                 strict_pot           int4_sym
#     GPTQ        0.6012 (9 draws)     0.6603 (5 draws)
#     RTN         0.6233 (exact)       0.6481 (exact)
# GPTQ's feedback is worth +1.21 pp on a uniform grid and -2.21 pp on a log grid.
#
# WHY. GPTQ retargets later columns to absorb this column's residual, raising weight error
# ~60% in both formats to buy output error. On a uniform grid every retarget is equally
# representable (error <= Delta/2 wherever you move it) so the trade pays. On a LOG grid the
# representable step at v is ~|v|/3.5, so moving a weight to a larger magnitude moves it to
# a COARSER part of the grid. GPTQ is grid-blind and therefore overspends representability
# it cannot afford. lambda=1 (stock) and lambda=0 (no propagation) are both already
# measured; this locates the interior.
#
# --feedback-clip is the PoT-native half: cap each element's correction at kappa * |w|/3.5,
# i.e. never ask a weight to move further than its own grid can follow. The cap tracks |w|
# precisely because a log grid's resolution tracks |w| -- there is no uniform-grid analogue.
#
# SCREEN ONLY (C4): one draw, offset 0, generates CANDIDATES. Ranks nothing. The winner
# goes to a 5- or 9-draw confirm before any claim is made.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-0}
run () { # $1=tag $2...=extra flags
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook strict_pot --group 128 --group-map default=64,self_attn.v_proj=128 \
    --act-order --method gptq --nsamples 128 --bpw-budget 4.2501 \
    --tasks ppl,arc_easy --calib-offset 0 --verify-layers all --tag $tag "$@" \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_fb.out
}
run FB_damp0.00 --feedback-damp 0.00
run FB_damp0.25 --feedback-damp 0.25
run FB_damp0.50 --feedback-damp 0.50
run FB_damp0.75 --feedback-damp 0.75
run FB_clip0.25 --feedback-clip 0.25
run FB_clip0.50 --feedback-clip 0.50
run FB_clip1.00 --feedback-clip 1.00
echo FB_SWEEP_COMPLETE >> $D/run_fb.out
