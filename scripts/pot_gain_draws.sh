#!/usr/bin/env bash
# DRAW SENSITIVITY OF --out-correct gain.
#
# This matters more than it looks. The incumbent's defining virtue is that `--method rtn`
# reads NO calibration data (`H := I`), so an arm is ONE exact artifact and one exact
# number -- SESSION 13 eliminated the draw lottery that produced three sessions of
# promote-then-withdraw. `--out-correct gain` fits a per-output-row gain against the
# calibration Gram and mean, so it REINTRODUCES calibration dependence and therefore a
# draw lottery. Its +3.24 / +6.73 pp could be a favourable draw.
#
# wikitext2 supplies 9 disjoint draws at nsamples=128 (SESSION 12.4). Three are run here.
#
# PRE-REGISTERED, before any arm ran:
#   * The incumbent it must beat is 0.6364 on ARC-Easy, and that number has NO error bar
#     because it is deterministic -- so the comparison is the 3-draw MEAN of gain against
#     a point.
#   * PASS iff the 3-draw mean ARC-E exceeds 0.6364 by more than the per-draw sd, AND
#     the 3-draw core_6 mean does not fall below the incumbent's 0.4973 by more than its
#     1.15 pp scatter.
#   * If the draws span more than ~4 pp, the correction has simply traded a deterministic
#     artifact for a lottery one, and that is a COST that must be reported alongside the
#     mean -- a 0.6688 that is one draw of a 6 pp-wide distribution is not an improvement
#     on a deterministic 0.6364, and this project has promoted exactly that mistake twice.
set -u
GPU=$1; shift
run () { local tag=$1; local off=$2
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag (offset $off) ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py --method rtn --nsamples 128 \
     --calib-offset "$off" --bpw-budget 4.2501 --tasks ppl,arc_easy,mc --verify-layers all \
     --exact-scale --codebook strict_pot --group 128 \
     --group-map default=64,self_attn.v_proj=128 --out-correct gain --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -10
}
for arm in "$@"; do case $arm in
  off128) run OC_gain_o128 128 ;;
  off256) run OC_gain_o256 256 ;;
  off384) run OC_gain_o384 384 ;;
  *) echo "unknown $arm"; exit 1 ;; esac; done
