#!/usr/bin/env bash
# REPLICATION OF --out-correct gain, the ladder's finest rung.
#
# OC_gain on the incumbent axis: ARC-E acc_norm 0.6364 -> 0.6688 (+3.24 pp),
# acc 0.6654 -> 0.6902, threshold-free auc_raw 0.8188 -> 0.8314 (+1.26 pp),
# bpw UNCHANGED at 4.244792, verify.ok on all 196 linears, every code bit-identical.
# Panel: -0.25 pp (arc_c +1.54, piqa -0.98, winogrande -1.18, rest ~flat).
#
# It also falsified the prediction I derived from SESSION 22.2 -- that fitting MORE
# parameters against the calibration Gram would be monotonically worse. The ladder is
# 1 param: -0.04 pp | 196: -0.38 pp | ~40k: +2.48 pp on acc. The most H-fitted rung wins.
#
# PRE-REGISTERED, before either arm ran:
#   Two independent axes, each a same-axis contrast against a baseline ALREADY IN THE LOG:
#     axis A  g=128 uniform, exact scale   -> baseline ZA3_g128_pot   (ARC-E 0.5947)
#     axis B  strict_pot_zero, incumbent grouping -> baseline PN_zero (ARC-E 0.6452)
#   MECHANISM iff gain wins on BOTH, with at least one win clearly outside the 1.15 pp
#   core_6 config scatter on ARC-E. 1 of 2, or two sub-scatter wins, means it is another
#   lottery ticket of the SG_rand / strict_pot_zero kind and is reported as such and
#   NOT promoted -- the same rule that just withdrew 0.6452.
#   A win on both REOPENS the H-fitting question that SESSION 13/22 closed, which is the
#   more important consequence either way.
set -u
GPU=$1; shift
run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py --method rtn --nsamples 128 \
     --calib-offset 0 --bpw-budget 4.2501 --tasks ppl,arc_easy,mc --verify-layers all \
     --exact-scale "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -12
}
for arm in "$@"; do case $arm in
  a_g128) run OG_g128_gain --codebook strict_pot --group 128 --out-correct gain ;;
  b_zero) run OG_zero_gain --codebook strict_pot_zero --group 128 \
                --group-map default=64,self_attn.v_proj=128 --out-correct gain ;;
  c_int4) run OG_int4_gain --codebook int4_sym --group 128 \
                --group-map default=64,self_attn.v_proj=128 --out-correct gain ;;
  *) echo "unknown $arm"; exit 1 ;; esac; done
