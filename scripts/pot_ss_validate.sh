#!/usr/bin/env bash
# VALIDATION OF --solve-scale, the session's leading candidate.
#
# SS_mu1 vs the incumbent, both at bpw 4.244792, verify.ok on all 196 linears, every
# code bit-identical:
#     ARC-E 0.6364 -> 0.6431 (+0.67 pp)     panel_7 0.5362 -> 0.5414 (+0.52 pp)
#     core_6 0.4973 -> 0.5027 (+0.54 pp)    ppl 19.921 -> 17.543 (-2.38)
# It is the only arm tonight positive on ALL of ARC-Easy, the panel, core_6 and ppl, and
# it has the highest panel_macro of any arm measured -- above int4_sym's 0.5354.
#
# Two reasons it is not promotable yet, and this script tests both:
#
#  (a) REPLICATION. Every headline this project has promoted from one axis has been
#      withdrawn (0.6574, 0.6839, SG_rand 0.6822, strict_pot_zero 0.6452). Axis B is
#      g=128 uniform, whose plain baseline ZA3_g128_pot (0.5947 / 0.5207 / 0.4846) is
#      already in the log.
#  (b) DRAW SENSITIVITY. The incumbent reads NO calibration data -- `--method rtn` sets
#      H := I -- so 0.6364 is one exact number with no error bar. --solve-scale fits
#      ~22M scales against the calibration Gram, which REINTRODUCES the draw lottery
#      SESSION 13 eliminated. A 0.6431 that is one draw of a wide distribution is not an
#      improvement on a deterministic 0.6364.
#
# PRE-REGISTERED, before any arm ran:
#   * PASS iff axis B is also positive on core_6, AND the 3-draw ARC-E mean stays above
#     the incumbent's 0.6364, AND the 3-draw core_6 mean stays above 0.4973.
#   * The per-draw SPREAD is reported alongside the mean whatever it shows. If the draws
#     span more than ~4 pp on ARC-E, the method has traded a deterministic artifact for a
#     lottery one and that is a COST, reported as such, not hidden behind a mean.
#   * A win on ARC-Easy alone, with core_6 flat or negative, is NOT promoted -- the rule
#     that correctly withdrew strict_pot_zero tonight.
set -u
GPU=$1; shift
run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py --method rtn --nsamples 128 \
     --bpw-budget 4.2501 --tasks ppl,arc_easy,mc --verify-layers all --exact-scale \
     --codebook strict_pot "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -10
}
GM=(--group 128 --group-map default=64,self_attn.v_proj=128)
for arm in "$@"; do case $arm in
  axisB)  run SS_g128_mu1 --calib-offset 0   --group 128 --solve-scale 1.0 ;;
  o128)   run SS_mu1_o128 --calib-offset 128 "${GM[@]}" --solve-scale 1.0 ;;
  o256)   run SS_mu1_o256 --calib-offset 256 "${GM[@]}" --solve-scale 1.0 ;;
  zero)   run SS_zero_mu1 --calib-offset 0   "${GM[@]}" --solve-scale 1.0 --codebook strict_pot_zero ;;
  *) echo "unknown $arm"; exit 1 ;; esac; done
