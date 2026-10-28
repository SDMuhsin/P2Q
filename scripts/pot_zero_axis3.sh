#!/usr/bin/env bash
# THIRD REPLICATION AXIS for the zero codebook -- the one lever that is currently
# positive on both the deliverable and the panel.
#
# `strict_pot_zero` is one of exactly two codebooks C1 permits. It replaces the bottom
# rung -- which carries a mean 43x MAGNITUDE INFLATION on ~2.5% of weights, the format's
# only unbounded-relative-error failure mode -- with an exact zero. Measured effect on
# bottom-rung occupancy: 0.0259 -> 0.0173.
#
# Evidence so far, and why a third axis is needed:
#   axis 1, incumbent grouping (g=64 / v_proj 128, exact scale):
#       ARC-E 0.6364 -> 0.6452  (+0.88 pp)   panel_macro 0.5362 -> 0.5380 (+0.18 pp)
#       -- the ONLY arm this session positive on BOTH.
#   axis 2, uniform g=64 grid scale:
#       ARC-E 0.6056 -> 0.6061  (+0.05 pp)   -- positive but negligible.
#   2 of 2 positive contrasts, but SESSION 15.1 measured config-to-config scatter at
#   sd 1.83 pp, so +0.88 pp is inside it and axis 2 does not lift it out.
#
# PRE-REGISTERED, before either arm ran:
#   MECHANISM iff the zero codebook wins on this third, maximally different axis too
#   (uniform g=128, no group map -- twice the group size, different bpw, different
#   clamp regime). 3/3 positive with one clear win is a mechanism; a loss here makes it
#   a principled default with no demonstrated benefit and it is reported as such.
#   SESSION 15.2's rule applies verbatim: choose by principle, report whatever number
#   the chosen config gives, never promote the config that happened to score highest.
set -u
GPU=$1; shift
BASE=(python3 -u src/pot/run_pot.py --method rtn --nsamples 128 --calib-offset 0
      --tasks ppl,arc_easy,mc --verify-layers all --exact-scale --group 128
      --bpw-budget 4.2501)
run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU "${BASE[@]}" "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -14
}
for arm in "$@"; do case $arm in
  pot)  run ZA3_g128_pot  --codebook strict_pot ;;
  zero) run ZA3_g128_zero --codebook strict_pot_zero ;;
  *) echo "unknown $arm"; exit 1 ;; esac; done
