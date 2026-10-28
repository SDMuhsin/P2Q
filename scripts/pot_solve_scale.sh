#!/usr/bin/env bash
# EXACT RE-SOLVE OF THE GROUP SCALES, CODES FROZEN  (src/pot/scale_solve.py)
#
# The argument, from this session's evidence rather than from theory:
#   Everything that FAILS on this format moves weights to new codewords --
#     GPTQ            ARC-E -7.91 pp, panel -1.49 pp   (measured today, panel CONFIRMS it)
#     refine_codes    ARC-E -6.51 pp
#     rotation        ARC-E -32.45 pp
#   The one H-fitted thing that does NOT fail leaves every code bit-identical and only
#   rescales: `--out-correct gain`, ARC-E +3.24 pp at zero extra bits.
# So: the codes RTN picks are right; the SCALES are what is left to optimise. And the
# scales are ALREADY STORED -- re-choosing their values costs nothing.
#
# `--out-correct gain` explores a 1-D slice (one multiplier per row). The full problem is
# exact: the layer output is LINEAR in a row's G group-scales, so minimising the
# calibration-weighted output error is a G x G normal equation per row. Measured on layer
# 0, mu=1.0: output error -13% to -47% per matrix, scales moving only 0.033-0.046
# octaves, bpw unchanged, verify.ok.
#
# PRE-REGISTERED, before any arm ran:
#   * This is the project's most aggressive H-fit yet (~22M free parameters against a
#     Gram from 128 x 2048 tokens), and overfitting the calibration Hessian is its most
#     reliable failure mode. The ridge mu is the guard: mu=inf is a PROVABLE no-op, so
#     the sweep has a known-safe endpoint and any collapse must be monotone in mu.
#   * PASS iff some mu beats the incumbent (ARC-E 0.6364, panel_macro 0.5362, core_6
#     0.4973) on ARC-Easy AND does not lose on core_6 by more than its 1.15 pp scatter.
#   * FAIL, and the H-is-harmful story survives intact, iff every mu is at or below the
#     incumbent -- which would be the FOURTH independent confirmation that fitting this
#     format against the calibration Gram hurts, now with the codes held fixed.
#   * A win that appears only on ARC-Easy and not on core_6 is NOT promoted; it is
#     reported as another amplification, exactly as SG_rand and strict_pot_zero were.
#   * Any winner gets the same two-axis replication that killed strict_pot_zero.
set -u
GPU=$1; shift
run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py --method rtn --nsamples 128 \
     --calib-offset 0 --bpw-budget 4.2501 --tasks ppl,arc_easy,mc --verify-layers all \
     --exact-scale --codebook strict_pot --group 128 \
     --group-map default=64,self_attn.v_proj=128 "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -12
}
for arm in "$@"; do case $arm in
  mu1)    run SS_mu1    --solve-scale 1.0 ;;
  mu10)   run SS_mu10   --solve-scale 10.0 ;;
  mu0.1)  run SS_mu0.1  --solve-scale 0.1 ;;
  mu100)  run SS_mu100  --solve-scale 100.0 ;;
  mu0)    run SS_mu0    --solve-scale 0.0 ;;
  *) echo "unknown $arm"; exit 1 ;; esac; done
