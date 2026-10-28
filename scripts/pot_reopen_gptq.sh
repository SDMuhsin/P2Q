#!/usr/bin/env bash
# REOPENING THE METHODS THAT ARC-EASY KILLED
#
# Why. The incumbent is RTN + exact scale. RTN is the least sophisticated quantizer
# there is, and the entire PTQ literature beats it -- so "RTN is optimal" is a priori
# implausible and the likely error is in how the alternatives were JUDGED.
#
# Every kill that collapsed this search onto RTN was measured on ARC-Easy:
#   GPTQ           SESSION 13:   GPTQ - RTN = -2.21 pp   (9 draws)
#   refine_codes   SESSION RF:   -6.51 pp, t = -4.84
#   rotation       SESSION 22:   -32.45 pp (PoT), -18.06 pp (int4)
#
# And this session established that ARC-Easy cannot rank artifacts in this family: seven
# strict_pot arms span 10.19 pp on ARC-Easy (9.93 se) and 1.15 pp on six other
# benchmarks (1.95 se). EVERY PANEL ARM SO FAR IS RTN. The evidence that dethroned GPTQ
# has never been read on the instrument that showed the dethroning metric is unreliable.
#
# PRE-REGISTERED, before any arm ran:
#   * If GPTQ - RTN is negative on panel_macro AND core_6 by more than their scatter
#     (core_6 spread among strict_pot arms = 1.15 pp), SESSION 13 stands, RTN is
#     genuinely better, and the "H is harmful on a log grid" story survives its first
#     real test.
#   * If GPTQ - RTN is null or POSITIVE on the panel, then SESSION 13's -2.21 pp was an
#     ARC-Easy artifact, RTN's incumbency is an artifact of the metric, and GPTQ /
#     act-order / code refinement are all REOPENED. That would also retire the
#     anti-alignment mechanism (SESSION 22.2) as the explanation for five failures,
#     since those failures would not be failures.
#   * int4_sym + GPTQ is the reference control: SESSION 13 has it at +1.21 pp over
#     int4-RTN, so the panel must reproduce a POSITIVE there or the comparison is broken.
set -u
GPU=$1; shift
BASE=(python3 -u src/pot/run_pot.py --nsamples 128 --calib-offset 0 --bpw-budget 4.2501
      --tasks ppl,arc_easy,mc --verify-layers all --exact-scale
      --group 128 --group-map default=64,self_attn.v_proj=128)
run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU "${BASE[@]}" "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -12
}
for arm in "$@"; do case $arm in
  gptq)      run GP_gptq      --method gptq --codebook strict_pot ;;
  gptq_ao)   run GP_gptq_ao   --method gptq --codebook strict_pot --act-order ;;
  gptq_int4) run GP_gptq_int4 --method gptq --codebook int4_sym ;;
  refine)    run GP_refine    --method rtn  --codebook strict_pot --refine-codes --refine-hessian calib --refine-sweeps 3 ;;
  gptq_o128) run GP_gptq_o128 --method gptq --codebook strict_pot --calib-offset 128 ;;
  *) echo "unknown $arm"; exit 1 ;; esac; done
