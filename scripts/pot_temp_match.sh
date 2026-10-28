#!/usr/bin/env bash
# TEMPERATURE MATCH -- one scalar on the final RMSNorm gain, zero bits, fitted by
# KL(bf16 || student) on CALIBRATION text. Mechanism H* and predictions are in
# src/pot/temp_match.py and llmdocs/SESSION_2026-09-11_INSTRUMENT.md.
# PRE-REGISTERED (23:20 UTC, before any arm ran):
#   * alpha != 1 by more than 0.01 for the incumbent (its BoolQ margin 2.26 sits below
#     bf16's 3.44, so alpha should come out > 1, i.e. it SHARPENS).
#   * BoolQ margin moves toward 3.44; per-token gold NLL moves toward bf16's.
#   * ARC-Easy acc_norm moves in the SAME direction the sharper artifacts sit, i.e. UP
#     for the incumbent. Six-task core: within its se.
#   FALSIFIED if alpha ~ 1, or BoolQ margin moves away from 3.44, or ARC-E falls.
set -u
GPU=$1; shift
BASE=(python3 -u src/pot/run_pot.py --method rtn --nsamples 128 --calib-offset 0
      --tasks ppl,arc_easy,mc --verify-layers all --exact-scale --bpw-budget 4.2501
      --group 128 --group-map default=64,self_attn.v_proj=128 --temp-match)
run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU "${BASE[@]}" "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -12
}
for arm in "$@"; do case $arm in
  pot)       run TM_pot        --codebook strict_pot ;;
  zero)      run TM_zero       --codebook strict_pot_zero ;;
  zero_bias) run TM_zero_bias  --codebook strict_pot_zero --out-correct bias --out-bias-bits 8 ;;
  rand0)     run TM_rand0      --codebook strict_pot --order rand --perm-seed 0 ;;
  *) echo "unknown $arm"; exit 1 ;; esac; done
