#!/usr/bin/env bash
# THE SCALE RESPONSE SURFACE -- the one continuous parameter the format has, mapped.
#
# The artifact is W_hat = s_{r,g} * (+-2^e). `--exact-scale` sets s to the exact argmin
# of group MSE. The window it selects is [s*2^-7.5, s]: RAISING s means more bottom-rung
# clamping (unbounded relative error, mean 43x magnitude inflation -- measured at every
# group size this session) and less top-rung clipping (bounded, ~0.098 relative);
# LOWERING s is the reverse. MSE picks one point on that trade. Whether it is the
# ACCURACY optimum has never been measured -- in 226 logged runs the scale has only ever
# been moved by REWEIGHTING the objective (diag/ldl/invdiag/powdiag/log/salience), never
# by displacing it directly.
#
# Everything here is bpw-IDENTICAL to the incumbent (4.244792): the scale is still stored
# fp16. These are diagnostics of the response surface, not deliverables.
#
# ---------------------------------------------------------------------------------
# E1  systematic offset  s -> s * 2^delta
# ---------------------------------------------------------------------------------
# PRE-REGISTERED, three mutually exclusive readings:
#   H1 "MSE is the accuracy optimum too": panel_macro peaks at delta=0 and falls on
#      both sides.
#   H2 "the optimum is displaced": panel_macro is monotone across the swept range, so
#      the accuracy optimum lies outside +-0.10 octaves in that direction. The sign then
#      says which failure mode is being under-priced -- delta<0 better => MSE is
#      under-pricing bottom clamping; delta>0 better => it is under-pricing clipping.
#   H3 "flat": no resolvable structure over +-0.10 octaves.
#
#   H3 would CONTRADICT the project's standing "the scale sits on a cliff" claim
#   (SESSION 25.1), which was never measured directly: it was INFERRED from the fact
#   that the diag(H)-weighted optimum sits 0.030 octaves away and costs ~9 pp. That
#   inference has an alternative explanation this sweep separates -- h-weighting does
#   not merely displace the scale, it displaces each group's scale by a DIFFERENT,
#   activation-correlated amount, and the damage may be in the correlation rather than
#   in the displacement. A uniform delta is the control that tells them apart.
#
# ---------------------------------------------------------------------------------
# E2  random rounding  log2(s) -> nearest multiple of Delta
# ---------------------------------------------------------------------------------
# Each group's scale moves by an uncorrelated amount <= Delta/2 octaves, with no
# systematic direction -- the exact complement to E1, and exactly the distortion a
# Delta-resolution STORED scale would suffer. So this sweep simultaneously (a) measures
# how fragile the artifact is to scale perturbation, which is the sharpest available
# probe of the artifact-to-artifact chaos, and (b) prices the scale's bit budget.
#
# ANCHOR: Delta=1.0 snaps every scale to a power of two. The nearest logged arm,
# DET_potscale (a SEARCHED power-of-two scale, which is strictly better than rounding to
# one), cost -9.85 pp. So Delta=1.0 MUST come back badly negative; if it does not, the
# arm is not doing what it claims and nothing else in the sweep should be believed.
# PREDICTED: flat to Delta ~ 0.05, clearly degraded by Delta = 0.25.
#
# ---------------------------------------------------------------------------------
# SR_fp16scale  an accounting fix that doubles as the smallest perturbation probe
# ---------------------------------------------------------------------------------
# `account_bpw` has always charged 16 bits per (row, group) scale. The pipeline stores
# float32. So the incumbent's 4.244792 bpw -- and every one of the 226 logged runs --
# assumes a rounding that was never applied. `--scale-fp16` applies it, making the
# artifact match its own accounting. The perturbation is <= 2^-11 relative, ~0.0007
# octaves, about 30x finer than the finest --scale-round-log2 arm here, and section 3.3
# of this session measured that even Delta = 0.005 octaves costs 0.00% of rel_mse.
# So any movement on the metric is pure artifact chaos at a perturbation size that is
# provably irrelevant to the weights -- the cleanest free noise-floor estimate the
# project can construct, in the tradition of its own "provably-equivalent pair" lesson.
#
# ---------------------------------------------------------------------------------
# D  group-size diagnostic -- ILLEGAL bpw, NEVER a deliverable
# ---------------------------------------------------------------------------------
# g=32 at fp16 costs 4.500000 bpw and is over the C3 ceiling. It is run ONLY to separate
# two questions that Lever A conflates: "does finer grouping help accuracy at all?" from
# "can a cheaper scale encoding pay for it?". This session's CPU precondition already
# FAILED its kill criterion on the stated mechanism (halving g cuts bottom clamping by
# 10.2%, not the 25% required), leaving the lever resting on rel_mse (-12%), a currency
# this project has twice shown can move opposite to accuracy. If this arm does not beat
# the incumbent on the panel, the lever is dead and no codec gets built.
set -u
GPU=$1; shift
BASE=(python3 -u src/pot/run_pot.py --method rtn --nsamples 128 --calib-offset 0
      --tasks ppl,arc_easy,mc --verify-layers all --codebook strict_pot --exact-scale
      --group 128 --group-map default=64,self_attn.v_proj=128)

run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU "${BASE[@]}" "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     2>&1 | grep -vE "it/s\]|it\]$" | tail -22
}

for arm in "$@"; do
case $arm in
  off-0.10) run SR_off-0.10 --bpw-budget 4.2501 --scale-offset-log2 -0.10 ;;
  off-0.05) run SR_off-0.05 --bpw-budget 4.2501 --scale-offset-log2 -0.05 ;;
  off-0.02) run SR_off-0.02 --bpw-budget 4.2501 --scale-offset-log2 -0.02 ;;
  off+0.02) run SR_off+0.02 --bpw-budget 4.2501 --scale-offset-log2  0.02 ;;
  off+0.05) run SR_off+0.05 --bpw-budget 4.2501 --scale-offset-log2  0.05 ;;
  off+0.10) run SR_off+0.10 --bpw-budget 4.2501 --scale-offset-log2  0.10 ;;
  rnd0.02)  run SR_rnd0.02  --bpw-budget 4.2501 --scale-round-log2 0.02 ;;
  rnd0.05)  run SR_rnd0.05  --bpw-budget 4.2501 --scale-round-log2 0.05 ;;
  rnd0.10)  run SR_rnd0.10  --bpw-budget 4.2501 --scale-round-log2 0.10 ;;
  rnd0.25)  run SR_rnd0.25  --bpw-budget 4.2501 --scale-round-log2 0.25 ;;
  rnd1.00)  run SR_rnd1.00  --bpw-budget 4.2501 --scale-round-log2 1.00 ;;
  g32)      run SR_g32_ILLEGAL --bpw-budget 4.5001 --group 32 --group-map default=32 ;;
  fp16)     run SR_fp16scale --bpw-budget 4.2501 --scale-fp16 ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac
done
