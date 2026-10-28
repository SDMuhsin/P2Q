#!/usr/bin/env bash
# =============================================================================
# Does a ZERO level fix the clamping damage?  A derived, single-mechanism test.
#
# PRE-REGISTRATION -- written before any run below.
#
# WHERE THIS COMES FROM
# ---------------------
# §20 established that the dominant failure mode of a strict-PoT alphabet is
# UPWARD clamping of small weights. At the incumbent operating point (nexp=8,
# g=64) 2.536% of all weights fall below the representable floor and are stored
# as +-2^-7*s, which INFLATES them by a mean factor of 7.46x. Those weights
# contribute only 0.0118% of the total squared error, which is exactly why every
# MSE-family objective in this project is blind to them; drive that population to
# 34% (nexp=4) and the model collapses to chance (acc_norm 0.3451, ppl 1826) at
# BETTER weight error and LOWER bpw.
#
# `strict_pot` has no zero, so an under-range weight cannot be made small -- only
# inflated. `strict_pot_zero` (explicitly legal under CHARTER C1) spends one code
# on 0. At 4 bits that is 15 levels: 7 octaves plus zero, versus 8 octaves and no
# zero. Both are charged ceil(log2 n) = 4.000000 bits, so this is EXACTLY
# bpw-neutral at 4.244792 -- a pure trade of one octave of range for a zero.
#
# THE DERIVED PREDICTION
# ----------------------
# With a zero level, an under-range weight w has absolute error |w| < floor and
# relative error 100%. Without one it has absolute error ~= floor and is inflated
# ~7.5x. So if upward clamping is the mechanism, adding zero should HELP, and the
# gain should be visible in the rung statistics as a collapse of the bottom-rung
# population.
#
# Against that: §20 also says dynamic RANGE is load-bearing, and this trade gives
# up an octave (floor moves 2^-7 -> 2^-6, doubling the number of weights that fall
# under it). The two effects oppose, which is what makes this a real test rather
# than a foregone conclusion.
#
# PRE-REGISTERED CALL, fixed now:
#   supports the clamping mechanism  <=> ZP beats 0.6364 on acc_norm AND 0.6654 on acc
#   refutes it                       <=> ZP is >= 1.83 pp WORSE (the config-scatter sd)
#   inconclusive                     <=> anything between
# Every arm is --method rtn, hence H := I, hence draw-independent and exactly
# reproducible: one run per arm IS the answer, no error bars, no draw averaging.
#
# PRIOR EVIDENCE, and why it does not settle this: strict_pot_zero has been run
# five times (ZC_off0..512) and ONLY under GPTQ, giving 0.6233/0.5328/0.6313/
# 0.5720/0.6221, mean 0.5963 against strict_pot-under-GPTQ's 0.6012. That is a
# 0.49 pp difference inside a 5.3 pp per-draw scatter, measured on top of a method
# that we now know injects -2.21 pp of its own damage. It is uninformative.
# strict_pot_zero has NEVER been run with RTN + exact scale.
# =============================================================================
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-1}
GM="default=64,self_attn.v_proj=128"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run () { local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --method rtn --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy \
    --calib-offset 0 --verify-layers all --group 128 --group-map "$GM" \
    --exact-scale --tag $tag "$@" > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_zp.out
}
run ZP_zero_exact  --codebook strict_pot_zero
run ZP_pot_exact   --codebook strict_pot          # control: reproduces the incumbent
echo ZP_COMPLETE >> $D/run_zp.out
