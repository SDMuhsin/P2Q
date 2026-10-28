#!/usr/bin/env bash
# =============================================================================
# RATE ALLOCATION (RA_) -- pre-registration
# =============================================================================
#
# QUESTION
# --------
# Strict PoT requires w = +-2^e with e an INTEGER, so the *relative* resolution of
# the alphabet is pinned at one octave and cannot be improved by spending more bits.
# Extra exponents buy dynamic RANGE, never precision. The incumbent pays 4.000000
# bits/weight for 8 octaves of range. This script asks whether that range is worth
# 4 bits, and moves any surplus rate into group granularity, where -- unlike the
# alphabet -- extra bits DO reduce distortion.
#
# WHAT THE CODEBASE ACTUALLY CHARGES  (the load-bearing constraint)
# -----------------------------------------------------------------
#   src/pot/codebooks.py :: Codebook.bits()  ->  ceil(log2(n_levels))
#   src/pot/run_pot.py   :: code_bits = cb.bits(); account_bpw(codes=code_bits, ...)
# The pipeline charges a FLAT ceil-of-log2 per code. It does NOT entropy-code.
# Therefore a non-power-of-two alphabet size buys NOTHING:
#   strict_pot(5) = 10 levels -> ceil(log2 10) = 4 bits  (NOT 3.322)
#   strict_pot(6) = 12 levels -> 4 bits   [confirmed in the log: F_nexp6_off0 code_bits 4.0]
#   strict_pot(7) = 14 levels -> 4 bits
# Rate is freed ONLY at a power-of-two level count:
#   strict_pot(4) =  8 levels -> 3 bits   (frees exactly 1.000000 bpw)
#   strict_pot_zero(3) = 7 levels -> 3 bits
# So the "0.678 bpw freed by a 5-octave alphabet" idea is not purchasable here. The
# real trade is the discrete one: 4 octaves at 3 bits, spent on a 4x finer group.
#
# BPW ARITHMETIC (n_weights = 1409286144; fp16 group scale; no selector, no g_idx)
#   bpw = code_bits + 16 / group      (uniform group)
#   incumbent  4 bits, g=64 w/ v_proj at 128        = 4.244792   <-- ceiling
#   RA1        3 bits, g=16 uniform                 = 4.000000   (-0.244792)
#   RA2        3 bits, g=32 uniform                 = 3.500000   (-0.744792)
#   RA3        3 bits, g=16 uniform, zero-bearing   = 4.000000   (-0.244792)
#   RA4        4 bits, g=64 w/ v_proj 128, nexp=5   = 4.244792   (matched; range ablation)
#   RA5        4 bits, g=64 w/ v_proj 128, nexp=6   = 4.244792   (matched; range ablation)
# Priced but NOT run (another agent owns column scaling): a per-input-column fp16
# exponent costs 16 * 63488 * 28 / 1409286144 = 0.005859 bpw; an int8 exponent
# 0.002930 bpw. Both are ~40x cheaper than one group-granularity halving (0.25 bpw),
# i.e. essentially free -- but they are also a different axis and are not tested here.
#
# CPU EVIDENCE THAT MOTIVATES THE ARMS (weighted rel_mse, exact scale, RTN-equivalent,
# 7 layers x 7 linear kinds sampled from Qwen3-1.7B):
#   incumbent  pot8 g64/v128 (4 bits, 4.2448 bpw)  rel_mse 0.028618
#   RA1        pot4 g16      (3 bits, 4.0000 bpw)  rel_mse 0.026303   <-- BETTER, cheaper
#   Empirical code entropy of the incumbent stream: H = 3.58 bits, H(code|group) = 3.34
#   bits, i.e. only ~0.42 bits/weight of marginal redundancy -- far LESS surplus than a
#   "bottom rungs are nearly unused" story predicts. The bottom rungs ARE used. The win
#   below does not come from unused symbols; it comes from the fact that a 4x finer
#   group needs far less range, so the octaves can be deleted outright.
#
# PRE-REGISTERED SUCCESS CONDITION
# --------------------------------
# The comparison is at matched or LOWER bpw than the incumbent 4.244792. An arm that
# beats 0.6364 acc_norm at HIGHER bpw does not count. All arms use --method rtn, so
# H := I: deterministic, calibration-independent, one run is the answer, zero variance.
# Rank on arc.acc, report arc.acc_norm (see llmdocs; acc_norm resolves only ~5.7 pp).
#
#   Incumbent (REP_deliverable): acc 0.6654, acc_norm 0.6364, bpw 4.244792.
#
#   RA1_pot4_g16     bpw must log as 4.000000 ; counts as a WIN iff acc >= 0.6654
#   RA2_pot4_g32     bpw must log as 3.500000 ; counts as a WIN iff acc >= 0.6654
#   RA3_potz3_g16    bpw must log as 4.000000 ; counts as a WIN iff acc >= 0.6654
#   RA4_pot5_g64     bpw must log as 4.244792 ; DIAGNOSTIC. Tests whether the bottom 3
#                    octaves earn their place. They are FREE here (ceil charges 4 bits
#                    either way), so acc >= 0.6654 means the octaves are dead weight and
#                    range can be cut with no loss; acc < 0.6654 means they are load-bearing.
#   RA5_pot6_g64     bpw must log as 4.244792 ; same diagnostic, 6 octaves.
#
# Every arm must additionally report verify.ok == true with --verify-layers all and
# bpw_ok == true. An arm failing either is reported as a failure, not quietly dropped.
# Arms that get worse are reported (charter C5).
#
# PREDICTION (registered before running): RA4/RA5 land within noise of the incumbent
# (the range is not load-bearing once the scale is fitted exactly); RA1 beats it at
# 4.000000 bpw; RA2 loses (4 octaves over 32 weights is too narrow); RA3 is near RA1.
# =============================================================================
set -u
cd /workspace/CoBALT

COMMON="--method rtn --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy \
--calib-offset 0 --verify-layers all --exact-scale"

run () {  # run <tag> <extra args...>
  tag="$1"; shift
  echo "=== $tag : $* ==="
  python3 -u src/pot/run_pot.py $COMMON --tag "$tag" "$@" 2>&1 \
    | tee "results/pot/logs/${tag}.log" | tail -25
}

mkdir -p results/pot/logs

# --- the rate-reallocation arms (strictly below the incumbent bpw) ------------
run RA1_pot4_g16  --codebook strict_pot      --codebook-nexp 4 --group 16
run RA2_pot4_g32  --codebook strict_pot      --codebook-nexp 4 --group 32
run RA3_potz3_g16 --codebook strict_pot_zero --codebook-nexp 3 --group 16

# --- matched-bpw range diagnostics (ceil() makes these free) -----------------
run RA4_pot5_g64  --codebook strict_pot --codebook-nexp 5 --group 128 \
                  --group-map default=64,self_attn.v_proj=128
run RA5_pot6_g64  --codebook strict_pot --codebook-nexp 6 --group 128 \
                  --group-map default=64,self_attn.v_proj=128

# =============================================================================
# RESULTS (appended after execution -- the pre-registration above is unedited)
# =============================================================================
# THE PREMISE WAS FALSIFIED. Rate reallocation by shrinking the alphabet destroys
# the model, and it does so INVISIBLY to every MSE-based objective.
#
#   arm            bpw        verify  ppl        arc.acc   arc.acc_norm   vs incumbent
#   incumbent      4.244792   true      21.6*    0.6654    0.6364          --
#   RA1_pot4_g16   4.000000   true    1826.21    0.3548    0.3451         -30.1 pp acc
#   RA2_pot4_g32   3.500000   true    3417.57    0.2828    0.2997         -38.3 pp acc
#   RA3_potz3_g16  --  KILLED before completion (same 3-bit family; RA1/RA2 decided it)
#   RA4_pot5_g64   --  NOT RUN (coordinator: no more GPU on nexp <= 5)
#   RA5_pot6_g64   4.244792   see results/pot/pot_runs.jsonl
#
# RA1 has LOWER weight error than the incumbent (rel_mse 0.026314 vs 0.028500 at the
# same sampling) and is 0.244792 bpw CHEAPER, and it is still 30 points worse on ARC-E,
# at chance. This is the largest proxy failure recorded on this project.
#
# MECHANISM (measured, scripts/../clamp analysis, 4 layers x 7 kinds, exact scale):
#   g    nexp  clamped%  their SSE share  mean |w_hat|/|w| among clamped
#   16   4     29.580%   22.4241%          8.94x
#   16   6      7.946%    0.5082%          7.91x
#   16   8      1.990%    0.0080%          6.98x
#   64   4     34.286%   23.0905%          9.71x
#   64   8      2.536%    0.0118%          7.46x
#  128   8      2.794%    0.0147%          7.25x
# A strict-PoT ladder has no zero and a hard bottom rung at 2^-(nexp-1). Every weight
# below that gets snapped UP. Cutting nexp from 8 to 4 takes the clamped population
# from 2.5% to 34% of all weights, and those weights are inflated ~9x in magnitude on
# average. Squared error cannot see this: a weight near zero inflated to 0.125*s
# contributes a tiny absolute error while being infinitely wrong in relative terms
# (max observed |w_hat-w|/|w| = 2.4e5). MSE nets the clamping damage off against the
# genuine precision gain from a 4x finer group and reports an IMPROVEMENT.
#
# CONCLUSION, inverting the task premise: on a logarithmic alphabet the dynamic RANGE
# is the load-bearing resource, not the precision, and rate must not be harvested from
# it. Any rate allocation driven by weight MSE (or by any Hessian-weighted quadratic
# proxy) will spend range for precision and destroy the model.
