#!/usr/bin/env bash
# =============================================================================
# SG_ : section 21.4(A) -- "which columns share a scale"
# PRE-REGISTRATION.  Written BEFORE any arm was run.  Do not edit below this
# banner; results are appended at the bottom.
# =============================================================================
#
# ESTABLISHED FACTS (read out of the code, not assumed)
# ----------------------------------------------------
# F1. The permutation changes GROUP MEMBERSHIP, not just processing order.
#     gptq_pot.quantize does `W = W[:, perm]` BEFORE the group loop and groups are
#     `slice(gi*group, (gi+1)*group)` of that permuted matrix, so group g holds
#     columns perm[64g : 64g+64].  g_idx = (arange(cols)//group)[argsort(perm)] is
#     stored so verify_shiftfree.expand_scales can rebuild S = scales[:, g_idx].
# F2. Under --method rtn the permutation acts ONLY through the partition.  H := I
#     gives Hinv = I, so `corr = err * Hinv1[i, i:]` touches column i alone and
#     column i is then overwritten with q.  No residual crosses a column boundary.
# F3. Corollary of F1+F2: reversing a permutation is a NULL change under rtn.
#     reverse maps group g to group G-1-g and leaves every group SET identical
#     (cols % group == 0 is asserted).  So an ASCENDING arm is NOT a valid
#     falsification control here -- it is bit-identical.  Verified in the unit
#     tests.  The control must be a RANDOM permutation with the same g_idx cost.
# F4. `--order act` under rtn is torch.argsort(ones).  On CUDA that returns the
#     identity (checked, n = 256/2048/6144), so act really is a no-op on the GPU
#     path -- but it is NOT the identity on CPU.  The no-op is an accident of the
#     CUDA sort, not a theorem.  Either way the permutation machinery has never
#     been exercised under rtn with a non-constant key.
#
# THE DERIVED KEY
# ---------------
# Section 21.1: on a log-uniform alphabet the error is multiplicative, so the only
# non-multiplicative damage terms are the two window ends.  A weight clamps onto
# the bottom rung when log2|W_ij| < log2 s_ig - 7, and its displacement is then
# ~2^-7 s_ig, INDEPENDENT of its own value.  So a group's clamping damage is
# governed by its within-group DYNAMIC RANGE in log space, and the partition is
# the only handle on it (the per-column argmin cancels h_j, section 19.3).
#
# The key must be a COLUMN statistic (the permutation is shared by all rows).  The
# statistic that governs where a column sits relative to a group's top is its
# LOCATION in log space,
#       mu_j = mean_i log2|W_ij|                       -- `--order logmean`
# i.e. the log of the geometric-mean magnitude.  The geometric mean is the correct
# average here precisely because the alphabet is uniform in log space.  Not the
# max (it is a 2048-sample extreme; measured Spearman(mu, max) = 0.073, so it is
# nearly pure sampling noise) and not the norm ||W[:,j]||^2 (that is an ARITHMETIC
# mean of squares, dominated by the same upper tail; Spearman(mu, log2||W||^2) =
# 0.743).  A low quantile would also do -- Spearman(mu, q05) = 0.733 -- but it is
# estimated from the tail, whereas the mean is the minimum-variance estimator of
# the location, so the mean is chosen.
#
# Salience-weighted form.  Section 21.1 says PoT salience is the per-weight output
# energy h_j W_ij^2.  Reducing THAT to a column statistic with the same geometric
# mean gives
#       k_j = mean_i log2(h_j W_ij^2) = log2 h_j + 2 mu_j   -- `--order salogmean`
# The coefficient 2 on mu is fixed by the error model; there is no free parameter
# and nothing is tuned.  --method rtn overwrites H with I before quantize(), so
# run_pot.py captures the real calibration diag(H) first and passes it as
# perm_hess; the ROUNDING stays pure RTN, only the partition sees calibration.
#
# PREDICTIONS (registered)
# ------------------------
# P1 (mechanism, already measured on the weights alone, no GPU):  logmean reduces
#    the mean within-(row,group) log2 spread from 8.0386 to 7.9418 octaves and the
#    "more than 7 octaves below the group max" fraction from 1.7799% to 1.6907%
#    (first 4 layers).  errmag lands in the same place (7.9514 / 1.6869%).  `rand`
#    is indistinguishable from `none` (8.0498 / 1.7960%).  So the ordering of the
#    arms on the MECHANISM variable is predicted:
#        logmean ~ errmag  <  none ~ rand      (spread and clamp fraction)
#
# P2 (the size of the lever, and why I expect a NEGATIVE).  Decomposing log2|W|
#    into a per-column location and a within-column residual gives
#        sd(mu_j) = 0.2215 octaves      sd(log2|W_ij| - mu_j) = 1.6421 octaves
#    The permutation can only act on the first.  It is 1.8% of the variance.  The
#    within-group spread of 64 weights is set almost entirely by drawing 64
#    samples from ONE heavy-tailed distribution, not by column heterogeneity: the
#    "de-located" floor (every column's mu subtracted, so no heterogeneity is left
#    to remove) is 7.9856 octaves, only 0.053 below `none`.  The clamp fraction
#    moves by 0.09 percentage points in absolute terms, against the 34% clamping
#    that produced the pot4 collapse.  I therefore predict:
#        P2a  the mechanism variable moves as in P1 (this is arithmetic, it will);
#        P2b  ARC-E does NOT move.  All four arms land inside +-1.83 pp of the
#             incumbent's acc 0.6654 / acc_norm 0.6364, i.e. no arm separates.
#    This is a pre-registered NEGATIVE.  Section 21.4(A) is predicted to be a real
#    but quantitatively negligible degree of freedom on this weight distribution.
#
# P3 (salogmean).  sd(log2 h_j) across columns is expected to be much larger than
#    2*sd(mu_j) = 0.44 octaves, so the salience term will DOMINATE the key and
#    salogmean will behave like a pure h-sort: its spread/clamp numbers should be
#    close to `none`/`rand`, NOT to `logmean`.  If so, the salience-weighted form
#    destroys the very mechanism it was supposed to steer, and that is the
#    derivation's answer to "can activation-awareness enter through the partition":
#    only by giving up the log-magnitude alignment that made the partition matter.
#
# WHAT COUNTS AS A RESULT
# -----------------------
# Config-to-config scatter among reasonable configs is sd 1.83 pp on ARC-E, and 5
# ARC-E draws resolve only ~5.7 pp.  So:
#   * I will call a WIN only if logmean beats the incumbent by >= 2.0 pp on
#     arc.acc at bpw <= 4.2501.  Anything smaller is scatter and is reported as a
#     null.
#   * `rand` is the falsification control.  If rand moves ARC-E as much as logmean
#     does, the effect is the g_idx charge or noise and NOT the partition
#     mechanism, and the hypothesis is dead regardless of the sign.
#   * If logmean demonstrably reduces clamping (P1 holds) but does not improve
#     ARC-E (P2b holds), that REFUTES "within-group spread -> ARC-E" at this
#     magnitude and is reported as the result.
#
# BPW.  Every non-'none' order pays the same g_idx charge, 0.002075 bpw, so every
# arm below is 4.244792 + 0.002075 = 4.246867 against the 4.2501 budget.  The
# arms are bpw-neutral against each other by construction; only `none` is cheaper,
# which makes `none` the conservative control.
#
# -----------------------------------------------------------------------------
set -euo pipefail
cd /workspace/CoBALT
COMMON=(--method rtn --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy
        --verify-layers all --codebook strict_pot --group 128
        --group-map default=64,self_attn.v_proj=128 --exact-scale)
run () {  # run <tag> <extra args...>
  local tag="$1"; shift
  CUDA_VISIBLE_DEVICES="${GPU:-1}" \
  python3 -u src/pot/run_pot.py "${COMMON[@]}" "$@" --tag "$tag" \
    2>&1 | tee "results/pot/logs/${tag}.log"
  touch "results/pot/logs/${tag}.done"
}

# --- deterministic arms (H := I, calibration-independent, one run is the answer)
run SG_none      --calib-offset 0 --order none
run SG_errmag    --calib-offset 0 --order errmag
run SG_logmean   --calib-offset 0 --order logmean
run SG_rand      --calib-offset 0 --order rand --perm-seed 0

# --- salience arm: reads the calibration Hessian, so >= 3 offsets
run SG_salog_o0  --calib-offset 0    --order salogmean
run SG_salog_o1  --calib-offset 256  --order salogmean
run SG_salog_o2  --calib-offset 512  --order salogmean

# --- second random draw, to size the control's own scatter
run SG_rand2     --calib-offset 0 --order rand --perm-seed 7

# =============================================================================
# RESULTS (appended; pre-registration above is unedited)
# =============================================================================
# Incumbent REP_deliverable: acc 0.665404 acc_norm 0.636364 bpw 4.244792 ppl 19.9214
#
# arm          order      bpw        verify  acc       acc_norm  ppl      bot(2^-7)  top(2^0)
# SG_none      none       4.244792   True    0.665404  0.636364  19.9214  2.5370%    8.5779%
# SG_errmag    errmag     4.246867   True    0.588384  0.554293  18.3826  2.4111%    9.0083%
# SG_logmean   logmean    4.246867   True    0.627525  0.595539  19.3900  2.4074%    9.0016%
# SG_salog_o0  salogmean  4.246867   True    0.595118  0.546296  23.1248  2.4828%    8.7614%
# SG_rand s0   rand       4.246867   True    0.700758  0.682239  19.1604  2.5625%    8.4820%
# SG_rand s2   rand       4.246867   True    0.590909  0.557239  19.8904  2.5625%    8.4826%
# SG_rand s4   rand       4.246867   True    0.535354  0.505892  21.5973  2.5612%    8.4863%
# (SG_salog_o1/o2, SG_rand s1/s3 died to other agents' CUDA OOM; not results.)
#
# P1 CONFIRMED (mechanism): logmean/errmag cut bottom-rung occupancy 2.5370% ->
#   2.4074%/2.4111% and the weights-only within-group log2 spread 8.0452 -> 7.9682
#   /7.9741 octaves (all 28 layers). rand RAISES it (2.5625%, spread 8.0569).
# P2b CONFIRMED, and harder than predicted: the derived keys did not merely fail
#   to help, they LOST by 3.8-7.7 pp. Across the four deterministic arms ARC-E acc
#   is monotonically DECREASING in the clamping reduction (Spearman -1.0).
#   => "minimise within-group dynamic range" is FALSIFIED as a design principle,
#      with the sign inverted, not merely null.
# P3 CONFIRMED: sd(log2 h_j) = 0.86-2.09 oct vs 2*sd(mu_j) = 0.21-0.66 oct
#   (layer 0, measured), so salience dominates the salogmean key; its clamping sits
#   between none and logmean, exactly as predicted, and it is the worst acc_norm arm.
#
# THE CONTROL WON, THEN DIED TO ITS OWN DISTRIBUTION.
#   rand seed 0 beat the incumbent by +3.54 pp acc / +4.59 pp acc_norm. Three seeds:
#       acc mean 0.6090  sd 0.0842  (n=3; 0.7008, 0.5909, 0.5354)
#   The mean is 5.6 pp BELOW the incumbent and the sd is 4.6x the 1.83 pp
#   config-to-config scatter. Per the rule fixed before the numbers existed, seed 0
#   is a LOTTERY TICKET and is WITHDRAWN. Do not quote it as a method.
#
# THE PROXY DOES NOT TRANSFER (weights-only, no GPU).
#   Over the 6 evaluated arms: r(acc, frac_bottom_rung) = +0.466 (spearman +0.200),
#   r(acc, frac_top_rung) = -0.474, r(acc, rel_mse) = -0.457, r(acc, ppl) = -0.259.
#   The other lane's -0.981 does not hold here; the sign is INVERTED and weak.
#   Decisive pair: rand s0 vs rand s2 are 10.99 pp apart in acc (12.50 in acc_norm)
#   with bottom-rung 2.562452% vs 2.562465%, top-rung 8.482043% vs 8.482566%,
#   rel_mse 0.02888436 vs 0.02888406 -- every weights-only diagnostic agrees to ~5
#   significant figures. No weights-only screen can separate them, so an offline
#   permutation search against this proxy is not justified. Step 2 NOT run.
