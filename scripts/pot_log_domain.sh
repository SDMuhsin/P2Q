#!/usr/bin/env bash
# =============================================================================
# PRE-REGISTRATION -- the format-native (log-domain) distortion measure
# Written and committed BEFORE any run below was launched.
# =============================================================================
#
# THESIS. Every objective this project has optimized is a squared error on
# weights, and squared error is structurally blind to the one failure mode a
# LOGARITHMIC alphabet has: clamping at the BOTTOM of the range does not make a
# small weight slightly wrong, it makes a small weight LARGE. Relative error
# unbounded, squared-error contribution negligible. (Evidence: the 4-octave
# alphabet at g=16, bpw 4.000000, rel_mse 0.026303 -- both BETTER than the
# incumbent -- collapsed ARC-E by 29 pp to the chance floor.)
#
# The format induces its own distortion. With u = log2|w|, storing +-2^e is
# exactly rounding u onto the alphabet's log-magnitude lattice, so
#       d(w, w_hat) = (log2|w_hat| - log2|w|)^2
# is the induced measure. It charges a small weight clamped UP by 3 octaves
# exactly as much as a large weight clamped DOWN by 3. It is minimized EXACTLY
# over the scale by a breakpoint argument in t = log2(s) (src/pot/log_scale.py):
# no grid, no gradient, no hyperparameter.
#
# TWO INDEPENDENT SWITCHES (this is the 2x2):
#   --log-scale  the OBJECTIVE the per-group scale is fitted against.
#                THIS is the lever on clamping: making bottom-rung clamp-up
#                expensive pushes the scale DOWN, so fewer weights fall below
#                the bottom rung (and more clip at the top, where the error is
#                bounded).
#   --log-round  the ASSIGNMENT rule. The decision boundary between 2^(e-1) and
#                2^e moves from the arithmetic midpoint 1.5*2^(e-1) to the
#                geometric midpoint sqrt(2)*2^(e-1) ~ 1.4142*2^(e-1); log
#                rounding rounds UP over that 8.6%-of-an-octave band and agrees
#                everywhere else. NOTE, derived and asserted in
#                tests/pot/test_log_scale.py: the OUTERMOST boundary (|x| < the
#                bottom rung) is unmoved, so --log-round CANNOT change which
#                weights are clamped up. Only --log-scale can.
#
# =============================================================================
# THE 2x2 AND ITS PREDICTED SIGNS  (predictions, not results)
# =============================================================================
#   cell   scale search      rounding    tag
#   A      linear (exact)    linear      LD_A_lin_lin   == the incumbent
#   B      linear (exact)    log         LD_B_lin_log
#   C      log (exact)       linear      LD_C_log_lin
#   D      log (exact)       log         LD_D_log_log   (objective+assignment
#                                                        mutually consistent)
#   ref    linear (exact)    linear      LD_R_int4sym   int4_sym REFERENCE ARM,
#                                                        a labelled control and
#                                                        NEVER a deliverable.
#                            (int4_sym contains 0, so log2|w_hat| is -inf for
#                             the zero code and the log measure is undefined
#                             there; run_pot.py refuses --log-scale/--log-round
#                             on it by design. Hence only one int4 cell.)
#
# PREDICTED rel_mse (weight MSE):  A < B, A < C, A < D.
#   A is by construction the exact minimizer of exactly this quantity, so every
#   other cell MUST be >= it. If any cell beats A on rel_mse, the exact searches
#   are wrong and the whole run is void.
#
# PREDICTED clamp diagnostics:
#   clamp_lo_frac: C, D << A, B    (the log objective buys this; it is the
#                                   entire mechanism)
#   clamp_lo_frac: B == A exactly  (--log-round cannot move it, see above)
#   clamp_hi_frac: C, D > A, B     (the scale moves down; top clipping is the
#                                   bounded-error end, so this is the trade)
#   log_mse / mean_rel_err: C, D < A, B.
#
# PREDICTED ARC-E (acc; acc_norm reported):
#   D >= C > A,  B ~ A.
#   The thesis says the clamping lever (--log-scale) is what matters and the
#   boundary shift (--log-round) is a small second-order correction.
#
# THE HEADLINE CONFIRMATION WE ARE LOOKING FOR: a cell with WORSE rel_mse and
# BETTER ARC-E than A. That is the strongest possible evidence that weight MSE
# is the wrong objective, and it is a result in its own right even if the
# absolute accuracy does not beat the incumbent.
#
# =============================================================================
# WHAT WOULD FALSIFY THIS, AND THE DECISION RULE
# =============================================================================
# --method rtn sets H := I, so every arm here is calibration-INDEPENDENT and
# exactly reproducible: one run per cell is the whole answer, there are no error
# bars within a cell. The relevant yardstick is the measured config-to-config
# scatter among reasonable configs, sd 1.83 pp on ARC-E.
#
#   * CALL AN EFFECT only for |delta acc| >= 1.83 pp (1 sd) vs cell A.
#     Anything smaller is reported as a NULL, explicitly, not as a trend.
#   * FALSIFIED if C and D are both within +-1.83 pp of A (the measure changes
#     the artifact measurably -- rel_mse and clamp_lo_frac WILL move -- but buys
#     no accuracy: the failure mode is real, this measure is not the fix).
#   * FALSIFIED HARDER if C or D is <= A - 1.83 pp: optimizing the induced
#     relative distortion actively hurts, i.e. squared error was closer to the
#     right objective than the format's own metric.
#   * VOID if any cell fails --verify-layers all, exceeds bpw 4.2501, or beats
#     cell A on rel_mse.
#
# bpw is IDENTICAL across all five cells by construction: none of these switches
# stores a single extra bit -- they change only which code and which scale value
# is written, never how many. Expect 4.244792 in every strict_pot cell.
#
# NO TUNING. Every quantity is exactly derived. There is no hyperparameter to
# sweep here and none will be swept.
#
# =============================================================================
# ADDENDUM -- ARM E: the SALIENCE log-domain objective (--log-scale-hess)
# Pre-registered AFTER cell A had read out (acc 0.6654 / acc_norm 0.6364, an exact
# replicate of the incumbent) and BEFORE any arm-E run was launched. Cells B, C, D
# and R were already in flight and are untouched.
# =============================================================================
#
# WHY THIS ARM COMPLETES THE FAMILY (llmdocs/SESSION_2026-09-10_GPTQ_NATIVE.md §21.5).
# The honest diagonal output error of y = Wx is  sum_j h_j sum_i w_ij^2 eps_ij^2,
# with h_j = E[x_j^2] and eps_ij = w_hat_ij/w_ij - 1 the RELATIVE error. Every
# candidate objective is that expression with a different factor on eps^2:
#
#   factor on eps_ij^2   objective            arm      status
#   -----------------    ------------------   -----    --------------------------
#   w_ij^2               weight MSE           A / B    the incumbent
#   h_j * w_ij^2         honest output error  --       MEASURED, CATASTROPHIC:
#                                                      0.5282 (diag) / 0.5812 (ldl)
#                                                      vs 0.6364
#   1                    log-domain           C / D    this task
#   h_j                  SALIENCE log-domain  E        THIS ARM
#
# So the log-domain objective is NOT "MSE plus a reweighting" -- it is MSE with the
# w^2 factor REMOVED, and w^2 is exactly the factor that makes squared error blind
# to a small weight clamped upward onto the bottom rung. h alone is the only
# untested member of the family.
#
#   E   log scale + salience h_j   log rounding   LD_E_saliencelog_off{0,64,128}
#       (= cell D plus the h_j weights, one factor changed, nothing else)
#
# PREDICTED SIGN -- E < D, and I expect E to be a LOSS, not a win.
#   Mechanism: the group scale is SHARED by 64 columns, and h for transformer
#   activations is enormously heterogeneous (outlier channels are orders of
#   magnitude above the median). Weighting the log fit by h therefore collapses the
#   effective sample size of the fit from 64 columns to a handful, and lets a few
#   outlier channels set the shared scale for the whole group. That is the same
#   mechanism that plausibly killed the h*w^2 arms at 0.5282 / 0.5812.
#   Point predictions: E <= D - 1.83 pp on ARC-E acc; clamp_lo_frac HIGHER in E
#   than in D (the outlier channels pull the scale back up, undoing exactly the
#   clamping protection the log objective bought).
#
# WHAT WOULD FALSIFY *THIS* PREDICTION: E >= D + 1.83 pp. That would say the win is
# the ADDITION of h, not the removal of w^2, and would be the first time in this
# project that anything H-dependent helped a log grid.
#
# DRAWS. --log-scale-hess reads the calibration Gram, so unlike every other arm here
# this one is calibration-DEPENDENT and scatters ~3.8 pp per draw. It gets THREE
# draws (--calib-offset 0, 64, 128) and is judged ON THE MEAN, never on one draw.
# The h-free arms A-D and R stay deterministic under --method rtn (H := I) and get
# one run each, which is the whole answer for them.
#
# The comparison D vs mean(E) is then one deterministic number against a 3-draw
# mean; I will call it only at >= 1.83 pp, and will report the per-draw spread.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
COMMON=(--method rtn --nsamples 128 --bpw-budget 4.2501
        --tasks ppl,arc_easy --calib-offset 0 --verify-layers all
        --group 128 --group-map default=64,self_attn.v_proj=128)
MARK="${MARK:-/tmp/pot_log_domain}"
mkdir -p "$MARK"

run () {   # run <tag> <extra args...>  -- idempotent: skips arms already done
  local tag="$1"; shift
  if [ -f "$MARK/$tag.done" ]; then echo "=== $tag (already done, skipping) ==="; return 0; fi
  echo "=== $tag ==="
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u src/pot/run_pot.py "${COMMON[@]}" "$@" \
      --tag "$tag" 2>&1 | tee "$MARK/$tag.log"
  echo "DONE $tag" >> "$MARK/$tag.done"
}

run LD_A_lin_lin  --codebook strict_pot --exact-scale
run LD_B_lin_log  --codebook strict_pot --exact-scale --log-round
run LD_C_log_lin  --codebook strict_pot --log-scale
run LD_D_log_log  --codebook strict_pot --log-scale  --log-round
# REFERENCE BASELINE ONLY -- int4_sym is not a deliverable under this charter.
run LD_R_int4sym  --codebook int4_sym   --exact-scale

# --- ARM E: salience log-domain. Calibration-dependent -> three draws, judged on
# --- the mean. Cell D plus h_j weights; every other setting identical.
for OFF in 0 64 128; do
  run "LD_E_saliencelog_off${OFF}" --codebook strict_pot --log-scale --log-round \
      --log-scale-hess --calib-offset "$OFF"
done

python3 -u src/pot/summarize.py --tags LD_ 2>/dev/null || true

# =============================================================================
# ADDENDUM 2 -- IS diag(H) ADEQUATE ON A LOG GRID?  (src/pot/diag_adequacy.py)
# Pre-registered BEFORE the diagnostic was run. No full-model job involved.
# =============================================================================
#
# MOTIVATION -- the one sub-result of the 2x2 that was not a negative. Arms D and
# E have nearly identical POOLED per-weight diagnostics (bottom rung 2.23% vs
# 2.27%, mean |dlog2| at the bottom 0.812 vs 0.810, top rung 15.85% vs 16.01%)
# and differ by 30.06 pp of ARC-E acc. The entire difference is WHICH groups got
# the bad scale. That is a statement about the error's CORRELATION STRUCTURE.
#
# DERIVATION. E[||dy||^2] = tr(E H E^T). Every activation-aware method here uses
# only diag(H), which is exact iff sum_{j!=k} H_jk <E[:,j],E[:,k]> = 0, i.e. iff
# the per-column errors are ~independent. On a uniform grid they nearly are
# (residuals ~ i.i.d. uniform, independent of the weight). On a log grid they are
# not: (1) all g columns of a group SHARE one scale and the relative error is a
# DETERMINISTIC function of frac(log2|w/s|); (2) every clamped weight is displaced
# to the same +-2^-(n-1) s, sign-preserving -- a systematic, nearly rank-structured
# perturbation sitting exactly where the error is largest.
#
# If true, ONE mechanism explains five separate failures, all of which use a
# diagonal or a diagonally-driven salience: GPTQ feedback (-2.21 pp), scale-weight
# 'ldl' (0.5812) and 'diag' (0.5282), equalization with h (-6.2 pp), and arm E
# (-10.84 pp).
#
# MEASUREMENT. rho = tr(E H E^T) / tr(E diag(H) E^T) on the TRUE Gram with
# off-diagonals kept, for strict_pot and int4_sym at matched group, on real
# Qwen3-1.7B layers. rho == 1 <=> the dropped cross terms are zero.
#
# THE CONTROL. rho can move because the errors are CORRELATED across columns, or
# merely because their magnitudes sit on channels with big H_jk. So every rho is
# paired with rho_signflip: the same error tensor witheach entry's SIGN randomized,
# which preserves the diagonal term EXACTLY and destroys all cross-column
# correlation. rho_signflip is what rho would be under independence. The claim is
# carried by  excess = |rho - 1| - |rho_signflip - 1|.
#
# PREDICTION: excess(strict_pot) > excess(int4_sym), and larger still restricted to
# the bottom-rung population than on the complement.
#
# FALSIFIED IF strict_pot's excess is <= int4_sym's, or if both are ~0. Then the
# diagonal approximation is no worse for PoT than for INT4, the derivation is dead,
# the activation-aware question CLOSES, and step 2 below is not run at all.
#
# STEP 2, ONLY IF STEP 1 CONFIRMS. The group scale is shared by exactly g columns,
# so the correct object is the g x g diagonal BLOCK H_gg, and the group objective
# is e_g^T H_gg e_g. NOTE, derived before running: the breakpoint argument SURVIVES
# intact -- on an interval where the assignment is fixed, e_g = c*s - w is affine
# in s, so e^T H e = s^2 (c^T H c) - 2s (c^T H w) + w^T H w is still an exact
# quadratic and the breakpoints (which depend only on the assignment map) do not
# move. So the exact minimizer holds; only the per-interval coefficients get more
# expensive (a g x g contraction instead of a dot product). No bracketed search
# needed. Judged vs the incumbent (0.6654 / 0.6364 / 4.244792) at >= 3 calib
# offsets, call threshold 1.83 pp, rung occupancy reported per arm.
# =============================================================================
