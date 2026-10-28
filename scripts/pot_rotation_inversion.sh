#!/usr/bin/env bash
# =============================================================================
# pot_rotation_inversion.sh
#
# PRE-REGISTRATION.  Written and committed to disk BEFORE any run below was
# executed.  Nothing in this header was edited after the first result landed.
#
# -----------------------------------------------------------------------------
# HYPOTHESIS
# -----------------------------------------------------------------------------
# Incoherence processing (QuIP / QuaRot / SpinQuant / QuIP#) multiplies the
# weights by a random orthogonal / Hadamard transform so that no coordinate is
# an outlier relative to the quantization grid.  It is the single most reliable
# trick in modern UNIFORM-grid PTQ, and it is free at inference because the
# transform is absorbed into neighbouring linear layers.
#
# CLAIM: this insight INVERTS on a logarithmic alphabet.
#
#   * A uniform grid has ABSOLUTE resolution.  Its worst case is set by the
#     single largest coordinate in a group, which fixes the step size for every
#     other coordinate.  Rotation Gaussianizes the group, collapsing
#     max/rms, which directly buys back step size.  Rotation should HELP.
#
#   * A strict-PoT grid {+-2^e}, 8 octaves wide, has RELATIVE resolution: the
#     step at magnitude m is proportional to m.  Its absolute squared error is
#     dominated by the TOP rung (measured on this project: 27.3% of total
#     squared error sits on the top rung, 0.011% on the bottom).  The alphabet
#     is only "paid for" if weight magnitudes are spread across the octaves --
#     roughly LOG-UNIFORM in magnitude, i.e. heavy-tailed.  Rotation drives the
#     group toward a Gaussian with small max/rms, which piles mass into a
#     narrow band near the rms, wastes the cheap dense octaves near zero, and
#     (because a Gaussian's log-magnitude has a long LEFT tail) pushes more
#     weights onto the bottom rung where they clamp.  Rotation should HURT.
#
# -----------------------------------------------------------------------------
# PRE-REGISTERED PREDICTIONS -- the 2x2, ranked on arc.acc (per project rule:
# rank on acc, report acc_norm; mean_layer_rel_mse and ppl are NOT proxies on a
# log grid)
# -----------------------------------------------------------------------------
#   cell                                predicted sign of (rotate - none) on arc.acc
#   int4_sym  + --rotate hadamard       POSITIVE  (or at worst neutral)
#   strict_pot+ --rotate hadamard       NEGATIVE, and larger in magnitude
#
# i.e. the predicted INTERACTION is what matters: sign(d_int4) > sign(d_pot).
# The project's measured config-to-config scatter among equally-reasonable
# deterministic configs is sd 1.83 pp, so a single-cell delta below ~2 pp is
# not evidence of anything.  The interaction must therefore be several pp, and
# it must replicate on the independent axis (random_hadamard x 3 seeds) to
# count.
#
# FALSIFICATION.  I am wrong if ANY of:
#   (F1) strict_pot + hadamard >= strict_pot + none on arc.acc (rotation does
#        not hurt the log grid), OR
#   (F2) int4_sym + hadamard <= int4_sym + none on arc.acc (rotation does not
#        help the uniform grid here either -- then the contrast is about this
#        model/pipeline, not about the alphabet), OR
#   (F3) the random_hadamard replication (3 seeds) straddles the none-baseline
#        for strict_pot, i.e. the PoT damage is not reproducible across draws.
#   (F4) MECHANISM: if the log-magnitude analysis shows rotation does NOT
#        narrow the within-group spread of log2(|w|/max|w|) and does NOT
#        increase bottom-rung clamping, then even a confirmed outcome would
#        not be explained by the stated mechanism and I should not claim it.
#
# A confirmed NEGATIVE (rotation helps PoT too) is a perfectly good outcome and
# will be reported as such.  No hyperparameters are tuned anywhere below.
#
# -----------------------------------------------------------------------------
# DESIGN NOTES
# -----------------------------------------------------------------------------
#  * --method rtn sets H := I, so every run here is completely independent of
#    the calibration draw and is EXACTLY reproducible.  One run per cell IS the
#    whole answer for the deterministic arm; there are no error bars to compute.
#  * --rotate hadamard is deterministic (Sylvester / Paley construction) and is
#    therefore the PRIMARY contrast.  --rotate random_hadamard draws a +-1
#    diagonal from --seed; --seed does NOT touch calibration data, so seeds are
#    a legitimate replication axis for rotation and ONLY for rotation.
#  * int4_sym is run ONLY as a clearly-labelled reference baseline (CHARTER
#    C1/C2).  It is never a deliverable.  Its verify.ok is null by construction
#    and that is expected.
#  * Budget C3: --bpw-budget 4.2501.  PoT cells must show verify.ok == true.
# =============================================================================
set -u
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=1
mkdir -p llmdocs/rot_logs

COMMON="--method rtn --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy \
 --calib-offset 0 --verify-layers all --group 128 \
 --group-map default=64,self_attn.v_proj=128 --exact-scale"

run () {  # run <tag> <extra args...>
  tag=$1; shift
  echo "=== $tag :: $* ==="
  python3 -u src/pot/run_pot.py $COMMON "$@" --tag "$tag" \
    > "llmdocs/rot_logs/${tag}.log" 2>&1
  echo "  exit=$? -> llmdocs/rot_logs/${tag}.log"
}

# ---- 2x2 primary (deterministic) -------------------------------------------
run ROT_pot_none   --codebook strict_pot --rotate none
run ROT_pot_had    --codebook strict_pot --rotate hadamard
run ROT_int4_none  --codebook int4_sym   --rotate none
run ROT_int4_had   --codebook int4_sym   --rotate hadamard

# ---- replication axis: random_hadamard x 3 seeds (seed picks the +-1 draw) --
for s in 0 1 2; do
  run ROT_pot_rh_s$s  --codebook strict_pot --rotate random_hadamard --seed $s
  run ROT_int4_rh_s$s --codebook int4_sym   --rotate random_hadamard --seed $s
done

# =============================================================================
# RESULTS AND POSTMORTEM  -- written AFTER the runs.  Everything above this line
# is the pre-registration, unedited.
# =============================================================================
#
# CONTROL 0 (run first, after ROT_pot_had came back at ppl 153): is --rotate even
# correct?  A fused Hadamard must be exactly equivalence preserving.
#   src/pot/check_rotation.py, fp32:  max|dlogit| 0.0011, argmax agreement 100.000%
#   rotated UNQUANTIZED, same eval path run_pot.py uses:
#     rotate=none      ppl 16.7146   acc 0.7264   acc_norm 0.7050   (bf16 ref
#                                     16.7185 / 0.7264 / 0.7050 -- exact)
#     rotate=hadamard  ppl 16.6596   acc 0.7256   acc_norm 0.7054
#   => deltas are bf16 noise (acc -0.08 pp).  THE ROTATION PATH IS CORRECT, and
#      --rotate-untie-lm-head is NOT required for correctness (the online variant
#      matches).  So the damage below is real, not a fusion bug.
#
# THE 2x2 (RTN, deterministic; int4_sym is a REFERENCE BASELINE, never a deliverable)
#   cell                  bpw        verify   relmse    ppl      acc      acc_norm
#   strict_pot  none   4.244792  ok=True    0.028963   19.921   0.6654   0.6364
#   strict_pot  hadamard 4.244792 ok=True   0.029264  153.655   0.3409   0.3354
#   int4_sym    none   4.244792  ok=null    0.011121   25.884   0.6751   0.6540
#   int4_sym    hadamard 4.244792 ok=null   0.010033   27.966   0.4945   0.4697
#   (int4_sym none re-run as ROT_int4_none2 after the first was SIGTERMed at layer
#    10/28; it reproduces REP_int4_exact exactly: 25.884 / 0.6751 / 0.6540.)
#   deltas on arc.acc:   strict_pot -32.45 pp     int4_sym -18.06 pp
#
# VERDICT AGAINST MY OWN F1-F4, verbatim:
#   F1 "strict_pot + hadamard >= strict_pot + none" -- NOT triggered (0.3409 < 0.6654).
#   F2 "int4_sym + hadamard <= int4_sym + none"     -- *** TRIGGERED ***
#      (0.4945 < 0.6751).  Rotation does not help the uniform grid here either.
#   F3 seed replication -- VACUOUS AXIS, see DESIGN FLAW below.  The one seed run
#      kept as a check, ROT_pot_rh_s0, gives ppl 153.597 acc 0.3375 acc_norm 0.3333
#      against hadamard's 153.655 / 0.3409 / 0.3354, with mean_layer_rel_mse EQUAL
#      to 6 dp (0.029264).  So the collapse is NOT an artifact of the deterministic
#      Sylvester structure -- it reproduces under a random Hadamard draw.
#   F4 mechanism -- the predicted direction held but only at ~1% magnitude, which
#      never licensed a 32 pp claim.
#   => BY MY OWN PRE-REGISTERED CRITERION THE HYPOTHESIS IS FALSIFIED.  Rotation
#      harms BOTH alphabets in this weight-only RTN pipeline.  The -32.45 vs -18.06
#      asymmetry is a difference in how badly two alphabets tolerate the same
#      damage, and must NOT be reported as "the inversion".
#
# DESIGN FLAW found mid-task (my error, caught before it produced a false result):
#   --rotate random_hadamard is H@diag(d), d in {+-1}.  RTN + a sign-symmetric
#   codebook (both strict_pot and int4_sym) + an absmax/exact group scale are ALL
#   equivariant to a sign flip, so quantize(W diag(d)) == quantize(W) diag(d)
#   EXACTLY (verified: max abs diff 0.0).  Under --method rtn, random_hadamard is
#   therefore bit-identical to hadamard and --seed cannot vary anything.  The
#   pre-registered "random_hadamard x 3 seeds" replication axis is VACUOUS; 5 of
#   the 6 seed runs were cancelled.  Confirmed end to end: ROT_pot_rh_s0 and
#   ROT_pot_had have IDENTICAL mean_layer_rel_mse (0.029264), i.e. the stored codes
#   really are the same up to sign.  The residual 0.34 pp ARC difference comes from
#   bf16 accumulation order in the ONLINE R1/R4 Hadamards, not from the codes.  (This does NOT hold under --method gptq, where
#   the Hessian breaks the symmetry -- the existing rot_rh_notie_s1/s2 runs differ.)
#
# MECHANISM (src/pot/analyze_log_magnitude.py, group 64, absmax normalized):
#                       none      hadamard    delta
#   sd_z              1.60707     1.58448    -1.41%   log-magnitude spread narrows
#   kl_unif           0.38547     0.40249    +4.42%   further from log-uniform
#   frac_top          0.07251     0.07815    +7.78%   more mass on the coarse top rung
#   frac_clamp        0.01256     0.01141    -9.17%   *** LESS bottom-rung clamping ***
#   max_over_rms      2.70845     2.60514    -3.81%   rotation does compress, as expected
#   err_pot           0.038711    0.039126   +1.07%
#   err_int4          0.011497    0.010815   -5.93%
#   Every pre-registered quantity moved as predicted EXCEPT frac_clamp, which moved
#   the wrong way.  That positively EXCLUDES the pot4-style mechanism (a range-starved
#   alphabet clamping small weights upward, which collapses to chance at ppl 1826):
#   rotation clamps LESS, not more.  Mechanism excluded, not merely unfound.
#   The confirmed part is a ~1% effect and cannot produce 32 pp.  The run's own
#   mean_layer_rel_mse agrees: 0.028963 -> 0.029264, +1.04%.
#
# WHAT THE UNWEIGHTED METRIC MISSED (scratchpad/full_hess.py, true H = X^T X,
# off-diagonals kept, 4 layers x 7 linears):
#                 tr(EHE)/tr(WHW)    none -> hadamard
#     strict_pot     0.021748 -> 0.040569   +86.5%
#     int4_sym       0.010527 -> 0.011705   +11.2%
#   Rotation acts as an ALIGNMENT EQUALIZER: it forces the quantization error to be
#   isotropic with respect to H, driving the amplification factor tr(EHE)/tr(EE)
#   to ~1.0 for both alphabets (PoT 0.569 -> 1.042, int4 0.836 -> 1.023).  In the
#   NATIVE basis both alphabets sit BELOW 1.0, i.e. both already place their error
#   preferentially in quiet directions -- the log grid more so (0.569), because its
#   absolute error is proportional to |w| and high-variance input channels carry
#   systematically smaller weights (corr(log H_jj, log||W_j||^2) = -0.198, which
#   rotation destroys: -0.004).  So rotation has NO upside here and a real downside
#   for BOTH, larger for PoT because PoT had more alignment to lose.
#   NOTE: this full-H metric was the ONLY instrument that got int4's sign right.
#   Unweighted said int4 improves (-5.93%); diag(H)-weighted said it improves a lot
#   (-35.2%); full-H said it gets slightly worse (+11.2%); the run says it gets worse.
#   The alignment account is POST-HOC -- found after seeing the collapse -- and is
#   labelled as such.
#
# WHY THIS DOES NOT CONTRADICT QuaRot/SpinQuant: they rotate to tame ACTIVATION
# outliers for ACTIVATION quantization, and pair it with GPTQ.  Here activations are
# bf16 and weights are RTN, so there is no error compensation and no activation grid
# to protect -- only the weight-error/H alignment, which rotation can only flatten.
# =============================================================================
