#!/bin/bash
# =====================================================================================
# PRE-REGISTRATION -- write-once header. Committed BEFORE any arm was run.
# Experiment: RF_  "non-greedy discrete refinement from an RTN start"
# =====================================================================================
#
# QUESTION
# --------
# GPTQ helps a uniform grid (+1.21 pp ARC-E) and hurts a strict-PoT log grid
# (-2.21 pp) at matched bpw 4.246867. Two confounded explanations:
#   (i)  the OBJECTIVE is wrong for PoT -- descending tr(E H E^T) is the wrong thing to
#        do on a logarithmic alphabet at all;
#   (ii) the MECHANISM is wrong -- GPTQ's greedy sequential error feedback pushes a
#        rounding residual onto not-yet-quantized columns, manufacturing targets a log
#        grid cannot represent, independent of whether tr(E H E^T) is a good objective.
#
# src/pot/code_refine.py separates them: exact discrete coordinate descent on
# tr(E H E^T) over code assignments (maintained r = H e, exact move cost
# dL = 2 d r[j] + d^2 H[j,j]). It keeps the objective and deletes the mechanism -- it is
# non-greedy, never retargets an unquantized weight, never creates an unrepresentable
# target, and monotonically decreases the true objective. Started from RTN (the best
# known starting point, and one that no error feedback has touched), it isolates (i).
#
# THE WIRING BUG THIS EXPERIMENT FIRST HAD TO FIX
# -----------------------------------------------
# run_pot.py sets `g.H = torch.eye(cols)` when --method rtn, BEFORE quantize and before
# refine_codes, so --method rtn --refine-codes descended tr(E I E^T) = plain weight MSE.
# From an RTN start that is a PROVABLE no-op: RTN already minimises per-entry weight MSE
# given the scale, so no adjacent-level move can strictly improve it. The pre-existing
# invocation therefore measured nothing. New flag --refine-hessian calib keeps the RTN
# initial assignment but hands the refinement the real accumulated (2/n) X^T X.
# Default stays `quantize`, i.e. every prior invocation is bit-identical.
# Proof: tests/pot/test_refine_from_rtn.py (identity-H no-op; trace/defaults identity;
# sweeps=0 anchor; monotone descent; strict-PoT artifact re-verifies).
#
# ARMS (all: Qwen3-1.7B, nsamples 128, --group 128 --group-map default=64,
#       self_attn.v_proj=128, --exact-scale, --method rtn, --verify-layers all,
#       --bpw-budget 4.2501, --tasks ppl,arc_easy)
#   P0  strict_pot, no refine                  -- deterministic (H:=I), 1 run, the incumbent
#   P3  strict_pot, refine calib, 3 sweeps     -- offsets 0,128,256
#   I0  int4_sym,  no refine                   -- deterministic, 1 run, REFERENCE BASELINE
#   I3  int4_sym,  refine calib, 3 sweeps      -- offsets 0,128,256   <== THE CONTROL
#   P6  strict_pot, refine calib, 6 sweeps     -- offset 0 only, optimizer verification
# int4_sym is a LABELLED REFERENCE BASELINE ONLY (charter C1/C2), never a deliverable.
#
# PRE-REGISTERED PREDICTIONS AND DECISION RULE
# --------------------------------------------
# Primary ranking metric: arc.acc. Reported metric: arc.acc_norm. ppl and rel_mse are
# NOT proxies on a log grid and are reported but never used to rank.
# Noise model: per-draw ARC-E sd ~3.8 pp => sem of a 3-draw mean ~2.20 pp; a 3-draw mean
# against a zero-variance baseline has se ~2.20 pp, so 2 se = 4.40 pp.
#   RESOLVED           |delta| >= 4.40 pp
#   DIRECTIONAL        1.83 pp <= |delta| < 4.40 pp  (1.83 pp = config-to-config sd)
#   INDISTINGUISHABLE  |delta| < 1.83 pp
#
#   VALIDITY GATE (checked first): the int4_sym control I3 - I0 must be >= 0. On a
#   uniform grid, descending tr(E H E^T) from RTN SHOULD help (GPTQ gets +1.21 pp there).
#   If I3 - I0 < 0 by more than DIRECTIONAL, the refinement is broken or mis-wired and
#   the PoT arm MUST NOT be interpreted -- debug instead.
#
#   SUPPORTS (i) OBJECTIVE IS WRONG FOR PoT:  sign(P3-P0) < 0 while sign(I3-I0) > 0.
#       => H-based layer-output objectives do not transfer to logarithmic alphabets;
#          redirect all future PoT work to non-H objectives.
#   SUPPORTS (ii) GREEDY MECHANISM IS THE CULPRIT: P3-P0 > 0 (at least DIRECTIONAL),
#       with I3-I0 > 0 too. => build non-sequential discrete refinement for PoT.
#   NEITHER: |P3-P0| < 1.83 pp with the gate passed => the objective is neither the
#       help it is for int4 nor an active harm from an undamaged start; GPTQ's -2.21 pp
#       is then attributable to the mechanism WITHOUT a compensating objective win.
#
# OPTIMIZER VERIFICATION (deterministic, zero scatter, judged independently of accuracy):
#   --refine-trace logs tr(E H E^T) after every sweep for every linear. REQUIRE: the
#   trace is non-increasing everywhere and obj_after < obj_before. If the objective
#   falls while ARC-E does not rise, that is itself the measurement: the proxy is not
#   trustworthy on a log grid, at matched everything.
#
# AMENDMENT 1 (added before any arm was run, on the rotation lane's finding)
# ------------------------------------------------------------------------------------
# Measured with the true H = X^T X (off-diagonals kept), RTN's error is naturally
# ANTI-aligned with the curvature, twice as strongly on a log grid:
#     align := [tr(E H E^T)/tr(E E^T)] / [tr(H)/cols]     (1.0 == random orientation)
#     strict_pot 0.569      int4_sym 0.836
# Rotation drives both to ~1.0 and costs -32.45 pp (PoT) / -18.06 pp (int4). GPTQ
# redirects the error and costs -2.21 pp (PoT) / gains +1.21 pp (int4). One explanation
# fits both: int4 starts near random so there is much to win back; PoT starts at 0.569
# so there is little to gain and much to lose. Every method that has failed on this
# format works by redirecting or randomising an error that was already well placed.
#
# Consequence: tr(EHE) alone is the falsified objective; tr(EHE)/tr(EE) is NOT yet
# falsified. GPTQ minimises tr(EHE) by CHANGING THE TARGETS -- it inflates ||E|| by 43%
# (rel_mse 0.0416 vs RTN 0.0290 even at nsamples 1024) and buys alignment with
# magnitude. Coordinate descent over a FIXED codeword set cannot do that; ||E|| can move
# only within a narrow band (unit-tested: < 1.6x on a synthetic layer, in practice a few
# percent). So this lane is the only clean test of whether alignment can be improved at
# roughly fixed error norm.
#
# Therefore, ADDITIONALLY pre-registered:
#   * report tr(EHE), tr(EE) and align at EVERY sweep for BOTH alphabets
#     (--refine-trace now records obj_trace, err2_trace and h_norm = mean diag(H));
#   * MECHANISM CRITERION, judged independently of ARC-E: refinement "works" if align
#     falls while tr(EE) rises by no more than a few percent. That is improving
#     placement at fixed norm, which nothing in this project has yet achieved;
#   * sharpened wiring control: int4_sym starts at 0.836 with room to improve, so a
#     correctly-wired refinement must show a LARGER align improvement on int4_sym than
#     on strict_pot. If it does not, the refinement is broken -- do not interpret PoT.
#
# NO hyperparameter search. Sweeps 1/3/6 exist only to show the objective is really
# being descended; the headline contrast is refine on/off at 3 sweeps.
# =====================================================================================
set -u
cd /workspace/CoBALT
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES before calling this script}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR="${RF_LOGDIR:-/tmp/pot_rf_logs}"
mkdir -p "$LOGDIR"

COMMON=(--method rtn --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy
        --verify-layers all --group 128
        --group-map default=64,self_attn.v_proj=128 --exact-scale)

run () {  # run <tag> <extra flags...>
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ==="
  python3 -u src/pot/run_pot.py "${COMMON[@]}" "$@" --tag "$tag" \
    > "$LOGDIR/$tag.log" 2>&1
  echo "    rc=$? $(date +%H:%M:%S)"
}

# --- validity gate + deterministic anchors first -------------------------------------
run RF_I0_int4_norefine   --codebook int4_sym   --calib-offset 0
run RF_P0_pot_norefine    --codebook strict_pot --calib-offset 0
for off in 0 128 256; do
  run "RF_I3_int4_ref3_off${off}"  --codebook int4_sym   --calib-offset "$off" \
      --refine-codes --refine-hessian calib --refine-sweeps 3 --refine-trace
done
for off in 0 128 256; do
  run "RF_P3_pot_ref3_off${off}"   --codebook strict_pot --calib-offset "$off" \
      --refine-codes --refine-hessian calib --refine-sweeps 3 --refine-trace
done
run RF_P6_pot_ref6_off0   --codebook strict_pot --calib-offset 0 \
    --refine-codes --refine-hessian calib --refine-sweeps 6 --refine-trace

echo "RF_ALL_COMPLETE"
