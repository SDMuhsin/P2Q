#!/usr/bin/env bash
# ============================================================================
# EQ_* : activation-aware, PoT-native column equalization (AWQ/SmoothQuant/DFQ
#        *insight*, transcribed to a log grid). src/pot/equalize.py
#
# PRE-REGISTERED BEFORE ANY RUN.  Do not edit below this line after the first run.
# ----------------------------------------------------------------------------
# THE LEVER.  A diagonal reparametrization W -> W diag(c)^-1, x -> diag(c) x is exact
# and absorbable upstream.  ABSORPTION ANALYSIS for Qwen3-1.7B (derived in
# pot.equalize.qwen3_equalize_plan, unit-tested in tests/pot/test_equalize.py):
#   q,k,v    share input_layernorm's output  -> ONE shared c, folded into its fp weight
#   gate,up  share post_attention_layernorm  -> ONE shared c, folded into its fp weight
#   o_proj   input = repeat_kv(V) reordered  -> c folded into v_proj's output-row group
#            scales; GQA (16 q heads / 8 kv heads) forces c to be constant within each
#            query-head group, i.e. 1024 free params for 2048 columns
#   down     input = silu(gate(x)) * up(x), LINEAR in up(x) -> c folded into up_proj's
#            output-row group scales, fully per-channel (6144 params)
# So all seven linears absorb for zero stored bits; the price is the three tying
# constraints, not bpw.  We nevertheless RUN with --eq-charge-bits 16, i.e. we pay fp16
# for every one of the 11264 exponents per block, so the reported bpw is legal WITHOUT
# relying on the absorption claim at all:  4.244792 + 0.003581 = 4.248373 <= 4.2501.
#
# THE THEOREM THAT RESHAPES THE ARMS.  On a strict-PoT alphabet the *integer* half of
# the lever is provably INERT: a log grid is exactly covariant under multiplication by
# 2^t, so for every weight that does not get pushed onto an end rung the reconstruction
# error is BIT-IDENTICAL before and after the shift (max |dE| = 0.0 over 400k weights,
# t in +-{1,2,3,5}).  The integer lever can only trade clipping at 2^0 against underflow
# at 2^-7, which the per-(row,group) scale already optimizes.  All of the lever's power
# therefore lives in the REAL-valued diagonal -- which is equally free here, because the
# absorption targets are fp.  This is the exact opposite of the uniform-grid intuition.
#
# WHAT MAKES IT AWQ-LIKE, AND THE SUBTLETY.  For a FIXED group scale r the columns are
# independent and the salience h_j is a positive per-column constant, so it CANCELS from
# the per-column argmin.  Activation-awareness can only act through the COUPLING: r is
# shared by 64 columns, and an h-weighted fit of r lets the salient columns set it while
# the free per-column lever repairs the rest.  That is the AWQ trade on this grid.
#
# ARMS (all --method rtn, otherwise the incumbent config verbatim):
#   A  EQ_int_noh    --equalize int                 H-free, deterministic.  PREDICTED
#                                                   INERT by the theorem: a falsifiable
#                                                   prediction, not a candidate.
#   B  EQ_real_noh   --equalize real                H-free, deterministic.  The candidate.
#   C  EQ_real_h     --equalize real --eq-hess      adds the calibration salience.
#   D  EQ_int_h      --equalize int --eq-hess       completeness: isolates the h-weighted
#                                                   group scale with an inert lever.
#
# INCUMBENT (deterministic, one run = the answer):
#   acc 0.6654040404  acc_norm 0.6363636364  bpw 4.244792  verify ok
#
# PRE-REGISTERED DECISION RULE.  Rank on arc.acc, report arc.acc_norm (ARC-E detection
# limit; config-to-config scatter among reasonable configs is sd 1.83 pp).
#   * An arm COUNTS AS A WIN only if BOTH:
#       (1) arc.acc exceeds the incumbent by more than 1.83 pp, i.e. acc >= 0.6837; and
#       (2) it REPLICATES on an independent axis -- rerun under uniform g=64 (--group 64,
#           no --group-map) and require the same SIGN of delta against the g=64 incumbent
#           (exactly the protocol that validated --exact-scale).
#   * Arm C is H-dependent, so it additionally needs >= 3 --calib-offset draws
#     (0 / 128 / 256) and is judged on the 3-draw MEAN, never a single draw.
#   * Arm A is a PREDICTION CHECK: |delta acc| <= 0.5 pp confirms the covariance theorem
#     end to end.  A large delta in EITHER direction falsifies my analysis and must be
#     reported as such.
#   * Every arm must pass --verify-layers all with verify.ok == true or it is void.
#   * No hyperparameter search: --eq-iters is fixed at 4 and the integer range is fixed
#     at +-8 (= the alphabet's octave span).  Nothing else is tunable.
# ============================================================================
# ---------------------------------------------------------------------------
# ADDENDUM (appended AFTER registration, BEFORE the arms were judged; the decision
# rule above is untouched). Two implementation facts discovered while launching:
#
# 1. The -4.8 to -10.5 % weight-MSE gains quoted from the CPU probe were measured on
#    128-ROW SUBSETS and OVERSTATE the lever by ~10x. At full row count the column
#    diagonal is shared by all 2048 output rows while each (row, group) has its own
#    scale, so the lever's leverage falls proportionally. Measured on block 0 at full
#    size: qkv -0.40 %, gate/up -0.08 %, o -0.60 %, down -0.71 %. That is SMALLER than
#    the 0.9 % relMSE for which search_col_scale was written off. Recorded before any
#    accuracy number was seen.
# 2. The inner r half-step uses the 201-point grid, not the exact breakpoint solver:
#    exact is ~30x the work in fp64 and measured at ~12 h for the model. The ARTIFACT's
#    scale is still exact (--exact-scale is passed to the sweep); only the choice of c
#    rests on the grid.
# Also: GPU 0 is shared, so arms run ONE at a time (a 2-concurrent attempt OOM'd).
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.."
COMMON="--model Qwen/Qwen3-1.7B --method rtn --nsamples 128 --bpw-budget 4.2501 \
 --tasks ppl,arc_easy --verify-layers all --codebook strict_pot --exact-scale"
INCUMBENT_GROUPING="--group 128 --group-map default=64,self_attn.v_proj=128"
UNIFORM_GROUPING="--group 64"

run () {  # run <tag> <extra args...>
  local tag="$1"; shift
  echo "=== $tag ==="
  python3 -u src/pot/run_pot.py $COMMON --tag "$tag" "$@" 2>&1 | tail -25
}

case "${1:-main}" in
main)
  run EQ_int_noh    $INCUMBENT_GROUPING --calib-offset 0 --equalize int  --eq-charge-bits 16
  run EQ_real_noh   $INCUMBENT_GROUPING --calib-offset 0 --equalize real --eq-charge-bits 16
  run EQ_real_h     $INCUMBENT_GROUPING --calib-offset 0 --equalize real --eq-hess --eq-charge-bits 16
  run EQ_int_h      $INCUMBENT_GROUPING --calib-offset 0 --equalize int  --eq-hess --eq-charge-bits 16
  ;;
draws)   # arm C is H-dependent: >= 3 disjoint calibration draws
  for off in 128 256; do
    run EQ_real_h_off$off $INCUMBENT_GROUPING --calib-offset $off --equalize real --eq-hess --eq-charge-bits 16
  done
  ;;
replicate)  # independent axis: uniform g=64
  run EQ_g64_base   $UNIFORM_GROUPING --calib-offset 0
  run EQ_g64_real   $UNIFORM_GROUPING --calib-offset 0 --equalize real --eq-charge-bits 16
  ;;
int4ref)    # labelled reference baseline ONLY, never a deliverable
  run EQ_int4_base  $INCUMBENT_GROUPING --calib-offset 0 --codebook int4_sym
  run EQ_int4_real  $INCUMBENT_GROUPING --calib-offset 0 --codebook int4_sym --equalize real --eq-charge-bits 16
  ;;
esac
