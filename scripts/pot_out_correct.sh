#!/usr/bin/env bash
# POST-HOC PER-OUTPUT GAIN / BIAS  (src/pot/out_correct.py)
#
# After the artifact is fixed, fit per output row r:
#     min over (a_r, b_r)  E_x [ ( a_r*(W_hat_r . x) + b_r - W_r . x )^2 ]
# `a` folds into the stored fp16 per-(row,group) scale -- zero extra bits, every code
# bit-identical. `b` is new storage, quantized to 8 bits/row against a per-matrix step
# and charged at 0.003255 bpw, which fits the incumbent's 0.005208 bpw headroom.
#
# CPU/GPU PRECONDITION, run first and passed (28 matrices, layers 0/9/18/27, 32
# wikitext2 sequences, at the incumbent recipe):
#     per-row BIAS removes  20.619% of layer output MSE (median 20.412%)
#     per-row GAIN removes  11.419%
# against a pre-registered kill bar of 3%.
#
# A CORRECTION I OWE THE RECORD: I pre-registered `gain` as an inert control, predicting
# <1% on the strength of SESSION 3.7's "the least-squares gain is 1 exactly". That is a
# WEIGHT-SPACE identity (exact-scale stationarity forces <W_hat,W_hat> = <W_hat,W> per
# group). The gain that matters at inference is the H-weighted one, which no identity
# constrains. My control was mis-derived; its failure says nothing about the bias number.
# I therefore have no control for this measurement and say so rather than claiming one.
#
# ---------------------------------------------------------------------------------
# PRE-REGISTERED, and the two arms are predicted to behave DIFFERENTLY
# ---------------------------------------------------------------------------------
# The distinction is whether the intervention touches the weight error E = W_hat - W.
#
#   OC_bias  does NOT change E at all. Weight MSE is bit-identical to the incumbent's;
#            the output error strictly falls. There is no trade. SESSION 22.2's mechanism
#            for all five previous activation-aware failures -- "strict PoT's RTN error is
#            naturally anti-aligned with H (0.569 vs int4's 0.836) and every method that
#            REDIRECTS it destroys that" -- cannot apply, because nothing is redirected.
#            PREDICTION: null or positive. A clear negative would falsify 22.2's mechanism
#            as the common explanation of those failures, which is worth as much as a win.
#
#   OC_gain  DOES change E: a != 1 moves W_hat off the exact-scale MSE optimum, buying
#            output fit with weight fidelity. That is precisely the trade GPTQ makes, and
#            SESSION 13 measured it at -2.21 pp for strict PoT (+1.21 pp for int4 at the
#            same bpw). This is the same trade at one parameter per ROW instead of one per
#            ELEMENT. PREDICTION: neutral to negative. If negative, it is a third
#            independent confirmation of the 13/22.2 mechanism at a new granularity.
#
#   OC_both  separates them: if `both` ~ `bias`, the gain contributes nothing and the
#            free lever is worthless; if `both` < `bias`, the gain is actively harmful.
#
# Standing caveat kept: 20% of layer output MSE is not 20% of anything that matters.
# SESSION 4b measured a flat basin spanning >=31% of tr(EHE^T) with no metric response,
# and a 20% output-MSE cut sits inside it. The honest prior is null.
#
# ---------------------------------------------------------------------------------
# AMENDMENT 2, 01:58 UTC -- THE GRANULARITY LADDER, added before any OC_ arm ran.
# ---------------------------------------------------------------------------------
# Three corrections of the SAME quantity (output scale) at three granularities:
#
#   --temp-match        1 scalar for the whole MODEL   (folds into final RMSNorm gain)
#   --out-correct gainmean   1 scalar per MATRIX       (folds into the group scales)
#   --out-correct gain       1 scalar per ROW          (folds into the group scales)
#
# All three cost ZERO bits and leave every code bit-identical. They differ only in how
# many parameters are fitted against the calibration Gram: 1, 196, and ~40k.
#
# H* PREDICTS THE COARSE RUNGS HELP AND THE FINEST HURTS. That is the opposite of what
# expressiveness predicts, which is exactly what makes the ladder a test rather than a
# sweep. The reason is SESSION 22.2: strict PoT's RTN error is naturally ANTI-aligned
# with H (tr(EHE)/tr(EE) = 0.569 vs int4's 0.836), and fitting many parameters against H
# REDIRECTS that error and destroys the alignment -- which is how GPTQ (-2.21 pp) and
# exact code refinement (-6.51 pp) both failed, harder the harder they fitted. A single
# scalar cannot redirect anything; it can only rescale. So:
#   * monotone-with-expressiveness (gain > gainmean > temp-match) FALSIFIES H* and says
#     this is just ordinary output-error fitting;
#   * an inverted-U, or coarse-helps/fine-hurts, CONFIRMS it and gives the project its
#     first constructive use of the calibration Hessian on this format.
#
# ---------------------------------------------------------------------------------
# AMENDMENT 1, added 23:07 UTC -- BEFORE any OC_ arm had run, after the first four
# panel arms landed. A new, sharper, and entirely falsifiable prediction.
# ---------------------------------------------------------------------------------
# The panel study turned up a mechanism that this arm speaks to directly. BoolQ's gold
# answer is "yes" 61.53% of the time. Measured rate at which each artifact ANSWERS yes:
#
#     PN_incumbent  63.0%   (mean logP(yes)-logP(no) = 2.26)   acc 0.7693
#     PN_zero       63.1%   (2.19)                             acc 0.7727
#     PN_int4       77.5%   (4.08)                             acc 0.7273
#     PN_rand0      93.3%   (6.34)                             acc 0.6620
#
# So the whole of BoolQ's 11.07 pp spread -- which is what drives `panel_macro`'s
# spread, leave-one-out r going -0.939 -> -0.127 when BoolQ is dropped -- is a COHERENT
# OUTPUT-BIAS SHIFT, not a capability difference. `PN_rand0` says yes to 93% of items;
# its 0.6620 is barely above the 0.6153 of answering yes every time.
#
# A per-output-channel bias is precisely the correction for a coherent output-bias
# shift. So:
#
#   AMENDMENT 1a, 23:08 UTC -- STILL BEFORE ANY OC_ ARM RAN. Amendment 1's prediction
#   was DERIVED WRONG and is withdrawn in full. It predicted BoolQ `pred_yes` would move
#   toward the GOLD rate 0.6153. That is not what the correction targets. `out_correct`
#   fits each layer's output to the UNQUANTIZED layer's output; it knows nothing about
#   labels. The reference is therefore bf16's behaviour, not gold -- and bf16 is ITSELF
#   yes-biased, which I only discovered by measuring it:
#
#       bf16          pred-yes 73.87%   margin 3.4401   acc 0.7713
#       PN_zero       pred-yes 63.07%   margin 2.1875   acc 0.7727
#       PN_incumbent  pred-yes 63.00%   margin 2.2607   acc 0.7693
#       PN_int4       pred-yes 77.47%   margin 4.0762   acc 0.7273
#       PN_rand0      pred-yes 93.33%   margin 6.3362   acc 0.6620
#
#   The incumbent is biased BELOW bf16, int4 and rand0 ABOVE it. BoolQ accuracy is flat
#   for margins at or under bf16's 3.44 and falls off above it, which is why rand0's
#   +2.9-nat drift costs 10 pp while the incumbent's -1.2-nat drift costs nothing.
#
#   CORRECTED PREDICTION, and it is now TWO-SIDED, which makes it a much sharper test:
#     * OC_bias moves the artifact's output distribution TOWARD bf16's, so the margin
#       should move toward 3.44 FROM WHICHEVER SIDE it starts.
#         - strict_pot (2.26, below): margin RISES, pred-yes rises toward 0.739, and
#           BoolQ accuracy is roughly UNCHANGED because that band is flat.
#         - int4_sym  (4.08, above): margin FALLS toward 3.44 and BoolQ accuracy RISES
#           toward ~0.77 from 0.7273.
#       The two reference arms are predicted to move in OPPOSITE directions. A common
#       cause (e.g. the correction just damping all logits) cannot produce that.
#     * The other six panel tasks move by less than their item-sampling se (~1-2 pp).
#     * ARC-Easy is held out and is unconstrained by this prediction either way.
#   FALSIFIED IF: either arm's margin moves AWAY from 3.44, or the two arms move in the
#     same direction.
#
# Read out on the 7-task panel (ranking instrument) with ARC-Easy held out.
set -u
GPU=$1; shift
BASE=(python3 -u src/pot/run_pot.py --method rtn --nsamples 128 --calib-offset 0
      --tasks ppl,arc_easy,mc --verify-layers all --exact-scale
      --group 128 --group-map default=64,self_attn.v_proj=128)

run () { local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU "${BASE[@]}" "$@" --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -20
}

run_int4 () {
  local tag=OC_bias_int4ref
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py --method rtn --nsamples 128 \
     --calib-offset 0 --tasks ppl,arc_easy,mc --verify-layers all --codebook int4_sym \
     --exact-scale --group 128 --group-map default=64,self_attn.v_proj=128 \
     --bpw-budget 4.2501 --out-correct bias --out-bias-bits 8 --tag "$tag" \
     --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
     > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -20
}

for arm in "$@"; do
case $arm in
  gain)     run OC_gain --codebook strict_pot     --bpw-budget 4.2501 --out-correct gain ;;
  gainmean) run OC_gainmean --codebook strict_pot --bpw-budget 4.2501 --out-correct gainmean ;;
  zero_gainmean) run OC_zero_gainmean --bpw-budget 4.2501 --codebook strict_pot_zero --out-correct gainmean ;;
  bias) run OC_bias --codebook strict_pot --bpw-budget 4.2501 --out-correct bias --out-bias-bits 8 ;;
  both) run OC_both --codebook strict_pot --bpw-budget 4.2501 --out-correct both --out-bias-bits 8 ;;
  # Reference control (int4_sym is NOT a deliverable under C1). If the bias helps
  # strict_pot but not int4_sym, the lever is format-specific and the mechanism is the
  # log grid's multiplicative error; if it helps both, it is a generic PTQ correction
  # that this project simply never applied. Either reading is worth one run.
  bias_int4) CB=int4_sym run_int4 ;;
  bias4)  run OC_bias4 --codebook strict_pot  --bpw-budget 4.2501 --out-correct bias --out-bias-bits 4 ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac
done
