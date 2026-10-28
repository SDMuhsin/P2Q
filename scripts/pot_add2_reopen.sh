#!/usr/bin/env bash
# =============================================================================
# ADD2 / APoT: reopening a line that CHARTER C1 closed by my own narrow reading.
#
# WHY THIS IS BEING REOPENED
# --------------------------
# The PI's standing constraint is, verbatim: "every activation x weight matrix
# operation should be doable without matmuls". CHARTER C1 narrowed that to
# "strict_pot / strict_pot_zero only", i.e. every stored weight is a SINGLE
# +-2^e. That narrowing was mine, not the PI's, and it is stricter than the
# stated constraint.
#
# An additive-PoT level (Li et al., APoT, ICLR'20) is +-2^a +- 2^b. The product
# x * w is then TWO exponent adds and ONE integer add. No multiplier, no matmul.
# `src/pot/verify_shiftfree.py` already machine-checks this class (`verify_add2`,
# shift_free == 'add2'), so legality is verifiable, not asserted.
#
# APoT is the canonical literature fix for exactly the defect this project keeps
# hitting: a single-term PoT ladder is geometric, so its levels crowd near zero
# and its resolution at the top of the range -- where 27.3% of the squared error
# lives -- is terrible, and NO number of bits fixes it because the exponent is an
# integer. Two terms break that floor while staying multiplication-free:
#   strict_pot8   |levels| = 1, .5, .25, .125, .0625, .03125, .015625, .0078125
#   learned_add2  |levels| = .875, .5625, .375, .28125, .21875, .140625, .078125, .02734
# The second is far closer to uniform over the range it must cover.
#
# THE STATE OF THE EVIDENCE, WHICH IS WHY THIS IS WORTH RUNS
# ----------------------------------------------------------
# add2 has been run EXACTLY ONCE in 187 logged runs:
#   HL_learned_add2  acc 0.6869  acc_norm 0.6667  bpw 4.125000  verify.ok=true
# versus the current strict-PoT deliverable:
#   REP_deliverable  acc 0.6654  acc_norm 0.6364  bpw 4.244792  verify.ok=true
# i.e. +2.15 pp acc / +3.03 pp acc_norm at 0.119792 bpw LOWER. That single run is
# a one-draw GPTQ number and is therefore worth very little on its own -- the
# project's single-draw artifact scatter is ~3.8 pp and this log has a documented
# winner's-curse problem. It is a candidate, not a result.
#
# It is also almost certainly UNDERSTATED, because it was run under GPTQ, and
# GPTQ is now measured at -2.21 pp on strict_pot vs RTN at matched bpw.
#
# PRE-REGISTERED
# --------------
# Q1 (the one that matters): does the GPTQ-is-harmful-on-log-grids result carry
#     over to add2? add2 is a much more uniform alphabet, so the prediction that
#     follows from the mechanism is that GPTQ should hurt LESS on add2 than on
#     strict_pot, and may help. Arms A1/A2 measure exactly that, deterministically
#     for RTN and over 3 draws for GPTQ.
# Q2: does --exact-scale, the mechanism that carried strict PoT from 0.6233 to
#     0.6364, replicate on add2? It replicated 6/6 on three independent axes, so
#     it should. If it does not, that is evidence against it being a mechanism.
#
# DECISION RULE, fixed before any run:
#   add2 supersedes strict_pot as the recommended deliverable ONLY IF the best
#   DETERMINISTIC add2 arm beats 0.6364 acc_norm at bpw <= 4.244792 AND passes
#   --verify-layers all. A GPTQ-only win does not count: it is draw-dependent and
#   this project has been burned by exactly that.
#   If it wins, it is reported as a SEPARATE CLASS, clearly labelled "add2: two
#   shifts + one add per weight", never silently merged with the strict-PoT line.
#   The strict single-term result stands on its own regardless of the outcome.
# =============================================================================
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
mkdir -p $D
GPU=${1:-0}
GM="default=64,self_attn.v_proj=128"

# Wait for GPU $GPU to drain. Scans /proc rather than `pgrep -f`, which matches
# its own command line and hangs forever (burned this project twice).
busy () { local n=0
  for p in $(pgrep -f "src/pot/run_pot.py" 2>/dev/null); do
    [ -r /proc/$p/environ ] || continue
    if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU"; then n=$((n+1)); fi
  done; [ $n -gt 0 ]; }
while busy; do sleep 60; done
echo "$(date +%H:%M:%S) GPU $GPU clear, starting" >> $D/run_add2.out

run () { local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy \
    --verify-layers all --group 128 --group-map "$GM" --tag $tag "$@" \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_add2.out
}

# A1: deterministic add2 (RTN, H := I -> draw-independent, one run IS the answer)
run AD_rtn_grid    --codebook learned_add2 --method rtn --calib-offset 0
run AD_rtn_exact   --codebook learned_add2 --method rtn --calib-offset 0 --exact-scale
run AD_apot_exact  --codebook apot_k2_b4   --method rtn --calib-offset 0 --exact-scale
# A2: GPTQ on add2, 3 disjoint draws -- tests whether GPTQ's log-grid harm carries over
for off in 0 128 256; do
  run AD_gptq_off$off --codebook learned_add2 --method gptq --calib-offset $off --exact-scale
done
echo ADD2_COMPLETE >> $D/run_add2.out
