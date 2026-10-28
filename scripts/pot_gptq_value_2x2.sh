#!/usr/bin/env bash
# THE ON-TOPIC EXPERIMENT: how much does GPTQ's machinery actually buy, per format?
#
# The brief is "GPTQ is unlikely to be optimal for such a different problem". That is a
# claim about the VALUE OF THE ALGORITHM, and it is directly measurable as a 2x2:
#
#                    strict_pot            int4_sym (reference)
#     GPTQ           0.6141 (5 draws)      0.6603 (5 draws)      <- measured, P3
#     RTN (H := I)   ???                   ???                   <- this script
#
# GPTQ's contribution = GPTQ - RTN, per format. If GPTQ buys INT4 a lot and PoT little,
# the brief's hypothesis is CONFIRMED QUANTITATIVELY and we learn the size of the prize.
# If GPTQ buys PoT just as much, the error-feedback machinery is fine and P3's gap is pure
# alphabet -- which kills the "wrong algorithm" thesis outright. Either way it is decisive.
#
# RTN needs only ONE draw: with --method rtn the Hessian is identity and
# hess_weighted_scale is off, so NOTHING in the pipeline reads the calibration data. The
# artifact is draw-independent BY CONSTRUCTION. offset 128 is run as a CONTROL to verify
# that empirically rather than assert it -- it must match offset 0 exactly.
#
# Identical bpw to the incumbent (4.246867) in all four cells. int4_sym is a LABELLED
# REFERENCE BASELINE under C1, never a deliverable.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-0}
run () { # $1=codebook $2=offset $3=tag
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook $1 --group 128 --group-map default=64,self_attn.v_proj=128 \
    --act-order --method rtn --nsamples 128 --bpw-budget 4.2501 \
    --tasks ppl,arc_easy --calib-offset $2 --verify-layers all --tag $3 \
    > $D/$3.log 2>&1
  echo "$(date +%H:%M:%S) done $3 rc=$?" >> $D/run_2x2.out
}
run strict_pot 0   RTN_pot_off0
run int4_sym   0   RTN_int4_off0
run strict_pot 128 RTN_pot_off128
run int4_sym   128 RTN_int4_off128
echo TWOBYTWO_COMPLETE >> $D/run_2x2.out
