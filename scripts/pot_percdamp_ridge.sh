#!/usr/bin/env bash
# THE PoT-NATIVE OBJECTIVE: how hard should the weight-movement penalty be?
#
# DIAGNOSIS (measured, not assumed):
#   2x2 at identical bpw 4.246867 --  GPTQ-RTN is +1.21 pp for int4_sym, -2.21 pp for
#   strict_pot. GPTQ trades weight-space fidelity for fit to a Hessian estimated from 128
#   sequences. The size of that trade scales with grid coarseness:
#       delta rel_mse   PoT 0.0290 -> 0.0462  (+0.0172)
#                      INT4 0.0111 -> 0.0183  (+0.0072)   ratio 2.39
#       draw-to-draw sd  PoT 0.0366 / INT4 0.0153          ratio 2.39
#   Same ratio twice. PoT overfits the calibration Hessian ~2.4x harder, which produces
#   BOTH the excess variance AND the worse generalization. Independently corroborated:
#   refine_codes (coordinate descent on the SAME objective) scored 0.5995 vs plain GPTQ's
#   0.6073 on matched offsets -- descending tr(EHE') harder made it WORSE.
#
# THE KNOB. percdamp mu makes the objective
#       tr(E (H + mu I) E')  =  tr(E H E')  +  mu ||E||^2
# i.e. an explicit penalty on weight-space movement -- exactly the regularizer the
# diagnosis calls for, and a standard shrinkage estimator for a Hessian estimated from
# limited data. mu -> infinity recovers RTN; mu = 0.01 is stock GPTQ. It has been FROZEN
# AT 0.01 for all 104 runs in this project.
#
# PRE-REGISTERED PREDICTIONS:
#   1. strict_pot's optimal mu is MUCH LARGER than 0.01, and the optimum is INTERIOR --
#      beating both stock GPTQ (0.6012 over 9 draws) and RTN (0.6233 exact).
#   2. int4_sym's optimum stays near 0.01, because its trade is 2.4x smaller. This is the
#      CONTROL that distinguishes "PoT needs more ridge" from "everything needs more
#      ridge" -- without it a PoT-only improvement proves nothing about PoT.
#   KILL: if strict_pot is monotone decreasing in mu, there is no interior optimum, the
#   regularization story is wrong, and RTN stands as the endpoint.
#
# SCREEN ONLY (C4): one draw at offset 0. Generates candidates, ranks nothing. The winner
# goes to a 9-draw confirm before any claim.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-0}
run () { # $1=codebook $2=percdamp $3=tag
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook $1 --group 128 --group-map default=64,self_attn.v_proj=128 \
    --act-order --method gptq --nsamples 128 --bpw-budget 4.2501 --percdamp $2 \
    --tasks ppl,arc_easy --calib-offset 0 --verify-layers all --tag $3 \
    > $D/$3.log 2>&1
  echo "$(date +%H:%M:%S) done $3 rc=$?" >> $D/run_pd.out
}
for mu in 0.03 0.1 0.3 1.0 3.0 10.0; do run strict_pot $mu PD_pot_mu$mu; done
# the control: same sweep on a uniform grid, where the trade is 2.4x smaller
for mu in 0.1 1.0 10.0; do run int4_sym $mu PD_int4_mu$mu; done
echo PD_SWEEP_COMPLETE >> $D/run_pd.out
