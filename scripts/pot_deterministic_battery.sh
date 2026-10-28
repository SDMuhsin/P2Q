#!/usr/bin/env bash
# DETERMINISTIC PoT BATTERY -- every arm here is independent of the calibration draw, so
# each is ONE exact number with no error bars and comparisons need no confirm protocol.
#
# WHY THIS IS THE HIGHEST-VALUE COMPUTE RIGHT NOW. A single-draw ARC-E screen carries
# ~3.3 pp sd; measured in-experiment, FB_damp0.00 (0.5854) and RTN (0.6233) have weight
# error agreeing to 3 significant figures (0.029065 vs 0.028981) and sit 3.8 pp apart. So
# any arm that READS the calibration data needs ~20 draws to rank. Arms that do not read it
# need exactly one run. With --method rtn the Hessian is identity, nothing in the pipeline
# touches calibration, and the artifact is draw-independent BY CONSTRUCTION -- verified
# empirically: RTN_pot off0 and off128 matched to delta 0.0e+00.
#
# The incumbent deterministic point is RTN_pot = 0.6233 at bpw 4.246867 (verify.ok on all
# 196 layers). With H := I the ONLY remaining freedom is where the 8-octave window sits,
# i.e. the group scale -- so these arms probe exactly that, and nothing else.
#
#   exact      : the breakpoint-based exact scale solver. SHIPPED BUT NEVER ONCE RUN in
#                104 runs (exact_scale is false in every logged config). With GPTQ it was
#                one knob among many; with RTN the scale search IS the entire algorithm,
#                so it is now first-class.
#   noao       : drop --act-order. With H := I, diag(H) is constant and the permutation is
#                the identity, so the artifact MUST be unchanged -- but the driver charges
#                g_idx 0.002075 bpw for a non-identity order. If the artifact matches
#                RTN_pot exactly, we have been paying for a no-op and the honest bpw is
#                4.244792. Doubles as a determinism check.
#   wide/narrow: the multiplier bracket [0.3, 1.2] has never been varied in 104 runs.
#   potscale   : constrain the group scale to a power of two, so dequantization is ALSO a
#                shift and nothing in the pipeline multiplies at all. Costs accuracy; the
#                question is how much, because it is what would buy a cheaper scale field.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
GPU=${1:-1}
run () { local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU python3 -u src/pot/run_pot.py \
    --codebook strict_pot --group 128 --group-map default=64,self_attn.v_proj=128 \
    --method rtn --nsamples 128 --bpw-budget 4.2501 \
    --tasks ppl,arc_easy --calib-offset 0 --verify-layers all --tag $tag "$@" \
    > $D/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done $tag rc=$?" >> $D/run_det.out
}
run DET_exact      --act-order --exact-scale
run DET_noao
run DET_wide       --act-order --scale-lo 0.15 --scale-hi 1.5
run DET_narrow     --act-order --scale-lo 0.45 --scale-hi 1.05
run DET_fine       --act-order --n-grid 1000
run DET_potscale   --act-order --pot-scale
echo DET_COMPLETE >> $D/run_det.out
