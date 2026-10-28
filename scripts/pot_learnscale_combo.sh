#!/bin/bash
# TOP PRIORITY. The learned scale residual (PoT-PTQ's step 2, which we were missing) turns
# out to be the largest single lever found:
#
#   config                                   ppl     acc     acc_norm   bpw
#   g=128 baseline                          23.393  0.5901   0.5349    4.1250
#   g=128 + learned scale residual          17.834  0.6890   0.6709    4.1250   (+13.6 pp)
#   g=64  + act-order (previous best)       17.751  0.6797   0.6574    4.2500
#   RTN-INT4 (NOT shift-free)               20.35   0.6780   0.6696    4.1875
#
# LSA_learn reached INT4 parity WITHOUT act-order and at g=128, i.e. without either of the
# two knobs already known to help. This script combines them.
#
# Only the SCALES are learned; codes stay on the power-of-two grid and the shift-free
# verifier runs on the artifact, so the immutable condition is preserved (verify ok on
# LSA_learn, scale_storage_ok true).
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

r(){ tag=$1; shift
     python3 -u src/pot/run_pot.py --codebook strict_pot --method gptq --nsamples 128 \
       --learn-scale --learn-scope block --tasks ppl,arc_easy --tag "$tag" "$@" \
       > "$S/$tag.log" 2>&1
     echo "EXIT $tag=$?"; }

# the one that matters: learned scale + act-order + the winning group size
r L_g64_ao_ls   --group 64  --act-order
# learned scale + act-order at g=128, to separate the group-size contribution
r L_g128_ao_ls  --group 128 --act-order
# learned scale at g=64 without act-order, completing the 2x2 against LSA_learn
r L_g64_ls      --group 64
echo LEARNSCALE_COMBO_COMPLETE
