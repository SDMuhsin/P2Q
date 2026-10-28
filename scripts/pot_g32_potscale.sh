#!/bin/bash
# The fully-shift dequantisation path, and the strongest form of the immutable condition.
#
# Every arm measured so far has PoT *weights* but a floating-point per-group scale, so the
# rescale after accumulation is still a multiply. With --pot-scale the scale is itself 2^k,
# so weight x scale is an exponent add and NOTHING anywhere in the layer is a multiply.
#
# --scale-bits 8 is now backed by the artifact: run_pot.py's check_scale_storable verifies
# every stored scale is exactly representable in 8 bits (frexp mantissa, no tolerance) and
# marks success:false otherwise. Real budget: 4 + 8/32 + codebook = 4.2500 bpw.
#
# Prediction on record: RAPQ (IJCAI'22) reports PoT scales are costly (MobileNetV2 W2/A4:
# 2.64 top-1 naive, 48.12 only with a global search), and our own weight-error measurement
# gave 0.03815 with a PoT scale vs 0.03137 with a float scale. So a loss is expected; the
# question is whether it is affordable.
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

python3 -u src/pot/run_pot.py --codebook strict_pot --group 32 --scale-bits 8 --pot-scale \
  --act-order --method gptq --nsamples 128 --tasks ppl,arc_easy --tag G32_potscale \
  > "$S/G32_potscale.log" 2>&1
echo "EXIT G32_potscale=$?"

# same, at the winning group size, so the PoT-scale cost is measured against S_g64_ao
python3 -u src/pot/run_pot.py --codebook strict_pot --group 64 --scale-bits 8 --pot-scale \
  --act-order --method gptq --nsamples 128 --tasks ppl,arc_easy --tag G64_potscale \
  > "$S/G64_potscale.log" 2>&1
echo "EXIT G64_potscale=$?"
echo POTSCALE_COMPLETE
