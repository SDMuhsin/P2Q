#!/bin/bash
# THE DECIDING RUN.
#
# The +13.60 pp learned-scale effect was measured on ONE calibration draw (offset 0):
#   LSA_base 0.5349 -> LSA_learn 0.6709.
# But the learned-scale arm turns out to be unstable across draws: LSA_learn is 0.6709 at
# offset 0 and 0.5564 at offset 128, a range of 11.45 pp. Scored against S_g64_ao's sd of
# 3.00 pp the effect is 4.5 SD and safe; scored against the learned-scale arm's own implied
# sd (~8 pp) it is ~1.7 SD and not resolvable.
#
# We have no baseline on a second draw, so we cannot tell whether the +13.60 pp is a real
# effect or two unstable arms passing each other. LSA_base at offset 128 decides it:
#   if LSA_base@128 is also ~0.53, the effect reproduces on a second draw and survives.
#   if LSA_base@128 is near 0.556, the effect vanishes and was a lucky draw.
cd /workspace/CoBALT
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-0/-workspace-CoBALT/c66a966d-95cc-42e7-be88-1ba1cafa00f0/scratchpad

python3 -u src/pot/run_pot.py --codebook strict_pot --group 128 --method gptq \
  --nsamples 128 --calib-offset 128 --tasks ppl,arc_easy --tag LSA_base_off128 \
  > "$S/LSA_base_off128.log" 2>&1
echo "EXIT LSA_base_off128=$?"

# and the same for the scope comparison, which has the same single-draw exposure
python3 -u src/pot/run_pot.py --codebook strict_pot --group 128 --method gptq \
  --learn-scale --learn-scope layer --nsamples 128 --calib-offset 128 \
  --tasks ppl,arc_easy --tag LSA_layer_off128 \
  > "$S/LSA_layer_off128.log" 2>&1
echo "EXIT LSA_layer_off128=$?"
echo DECIDE_COMPLETE
