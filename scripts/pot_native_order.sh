#!/usr/bin/env bash
# PoT-native GPTQ, arm D: PoT-aware column ORDER for the GPTQ sweep.
# act-order sorts by diag(H) alone, which is exact only on a uniform grid (there every
# column's rounding error is the same Delta/2 and factors out of H_jj * e_j). On a log
# grid e_j is proportional to the column's magnitude, so it does not factor out.
#   --order err     descending H_jj * e_j (e_j measured by an RTN pre-pass)
#   --order errmag  descending H_jj * ||W[:,j]||^2  (pre-pass-free proxy)
#   --order errasc  ascending H_jj * e_j            (falsification control)
# All three cost exactly the same g_idx term as --act-order: bpw-neutral.
set -u
cd /workspace/CoBALT
D=/tmp/claude-0/-workspace-CoBALT/7a387013-dec7-4ebb-8c7f-86ef9b3daae3/scratchpad/potnative
run () {  # $1=gpu $2=order $3=offset
  CUDA_VISIBLE_DEVICES=$1 python3 -u src/pot/run_pot.py \
    --codebook strict_pot \
    --group-map default=64,self_attn.v_proj=128 --order $2 --method gptq \
    --nsamples 128 --bpw-budget 4.2501 --tasks ppl,arc_easy \
    --calib-offset $3 --verify-layers all --tag OD_$2_off$3 \
    > $D/OD_$2_off$3.log 2>&1
  echo "done $2 off$3 rc=$?"
}
"$@"
