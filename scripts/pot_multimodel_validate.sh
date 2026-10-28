#!/usr/bin/env bash
# ==============================================================================
# VALIDATION OF --solve-scale ACROSS MODELS AND TASKS
#
# Background. `SS_mu1` (src/pot/scale_solve.py, --solve-scale 1.0) is the first
# intervention in this project to move ARC-Easy, the 7-task panel, core_6 and
# perplexity the SAME way, at identical 4.244792 bpw with every code bit-identical:
#     ARC-E 0.6364 -> 0.6431   panel_7 0.5362 -> 0.5414   ppl 19.921 -> 17.543
# It is NOT promoted. The pre-registered validation in scripts/pot_ss_validate.sh
# was launched on 2026-09-12 and ALL FOUR ARMS DIED OF CUDA OOM -- three inside
# pot.scale_solve (`cH` is R*in*G float64: 2.4 GB at the old fixed row_chunk=512)
# and one inside the panel evaluator, with 28-30 GB held by other tenants of this
# shared box. No arm produced a number. Both causes are now fixed:
#     * scale_solve plans its row block from free VRAM and halves on OOM (exact at
#       any block size -- tests/pot/test_scale_solve.py::test_row_chunking_is_exact);
#     * every job here waits on src/pot/gpu_gate.py until the memory it needs
#       actually exists, instead of being shrunk to fit. The evaluator is a ranking
#       instrument and its batch packing is a constant of the code; adapting it to
#       dodge an OOM would perturb scores by ~0.4 pp, which is larger than the
#       effects being ranked.
#
# What this script adds beyond a re-run: the two axes the project has never tested.
# Every result in 267 logged runs is Qwen3-1.7B on ARC-Easy. "The codes RTN picks
# are right; the scales are what is left to optimise" is a claim about the FORMAT,
# so it must hold on another size and another architecture or it is a Qwen3-1.7B
# fact. Locally available with weights: Qwen3-0.6B (same family, 0.44B) and
# gemma-2b (different family, 1.98B, GeGLU, 256k vocab, 16384 intermediate).
#
# ------------------------------------------------------------------------------
# PRE-REGISTERED, before any arm ran:
#
#   PRIMARY (replication, Qwen3-1.7B). The criteria are UNCHANGED from
#   pot_ss_validate.sh, which wrote them before its arms OOMed:
#     * PASS iff axis B (uniform g=128) is also positive on core_6, AND the 3-draw
#       ARC-E mean stays above the incumbent's 0.6364, AND the 3-draw core_6 mean
#       stays above 0.4973.
#     * The per-draw SPREAD is reported alongside the mean whatever it shows. If the
#       draws span more than ~4 pp on ARC-E, the method has traded a deterministic
#       artifact for a lottery one, and that is a COST, reported as such.
#     * A win on ARC-Easy alone, with core_6 flat or negative, is NOT promoted --
#       the rule that correctly withdrew strict_pot_zero and SG_rand.
#
#   SECONDARY (generalisation, new). For EACH of Qwen3-0.6B and gemma-2b the
#   contrast is the paired difference SS_mu1 - incumbent at matched bpw, matched
#   codes, matched calibration draw:
#     * The panel is the selection instrument; ARC-Easy stays HELD OUT and is
#       reported but never decides. Both are reported per model regardless.
#     * GENERALISES iff panel_macro is positive on BOTH models.
#     * FORMAT-SPECIFIC (an honest negative) iff it is at or below zero on both --
#       which would mean the mechanism is a Qwen3-1.7B fact, not a property of
#       strict PoT, and SS_mu1 must not be promoted as the latter.
#     * SPLIT (positive on one, negative on the other) is reported as SPLIT and
#       promotes nothing. It is not resolved by running more arms on the winner.
#
#   AMENDMENT 2026-09-13, for a RESOURCE reason, recorded before the amended arms ran.
#   Profiling this box over 2123 samples across ~15 h: median free VRAM 3.4 GB, only
#   0.6% of the time is >= 14 GB free, and the longest stretch above that threshold was
#   a single 30 s poll -- one tenant holds 42.09 GB of 44.42. gemma-2b needs ~20 GB and
#   is therefore NOT RUNNABLE here, so it is dropped and stage 2 keeps Qwen3-0.6B only.
#   The consequence is stated rather than worked around:
#     * The GENERALISES / FORMAT-SPECIFIC / SPLIT verdict above CANNOT BE EVALUATED.
#       It needs two models and only one is reachable. One model decides none of them,
#       and a one-model result must not be reported as if it settled the question.
#     * What survives is a single further data point: does the Qwen3-1.7B effect
#       reappear on a second SIZE of the same family? That is a size axis, not an
#       architecture axis, and is reported as such.
#     * MC_BUDGET drops 4096 -> 1024 so the panel fits the gaps. This is legitimate
#       ONLY because every pre-registered criterion is a WITHIN-model paired
#       difference and both arms of the Qwen3-0.6B pair share the budget, so that
#       contrast stays exact. It does mean Qwen3-0.6B panel numbers are NOT comparable
#       to Qwen3-1.7B's logged ones -- different packing, and the evaluator's batching
#       noise floor is 0.42 pp. Cross-model ABSOLUTE panel comparisons are void; only
#       the within-model deltas may be read.
#     * bf16 is measured per model FIRST. A damage fraction against another model's
#       bf16 reference is meaningless, so eval_mc.damage_fraction now refuses to
#       guess (BF16_BY_MODEL) rather than silently using Qwen3-1.7B's.
#     * The 4.25 bpw ceiling is model-dependent through the group map; precomputed:
#       Qwen3-0.6B 4.241667, Qwen3-1.7B 4.244792, gemma-2b 4.249405. All legal, and
#       all asserted by --bpw-budget at run time rather than trusted here.
#
#   Nothing in this file selects a draw, a task or a model after seeing a number.
# ------------------------------------------------------------------------------
#
# Usage:  bash scripts/pot_multimodel_validate.sh [stage1|stage2|all]
# Resumable: any tag already present in the results log is skipped, so the script
# can be re-run after an interruption (long detached runs on this box do get killed).
set -u
cd "$(dirname "$0")/.."
STAGE="${1:-all}"
mkdir -p logs results/pot/mcdump
QLOG=logs/q_multimodel.log

RUNS=results/pot/pot_runs.jsonl
BASE=results/pot/baselines.jsonl

note () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$QLOG"; }

have_tag () {  # tag file -> 0 if a SUCCESSFUL record already exists
  python3 - "$1" "$2" <<'EOF'
import json, sys
tag, path = sys.argv[1], sys.argv[2]
try:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("tag") == tag and r.get("success", True):
            sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
EOF
}

# gated_run <tag> <need_mb> <results_file> <command...>
#
# Admission is NOT a reservation: the other tenants of this box can allocate at any
# moment, so a job can still be OOMed after it starts. An OOM is therefore retried
# (up to ATTEMPTS) rather than recorded as a result -- it is a fact about the box, not
# about the artifact. Any OTHER failure is left alone and reported, because retrying a
# real error would just hide it. A short poll catches the free windows this box opens
# and closes every few minutes; --stable guards the launch instant.
ATTEMPTS=${ATTEMPTS:-3}
gated_run () {
  local tag=$1 need=$2 store=$3; shift 3
  if have_tag "$tag" "$store"; then note "SKIP $tag (already in $store)"; return 0; fi
  local attempt=1
  while [ $attempt -le $ATTEMPTS ]; do
    local gpu
    gpu=$(PYTHONPATH=src python3 src/pot/gpu_gate.py --need-mb "$need" \
          --poll-s 30 --stable 2 2>>"$QLOG")
    if [ -z "$gpu" ]; then note "GATE FAILED for $tag"; return 1; fi
    note "START $tag on GPU$gpu (needed ${need}MB, attempt $attempt/$ATTEMPTS)"
    # expandable_segments reduces allocator fragmentation, which is what turns a
    # marginal allocation into an OOM when a neighbour has taken most of the card. It
    # changes allocation bookkeeping only -- no kernel, no numerics, no score.
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src \
      PYTORCH_ALLOC_CONF=expandable_segments:True "$@" > "logs/$tag.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then note "DONE  $tag"; return 0; fi
    if grep -q "OutOfMemoryError" "logs/$tag.log"; then
      note "OOM   $tag (attempt $attempt/$ATTEMPTS) -- another tenant took the GPU; requeueing"
      cp "logs/$tag.log" "logs/$tag.oom$attempt.log"
      attempt=$((attempt + 1))
      sleep 120
      continue
    fi
    note "FAIL  $tag rc=$rc: $(grep -oE '[A-Za-z.]*Error[^\"]{0,120}' "logs/$tag.log" | tail -1)"
    return 0
  done
  note "GAVE UP $tag after $ATTEMPTS OOMs"
  return 0
}

# A strict_pot artifact run: RTN + exact scale, the incumbent recipe. Extra flags pass
# through. MCX is set to --mc-extra by the stages that score MMLU; it is empty for
# stage 1, whose comparators (PN_incumbent, ZA3_g128_pot, PN_zero) were scored before
# MMLU existed -- adding a task to only one side of a contrast buys nothing.
MCX=""
pot_run () {
  local tag=$1 model=$2 need=$3; shift 3
  gated_run "$tag" "$need" "$RUNS" \
    python3 -u src/pot/run_pot.py --model "$model" --method rtn --nsamples 128 \
      --bpw-budget 4.2501 --tasks ppl,arc_easy,mc --verify-layers all --exact-scale \
      --codebook strict_pot $MCX "$@" --tag "$tag" \
      --mc-dump "results/pot/mcdump/$tag" --arc-dump "results/pot/mcdump/${tag}_arce.npz"
}

GM=(--group 128 --group-map default=64,self_attn.v_proj=128)

# ============================================================== STAGE 1
# The four arms that OOMed. Same tags, same flags, same pre-registration.
stage1 () {
  note "===== STAGE 1: Qwen3-1.7B replication of SS_mu1 (4 arms, re-run after OOM) ====="
  local M=Qwen/Qwen3-1.7B N=14000
  # axis B: uniform g=128. Baseline ZA3_g128_pot (0.5947 / 0.5207 / 0.4846) is logged.
  pot_run SS_g128_mu1 "$M" $N --calib-offset 0   --group 128 --solve-scale 1.0
  # draw spread: the incumbent reads no calibration data, --solve-scale does.
  pot_run SS_mu1_o128 "$M" $N --calib-offset 128 "${GM[@]}" --solve-scale 1.0
  pot_run SS_mu1_o256 "$M" $N --calib-offset 256 "${GM[@]}" --solve-scale 1.0
  # second codebook
  pot_run SS_zero_mu1 "$M" $N --calib-offset 0 "${GM[@]}" --solve-scale 1.0 \
          --codebook strict_pot_zero
}

# ============================================================== STAGE 2
# bf16 reference, incumbent, and SS_mu1 on two further models. Paired by construction:
# within a model the two artifacts differ ONLY in --solve-scale.
stage2 () {
  note "===== STAGE 2: size axis -- Qwen3-0.6B (gemma-2b dropped: needs ~20 GB) ====="
  # Both arms of the pair are run fresh here with IDENTICAL settings, so the paired
  # contrast is exact even at the reduced token budget. See the AMENDMENT above.
  MCX="--mc-extra"
  local B=(--mc-token-budget 1024)
  local N=5000          # measured peak for this model is ~3.5 GB; 5 GB leaves headroom
  gated_run MM_q06_bf16 $N "$BASE" \
    python3 -u src/pot/run_baseline.py --model Qwen/Qwen3-0.6B --method none \
      --tasks ppl,arc_easy,mc --mc-extra "${B[@]}" --tag MM_q06_bf16
  pot_run MM_q06_rtn Qwen/Qwen3-0.6B $N --calib-offset 0 "${GM[@]}" "${B[@]}"
  pot_run MM_q06_ss1 Qwen/Qwen3-0.6B $N --calib-offset 0 "${GM[@]}" "${B[@]}" --solve-scale 1.0
  MCX=""

  # gemma-2b: NOT dropped on the merits. It is verified working end-to-end (CPU smoke,
  # 18 layers, bpw 4.249407, verify ok) and is the only ARCHITECTURE axis available.
  # It needs ~20 GB, which this box has had for 0.5% of the last 15 hours. Re-enable
  # the moment real capacity exists -- the architecture axis is the more valuable of
  # the two and should be the first thing run on a free GPU.
  #   gated_run MM_g2b_bf16 20000 "$BASE" python3 -u src/pot/run_baseline.py \
  #     --model google/gemma-2b --method none --tasks ppl,arc_easy,mc --mc-extra --tag MM_g2b_bf16
  #   pot_run MM_g2b_rtn google/gemma-2b 20000 --calib-offset 0 "${GM[@]}"
  #   pot_run MM_g2b_ss1 google/gemma-2b 20000 --calib-offset 0 "${GM[@]}" --solve-scale 1.0
}

# ============================================================== STAGE 3
# The MMLU contrast on the flagship model. PN_incumbent and SS_mu1 predate MMLU, so the
# pair is re-run under new tags with --mc-extra. This also buys a free determinism
# check: MX_q17_* must reproduce their predecessors' panel_macro EXACTLY (0.5362 and
# 0.5414), because artifact -> score is an exact function in this pipeline. A mismatch
# would invalidate every cross-arm comparison in the study and is worth more than the
# MMLU number itself.
stage3 () {
  note "===== STAGE 3: MMLU contrast on Qwen3-1.7B (+ determinism re-check) ====="
  MCX="--mc-extra"
  local M=Qwen/Qwen3-1.7B N=16000
  pot_run MX_q17_rtn "$M" $N --calib-offset 0 "${GM[@]}"
  pot_run MX_q17_ss1 "$M" $N --calib-offset 0 "${GM[@]}" --solve-scale 1.0
  MCX=""
}

note "queue start (stage=$STAGE)"
case "$STAGE" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  stage3) stage3 ;;
  all)    stage2; stage1; stage3 ;;   # cheapest-and-reachable first: stage 2 fits
                                    # the gaps this box actually offers, stage 1 does not
  *) echo "usage: $0 [stage1|stage2|stage3|all]"; exit 1 ;;
esac
note "queue finished (stage=$STAGE)"
