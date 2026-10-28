#!/usr/bin/env bash
# PANEL STUDY -- is the artifact-to-artifact ARC-Easy swing real capability, or is it
# idiosyncratic to ARC-Easy?
#
# Motivation, from results/pot/pot_runs.jsonl:
#   SG_rand (perm_seed 0) and SG_rand_s2 (perm_seed 2) are the SAME recipe with a
#   different random column permutation. mean_layer_rel_mse agrees to six decimals
#   (0.028884 both). ARC-Easy acc_norm: 0.6822 vs 0.5572 -- 12.5 pp apart, against a
#   binomial se of ~1.0 pp on 2376 items. Three sessions have called this "noise" and
#   then ranked methods with it anyway.
#
# Each arm is scored on BOTH:
#   * the 7-task panel (pot/eval_mc.py, ~11k items, ARC-Easy deliberately excluded)
#   * ARC-Easy, which is HELD OUT: reported, never used to choose anything.
#
# PRE-REGISTERED READING (written before the first arm was launched):
#   A. panel tracks ARC-Easy (Spearman over the 8 arms high, and the panel's spread
#      between rand0/rand2 comparable in sd units to ARC-Easy's 12.5 pp)
#      => the artifacts genuinely differ; the panel is a lower-variance estimator of
#         the same quantity and becomes the selection instrument.
#   B. panel does NOT track ARC-Easy (Spearman ~0, and rand0 vs rand2 collapses to
#      ~1 pp on the panel)
#      => the swing is an ARC-Easy artefact. rand0/incumbent/rand2 are the same model,
#         the 2.59 pp charter gap is a lottery on one benchmark, and that goes to the
#         user as the headline finding.
#   The control arms make the reading falsifiable rather than a vibe:
#     PN_logobj   -- log-domain scale objective, ARC-E 0.2487 (chance). A panel that
#                    cannot separate a chance-level model is broken, not insensitive.
#     PN_int4     -- int4_sym at IDENTICAL bpw. Labelled reference baseline under C1,
#                    never a deliverable. Known +1.76 pp on ARC-E over the incumbent.
#     PN_incumbent-- must reproduce ARC-E 0.6364 exactly (5th determinism check).
#
# Usage: scripts/pot_panel_study.sh <gpu> <arm> [<arm> ...]
set -u
GPU=$1; shift
BASE=(python3 -u src/pot/run_pot.py --method rtn --nsamples 128 --calib-offset 0
      --bpw-budget 4.2501 --tasks ppl,arc_easy,mc --verify-layers all
      --group 128 --group-map default=64,self_attn.v_proj=128)

run () {  # run <tag> <extra args...>
  local tag=$1; shift
  echo "=== $(date -u +%H:%M:%S) GPU$GPU $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU "${BASE[@]}" "$@" --tag "$tag" \
      --mc-dump results/pot/mcdump/"$tag" --arc-dump results/pot/mcdump/"$tag"_arce.npz \
      > logs/"$tag".log 2>&1
  grep -vE "it/s\]|it\]$" logs/"$tag".log | tail -20
}

for arm in "$@"; do
case $arm in
  incumbent) run PN_incumbent --codebook strict_pot --exact-scale ;;
  rand0)     run PN_rand0     --codebook strict_pot --exact-scale --order rand --perm-seed 0 ;;
  rand2)     run PN_rand2     --codebook strict_pot --exact-scale --order rand --perm-seed 2 ;;
  rand4)     echo "SKIPPED PN_rand4 -- deprioritised in favour of lever arms" ;;
  logmean)   run PN_logmean   --codebook strict_pot --exact-scale --order logmean ;;
  zero)      run PN_zero      --codebook strict_pot_zero --exact-scale ;;
  int4)      run PN_int4      --codebook int4_sym --exact-scale ;;
  brnarrow)  run PN_brnarrow  --codebook strict_pot --scale-lo 0.7 --scale-hi 0.85 ;;
  logobj)    run PN_logobj    --codebook strict_pot --log-scale ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac
done
