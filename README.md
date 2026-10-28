# P2Q: Strict Power-of-Two 4-bit Post-Training Quantization

P2Q quantizes a language model so that **every stored weight is a single signed power of
two**, `w = ±2^e`. On a bf16/fp16 activation, `x · 2^e` is an integer add into the
exponent field, so the weight-activation product needs no multiply. A matmul reduces to
exponent-adds and adds.

Target model: `Qwen/Qwen3-1.7B`. Post-training only, no retraining, at most 4.25 bits per
weight with all metadata counted (codes, group scales, codebook).

## Method

1. **Codebook.** Strict PoT: 1 sign bit and 3 exponent bits, giving `±2^e` for
   `e ∈ {0..-7}` times a per-group scale.
2. **Codes by round-to-nearest.** Each weight maps to its nearest codeword (RTN with a
   codebook oracle). GPTQ error feedback was tried and is harmful on a logarithmic grid, so
   the incumbent uses none, which also makes it deterministic.
3. **Scale search.** Each group's scale is chosen by exact search rather than a grid.
4. **Grouping.** Group size 64, with `self_attn.v_proj` at 128 to stay inside the budget.
5. **`--solve-scale` (candidate).** With codes frozen, a row's output is linear in its
   group scales, so the calibration-weighted output error has a closed-form per-row solve.
   It adds zero bits. Its gain is small and not yet established across models.
6. **Verification.** `verify_shiftfree.py` checks every layer for a bit-exact
   `code * scale == W_hat` witness, so the multiply-free claim is machine-checked.

## Results (Qwen3-1.7B, 4.2448 bpw)

| | ARC-Easy acc_norm |
|---|---|
| bf16 | 0.7050 |
| P2Q incumbent (RTN, exact scale) | 0.6364 |

ARC-Easy is noisy on this problem, so selection uses a seven-task panel with ARC-Easy
held out. The multiply-free claim concerns the numeric format only. No inference kernel
exists yet.

## Where things are

| Path | Contents |
|---|---|
| `src/pot/run_pot.py` | main entry point: quantize, verify, evaluate |
| `src/pot/codebooks.py` | strict PoT and relaxed codebooks |
| `src/pot/gptq_pot.py` | quantizer with a codebook oracle (RTN and GPTQ paths) |
| `src/pot/exact_scale.py`, `scale_search.py` | group scale selection |
| `src/pot/scale_solve.py` | `--solve-scale` closed-form scale re-solve |
| `src/pot/verify_shiftfree.py` | multiply-free verifier |
| `src/pot/eval_mc.py`, `eval_ppl.py`, `eval_arc.py` | evaluators |
| `tests/pot/` | test suite |
| `scripts/pot_*.sh` | experiment sweeps |
| `benchmarks/benchmark_suite.py` | calibration data loader |

## Reproduce the incumbent

```bash
python3 -u src/pot/run_pot.py --codebook strict_pot --method rtn --exact-scale \
  --group-map default=64,self_attn.v_proj=128 \
  --bpw-budget 4.2501 --tasks ppl,arc_easy --verify-layers all --tag REP_deliverable
```

## Contact

Sayed Muhsin, sdmuhsin@gmail.com
