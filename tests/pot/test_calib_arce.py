"""
Tests for the ARC-Easy-domain calibration sources (`arce_train`, `arce_val`) added to
`benchmarks/benchmark_suite.py::get_calibration_data` / `DATASETS`.

These exist so `src/pot/run_pot.py --calib-dataset arce_train` can calibrate GPTQ on
text that matches the ARC-Easy evaluation domain, instead of generic wikitext2/c4.

CRITICAL correctness requirement: `src/pot/eval_arc.py` evaluates on
`load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")`. Calibrating on that same
split would be direct train/test contamination. These tests assert the calibration
source is wired to 'train' (and 'validation'), never 'test', and that the HF-native
splits are in fact disjoint index sets (they are separate `datasets` splits by
construction, but we don't take that on faith -- we check for accidental id overlap).

Network access (HF dataset + tokenizer download) is required; this mirrors how the
rest of tests/pot/ operates (CPU-only, no GPU).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_ROOT, "src")
_BENCH = os.path.join(_ROOT, "benchmarks")
for p in (_SRC, _BENCH, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch

from benchmark_suite import DATASETS, get_calibration_data  # noqa: E402
import pot.eval_arc as eval_arc  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"


@pytest.fixture(autouse=True)
def _allow_void_task_calibration(monkeypatch):
    """These tests exercise the C6-BANNED `arce_*` calibration path on purpose.

    `benchmark_suite.get_calibration_data` now hard-fails on those two dataset keys
    (charter C6; it is the path that produced the void 0.6839 headline). The tests
    remain valuable as provenance -- they pin what that void run actually did -- so they
    opt in explicitly, which is exactly the gate the guard was built to require.
    """
    monkeypatch.setenv("POT_ALLOW_TASK_CALIBRATION", "i-understand-this-is-void")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, use_fast=True)


def test_eval_split_is_test():
    """Sanity-check our assumption about what eval_arc.py reads, since everything else
    in this file is built on top of it."""
    src = open(eval_arc.__file__).read()
    assert 'split="test"' in src
    assert eval_arc.ARC == ("allenai/ai2_arc", "ARC-Easy")


@pytest.mark.parametrize("key,expected_split", [("arce_train", "train"),
                                                 ("arce_val", "validation")])
def test_dataset_key_points_at_non_test_split(key, expected_split):
    cfg = DATASETS[key]
    assert cfg["name"] == "allenai/ai2_arc"
    assert cfg["subset"] == "ARC-Easy"
    assert cfg["train_split"] == expected_split
    assert cfg["train_split"] != "test"


def test_splits_are_disjoint_id_sets():
    """The HF `ai2_arc`/ARC-Easy train/validation/test splits are constructed as
    disjoint files upstream; verify there is no accidental id overlap rather than
    trusting that by convention."""
    from datasets import load_dataset
    train_ids = set(load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")["id"])
    val_ids = set(load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")["id"])
    test_ids = set(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")["id"])
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) > 0 and len(test_ids) > 0


def test_calibration_tensor_shape_and_dtype(tokenizer):
    n, seq_len = 8, 2048
    t = get_calibration_data(tokenizer, n_samples=n, seq_len=seq_len,
                             dataset_key="arce_train")
    assert t.shape == (n, seq_len)
    assert t.dtype == torch.long


def test_max_available_sequences_is_small(tokenizer):
    """ARC-Easy train is ~2251 short QA docs; assert (and pin) that it cannot supply
    anywhere near the 128 seq_len=2048 sequences the baseline --nsamples uses. This is
    the number reported to the PI as the true ceiling on --calib-offset usability."""
    seq_len = 2048
    t = get_calibration_data(tokenizer, n_samples=1000, seq_len=seq_len,
                             dataset_key="arce_train")
    n_available = t.shape[0]
    assert 0 < n_available < 128, (
        f"expected arce_train to be far short of 128 sequences at seq_len={seq_len}, "
        f"got {n_available} -- if this changed, update the reported ceiling")


def test_offset_produces_disjoint_chunk(tokenizer):
    """Same contiguous-chunk semantics as wikitext2/c4/ptb: offset N drops the first N
    chunks, so offset and offset-0 draws must not share a sequence."""
    seq_len = 2048
    n_samples = 4
    offset = 4
    full = get_calibration_data(tokenizer, n_samples=n_samples + offset, seq_len=seq_len,
                                dataset_key="arce_train")
    assert full.shape[0] >= n_samples + offset, (
        "arce_train doesn't have enough sequences for this offset/nsamples combo; "
        "shrink offset or n_samples in this test")
    base = full[0:n_samples]
    shifted = full[offset:offset + n_samples]
    assert base.shape == shifted.shape == (n_samples, seq_len)
    # No row in `shifted` should be identical to any row in `base`.
    for i in range(n_samples):
        for j in range(n_samples):
            assert not torch.equal(base[i], shifted[j])
