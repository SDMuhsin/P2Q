"""`--calib-cache`: an on-disk cache of the tokenized calibration tensor.

The generic-corpus loaders re-tokenize the whole training split on every call. wikitext2
costs seconds; wikitext103 costs ~11 minutes, which dominates a ~7 minute quantization run
and is what makes many-draw protocols unaffordable. Because every loader chunks
contiguously from the head of the split with no RNG, the head of a larger cached tensor is
bit-identical to a fresh smaller load -- so caching is a pure speedup.

These tests pin exactly that: same bits out, the cache is never consulted unless asked for,
and a cache that does not cover the request is rebuilt rather than silently padded.
"""
import torch

from pot.run_pot import load_calibration_cached


class _Loader:
    """Stands in for benchmark_suite.get_calibration_data; counts real tokenizations."""

    def __init__(self):
        self.calls = []

    def __call__(self, tok, n_samples, seq_len, dataset_key):
        self.calls.append((n_samples, seq_len, dataset_key))
        # deterministic, contiguous-from-the-head, exactly like the real loaders
        return torch.arange(n_samples * seq_len).reshape(n_samples, seq_len)


def test_no_cache_path_means_no_caching(tmp_path):
    ld = _Loader()
    a = load_calibration_cached(ld, None, "wikitext2", 4, 8, None)
    b = load_calibration_cached(ld, None, "wikitext2", 4, 8, None)
    assert torch.equal(a, b)
    assert len(ld.calls) == 2
    assert not list(tmp_path.iterdir())


def test_second_call_is_served_from_cache_and_is_bit_identical(tmp_path):
    ld = _Loader()
    path = str(tmp_path / "c" / "wt.pt")
    fresh = load_calibration_cached(ld, None, "wikitext103", 6, 8, path)
    cached = load_calibration_cached(ld, None, "wikitext103", 6, 8, path)
    assert torch.equal(fresh, cached)
    assert len(ld.calls) == 1          # the second call did not re-tokenize


def test_a_larger_cache_serves_a_smaller_request_bit_identically(tmp_path):
    """The whole point: build once at the deepest offset, reuse for every draw."""
    ld = _Loader()
    path = str(tmp_path / "wt.pt")
    load_calibration_cached(ld, None, "wikitext103", 10, 8, path)
    small = load_calibration_cached(ld, None, "wikitext103", 3, 8, path)
    direct = _Loader()(None, n_samples=3, seq_len=8, dataset_key="wikitext103")
    assert torch.equal(small, direct)
    assert len(ld.calls) == 1


def test_cache_too_small_is_rebuilt_not_padded(tmp_path):
    ld = _Loader()
    path = str(tmp_path / "wt.pt")
    load_calibration_cached(ld, None, "wikitext103", 3, 8, path)
    big = load_calibration_cached(ld, None, "wikitext103", 9, 8, path)
    assert big.shape == (9, 8)
    assert len(ld.calls) == 2
    # and the rebuilt cache now covers the larger request
    again = load_calibration_cached(ld, None, "wikitext103", 9, 8, path)
    assert torch.equal(big, again)
    assert len(ld.calls) == 2


def test_cache_is_keyed_by_dataset_and_seqlen(tmp_path):
    ld = _Loader()
    path = str(tmp_path / "wt.pt")
    load_calibration_cached(ld, None, "wikitext2", 4, 8, path)
    load_calibration_cached(ld, None, "wikitext103", 4, 8, path)   # different corpus
    load_calibration_cached(ld, None, "wikitext103", 4, 16, path)  # different seqlen
    assert [c[2] for c in ld.calls] == ["wikitext2", "wikitext103", "wikitext103"]
    assert len(ld.calls) == 3
