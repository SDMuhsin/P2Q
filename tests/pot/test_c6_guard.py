"""Charter C6: calibrating on ARC-Easy must be impossible by accident.

`PI_arce_n32` (acc_norm 0.6839) was produced by calibrating on ARC-Easy train and is
VOID. SESSION_2026-09-10_GPTQ_NATIVE.md section 7 flagged the two `arce_*` calibration
sources as a standing hazard; these tests pin the hard fail that closes it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))


@pytest.mark.parametrize("key", ["arce_train", "arce_val"])
def test_task_calibration_is_refused_by_default(key, monkeypatch):
    from benchmark_suite import get_calibration_data
    monkeypatch.delenv("POT_ALLOW_TASK_CALIBRATION", raising=False)
    with pytest.raises(RuntimeError, match="C6"):
        get_calibration_data(None, n_samples=1, seq_len=16, dataset_key=key)


@pytest.mark.parametrize("key", ["arce_train", "arce_val"])
def test_wrong_override_value_is_still_refused(key, monkeypatch):
    from benchmark_suite import get_calibration_data
    monkeypatch.setenv("POT_ALLOW_TASK_CALIBRATION", "yes")
    with pytest.raises(RuntimeError, match="C6"):
        get_calibration_data(None, n_samples=1, seq_len=16, dataset_key=key)


def test_the_guard_gates_exactly_the_two_banned_keys():
    """The guard is keyed on the dataset name, so pin which names it gates.

    Asserting the branch condition rather than exercising the loaders keeps this test
    offline and fast, and it is the property that matters: no legal corpus may reach
    the raise, and neither banned one may bypass it.
    """
    import inspect
    from benchmark_suite import get_calibration_data
    src = inspect.getsource(get_calibration_data)
    assert "if dataset_key in ('arce_train', 'arce_val'):" in src
    # exactly one gated branch, and the raise lives inside it
    assert src.count("POT_ALLOW_TASK_CALIBRATION") == 2      # the check and the message
    head, tail = src.split("if dataset_key in ('arce_train', 'arce_val'):", 1)
    assert "POT_ALLOW_TASK_CALIBRATION" not in head, "guard must not gate legal corpora"
