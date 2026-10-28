"""MMLU is scored alongside the panel but must never enter the panel statistic.

`panel_macro` is the project's pre-registered selection statistic and several arms have
already been ranked on it. If an eighth task could leak into that mean, every logged
panel number would silently become incomparable to every new one -- the exact failure
the ARC-Easy history is a lesson in. So the exclusion is asserted, not assumed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pot import eval_mc  # noqa: E402


@pytest.fixture
def fake_scores(monkeypatch):
    """Score every task deterministically without loading a model or a dataset."""
    calls = []
    fixed = {t: 0.5 for t in eval_mc.PANEL_TASKS}
    fixed["mmlu"] = 0.99          # wildly off-panel, so any leak is unmissable

    def fake_eval_task(model, tokenizer, task, **kw):
        calls.append(task)
        acc = fixed[task]
        return {"acc": acc, "acc_norm": acc, "acc_norm_token": acc, "n": 100}

    monkeypatch.setattr(eval_mc, "eval_task", fake_eval_task)
    return calls


def test_panel_macro_ignores_extra_tasks(fake_scores):
    with_extra = eval_mc.eval_panel(None, None, extra_tasks=("mmlu",), verbose=False)
    without = eval_mc.eval_panel(None, None, verbose=False)

    assert with_extra["panel_macro"] == without["panel_macro"] == 0.5
    assert with_extra["panel_micro"] == without["panel_micro"] == 0.5
    assert with_extra["n_items"] == without["n_items"]
    assert "mmlu" not in with_extra["tasks"]
    assert with_extra["extra"]["mmlu"]["headline"] == 0.99
    assert "extra" not in without


def test_extra_tasks_are_not_in_the_panel_and_are_declared():
    assert "mmlu" in eval_mc.EXTRA_TASKS
    assert not set(eval_mc.EXTRA_TASKS) & set(eval_mc.PANEL_TASKS)
    for t in eval_mc.EXTRA_TASKS:
        assert t in eval_mc._HEADLINE, f"{t} has no declared headline metric"
        assert t in eval_mc.CHANCE, f"{t} has no declared chance floor"


def test_mmlu_requests_are_four_single_letter_continuations():
    """The 0-shot hendrycksTest shape: choices live in the CONTEXT, the continuation
    is one letter. Length normalisation is then a no-op, which is why acc is the
    headline rather than acc_norm."""
    docs = [{"question": "2 + 2 = ?", "subject": "abstract_algebra",
             "choices": ["3", "4", "5", "6"], "answer": 1}]
    requests, golds = eval_mc.build_requests("mmlu", docs)

    assert golds == [1]
    assert len(requests) == 4
    assert [k for _, _, k in requests] == [" A", " B", " C", " D"]
    ctx = requests[0][1]
    assert len({c for _, c, _ in requests}) == 1, "context must be shared across choices"
    assert "about abstract algebra" in ctx
    assert ctx.endswith("Answer:")
    for lt, ch in zip("ABCD", ["3", "4", "5", "6"]):
        assert f"{lt}. {ch}" in ctx
