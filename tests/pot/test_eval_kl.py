"""`pot.eval_kl`: the task-free behavioural-fidelity metric.

These pin the properties the SELECTION rule depends on -- if KL is going to be used to
choose between quantized artifacts, it has to be a real divergence (zero iff identical,
non-negative, monotone in perturbation size) and it has to be reading generic text and
nothing else. They run on a tiny random causal LM on CPU, so they are cheap and never
touch the model cache.

C6 note: `eval_kl_fidelity` may only ever be pointed at generic unlabelled text. The
corpus is hard-wired to `eval_ppl.get_wikitext2_testenc`; `test_corpus_is_generic_text`
pins that there is no dataset argument through which a benchmark could be substituted.
"""
import inspect

import pytest
import torch
import torch.nn as nn

from pot import eval_kl
from pot.eval_kl import eval_kl_fidelity


class _TinyLM(nn.Module):
    """Minimal HF-causal-LM-shaped stand-in: forward(ids) -> object with .logits."""

    def __init__(self, vocab=37, dim=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, dim, generator=g))
            self.head.weight.copy_(torch.randn(vocab, dim, generator=g))
        self.training_flag = False

    def forward(self, ids):
        logits = self.head(self.emb(ids))
        return type("Out", (), {"logits": logits})()


class _TinyTok:
    name_or_path = "tiny-test-tokenizer"


@pytest.fixture
def patched_corpus(monkeypatch):
    """Substitute a deterministic token stream for the wikitext-2 test split."""
    ids = torch.arange(4 * 16, dtype=torch.long).remainder(37).reshape(1, -1)
    monkeypatch.setattr(eval_kl, "get_wikitext2_testenc", lambda tok, **k: ids)
    return ids


def _run(student, teacher, **kw):
    return eval_kl_fidelity(student, teacher, _TinyTok(), seqlen=16, device="cpu",
                            verbose=False, **kw)


def test_identical_models_give_zero_divergence(patched_corpus):
    m = _TinyLM(seed=1)
    out = _run(m, m)
    assert out["kl_fwd"] == pytest.approx(0.0, abs=1e-6)
    assert out["kl_rev"] == pytest.approx(0.0, abs=1e-6)
    assert out["top1_agree"] == 1.0
    assert out["top5_agree"] == 1.0
    assert out["n_tokens"] == patched_corpus.numel()
    assert out["n_windows"] == 4


def test_divergence_is_non_negative_and_grows_with_perturbation(patched_corpus):
    teacher = _TinyLM(seed=2)
    prev_fwd = prev_rev = -1.0
    prev_agree = 2.0
    for eps in (0.05, 0.25, 1.0):
        student = _TinyLM(seed=2)
        with torch.no_grad():
            g = torch.Generator().manual_seed(7)
            student.head.weight.add_(eps * torch.randn(
                student.head.weight.shape, generator=g))
        out = _run(student, teacher)
        assert out["kl_fwd"] > prev_fwd >= -1.0
        assert out["kl_rev"] > prev_rev >= -1.0
        assert out["top1_agree"] <= prev_agree
        assert 0.0 <= out["top1_agree"] <= out["top5_agree"] <= 1.0
        prev_fwd, prev_rev, prev_agree = out["kl_fwd"], out["kl_rev"], out["top1_agree"]


def test_forward_and_reverse_are_different_directions(patched_corpus):
    """A real asymmetric divergence, not a symmetrized distance dressed up as two keys."""
    teacher = _TinyLM(seed=3)
    student = _TinyLM(seed=4)
    fwd_ts = _run(student, teacher)
    fwd_st = _run(teacher, student)
    assert fwd_ts["kl_fwd"] == pytest.approx(fwd_st["kl_rev"], rel=1e-5)
    assert fwd_ts["kl_fwd"] != pytest.approx(fwd_ts["kl_rev"], rel=1e-3)


def test_chunking_does_not_change_the_result(patched_corpus):
    teacher = _TinyLM(seed=5)
    student = _TinyLM(seed=6)
    a = _run(student, teacher, chunk=3)
    b = _run(student, teacher, chunk=1000)
    for k in ("kl_fwd", "kl_rev", "top1_agree", "top5_agree", "ce_teacher"):
        assert a[k] == pytest.approx(b[k], rel=1e-5, abs=1e-7)


def test_n_windows_truncates(patched_corpus):
    m = _TinyLM(seed=7)
    out = _run(m, m, n_windows=2)
    assert out["n_windows"] == 2
    assert out["n_tokens"] == 2 * 16


def test_models_are_left_in_eval_mode_unchanged(patched_corpus):
    teacher = _TinyLM(seed=8)
    student = _TinyLM(seed=9)
    student.train()
    teacher.eval()
    _run(student, teacher)
    assert student.training is True
    assert teacher.training is False


def test_corpus_is_generic_text(patched_corpus):
    """C6: no dataset/task argument exists, so this eval cannot be aimed at a benchmark."""
    params = inspect.signature(eval_kl_fidelity).parameters
    for banned in ("dataset", "dataset_key", "task", "tasks", "split", "corpus"):
        assert banned not in params
    m = _TinyLM(seed=10)
    assert _run(m, m)["corpus"] == "wikitext2-raw-v1/test"


# --------------------------------------------------------------- conditional readouts
# A mean over ~300k tokens is dominated by high-entropy positions where the teacher is
# unsure and nothing about the model's knowledge is at stake. These pin the machinery
# that lets KL be read out on the confident tail instead.

def test_deciles_partition_the_tokens(patched_corpus):
    teacher = _TinyLM(seed=11)
    student = _TinyLM(seed=12)
    out = _run(student, teacher)
    assert len(out["decile_kl"]) == eval_kl.N_DECILES
    assert len(out["decile_top1"]) == eval_kl.N_DECILES
    assert len(out["decile_edges"]) == eval_kl.N_DECILES - 1
    assert sum(out["decile_frac"]) == pytest.approx(1.0)
    # decile means reconstruct the aggregate mean when weighted by occupancy
    recon = sum(f * k for f, k in zip(out["decile_frac"], out["decile_kl"]))
    assert recon == pytest.approx(out["kl_fwd"], rel=1e-6)
    # deciles are ordered by teacher entropy, low first
    assert out["decile_ent"] == sorted(out["decile_ent"])
    assert out["decile_edges"] == sorted(out["decile_edges"])


def test_decile_edges_depend_only_on_the_teacher(patched_corpus):
    """The whole comparison rests on identical bucket boundaries across artifacts."""
    teacher = _TinyLM(seed=13)
    edges = [_run(_TinyLM(seed=s), teacher)["decile_edges"] for s in (14, 15, 16)]
    for e in edges[1:]:
        assert e == pytest.approx(edges[0], rel=1e-9, abs=1e-12)


def test_tail_and_margin_readouts_are_monotone_and_well_formed(patched_corpus):
    teacher = _TinyLM(seed=17)
    student = _TinyLM(seed=18)
    out = _run(student, teacher)
    tails = [out["tail_frac"][str(t)] for t in eval_kl.TAIL_THRESHOLDS]
    assert all(0.0 <= x <= 1.0 for x in tails)
    assert tails == sorted(tails, reverse=True)      # higher threshold -> less mass
    for thr in eval_kl.MARGIN_THRESHOLDS:
        assert 0.0 <= out["margin_frac"][str(thr)] <= 1.0
    margins = [out["margin_frac"][str(t)] for t in eval_kl.MARGIN_THRESHOLDS]
    assert margins == sorted(margins, reverse=True)


def test_identical_models_are_zero_in_every_decile(patched_corpus):
    m = _TinyLM(seed=19)
    out = _run(m, m)
    assert all(abs(k) < 1e-6 for k in out["decile_kl"])
    assert all(t == pytest.approx(1.0) for t in out["decile_top1"])
    assert all(v == 0.0 for v in out["tail_frac"].values())


def test_dump_round_trips_the_per_token_vectors(patched_corpus, tmp_path):
    np = pytest.importorskip("numpy")
    teacher = _TinyLM(seed=20)
    student = _TinyLM(seed=21)
    dump = tmp_path / "kl.npz"
    out = _run(student, teacher, dump_path=str(dump))
    z = np.load(dump)
    assert z["kl_fwd"].shape == (out["n_tokens"],)
    assert float(z["kl_fwd"].mean()) == pytest.approx(out["kl_fwd"], rel=1e-5)
    assert float(z["top1"].mean()) == pytest.approx(out["top1_agree"], rel=1e-5)
    assert z["teacher_entropy"].min() >= 0.0
    assert 0.0 <= z["teacher_margin"].min() <= z["teacher_margin"].max() <= 1.0
