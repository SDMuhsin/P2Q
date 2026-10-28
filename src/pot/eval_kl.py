"""
Task-free behavioural fidelity: KL( bf16 teacher || quantized student ) on generic text.

Every proxy this project has used so far is a *weight-space* measure (`rel_mse`,
`tr(E H E^T)`) or a *likelihood* measure (perplexity). None of them measures the thing a
PTQ method actually wants to preserve: the model's **behaviour**. Perplexity asks "does
the quantized model assign high likelihood to the ground-truth token"; KL asks "does it
produce the distribution the original model produced". Those are different questions, and
only the second is a statement about capability preservation -- a student can match the
teacher's perplexity while disagreeing with it on every token where the teacher is wrong.

C6 compliance: the corpus is **wikitext-2-raw-v1 `test`**, the identical non-overlapping
2048-token windows `eval_ppl.get_wikitext2_testenc` already tokenizes, and no labels are
read. This is generic unlabelled text, exactly as legal as the perplexity eval that has
always run here, and it is NOT the evaluation task. Nothing in this file may ever be
pointed at ARC or any other benchmark.

Aggregate metrics (averaged over every position of every window):

  kl_fwd      sum_v p_t(v) [log p_t(v) - log p_s(v)]      <-- PRIMARY
  kl_rev      sum_v p_s(v) [log p_s(v) - log p_t(v)]
  top1_agree  fraction of positions where argmax p_s == argmax p_t
  top5_agree  fraction of positions where argmax p_t is in the student's top-5
  ce_teacher  teacher next-token CE (a sanity constant, identical across students)

Why the forward direction is the selector: forward KL integrates against the *teacher's*
probability, so it is large exactly when the student has moved mass away from something
the original model was confident about -- which is what losing a capability looks like.
Reverse KL is mode-seeking and is small for a student that has collapsed onto a subset of
the teacher's modes, i.e. it under-penalises the failure mode of interest. Reverse KL and
the agreement rates are reported as secondaries and are not used to select.

## Conditional readouts: why a mean over 300k tokens may be the wrong statistic

A mean per-token KL is dominated by high-entropy positions -- generic next-token
uncertainty where the teacher itself is unsure and nothing about the model's *knowledge*
is at stake. A multiple-choice benchmark reads out something much narrower: whether summed
log-probs still order a handful of short continuations correctly, which is a question about
the model's *confident* predictions. So the aggregate can be flat while the part of the
distribution that carries capability has moved.

This module therefore also reports KL **conditioned on the teacher's own predictive
entropy**, bucketed into deciles:

  decile_edges   the 9 interior entropy quantiles (nats)
  decile_kl      mean forward KL within each teacher-entropy decile, low entropy first
  decile_top1    top-1 agreement within each decile
  decile_frac    fraction of tokens in each decile (~0.1 by construction)
  margin_kl      mean forward KL over tokens where the teacher's top-1 margin p1 - p2
                 exceeds each of MARGIN_THRESHOLDS
  tail_frac      fraction of tokens whose per-token forward KL exceeds each of
                 TAIL_THRESHOLDS nats -- tail mass rather than central tendency

The bucket boundaries are a property of the TEACHER on fixed text, so they are identical
across every artifact compared, without needing to be cached or passed in. This is
essential: deciles re-estimated per student would not be comparable.

Teacher entropy is likewise task-free. It is the bf16 model's own uncertainty on generic
wikitext; no label, no benchmark and no task format enters anywhere. Conditioning on it is
exactly as C6-legal as conditioning on token position would be.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pot.eval_ppl import get_wikitext2_testenc

N_DECILES = 10
#: per-token forward-KL levels (nats) at which tail mass is reported
TAIL_THRESHOLDS = (0.1, 0.5, 1.0, 2.0)
#: teacher top-1 margin (p1 - p2) levels defining "the teacher is confident here"
MARGIN_THRESHOLDS = (0.5, 0.9)


@torch.no_grad()
def eval_kl_fidelity(student, teacher, tokenizer, seqlen=2048, device="cuda",
                     n_windows=None, chunk=256, verbose=True, dump_path=None):
    """
    Behavioural fidelity of `student` against `teacher` on WikiText-2 test.

    Args:
        student:   the quantized HF causal LM (already on `device`).
        teacher:   the unquantized reference, same architecture/tokenizer, on `device`.
        tokenizer: matching tokenizer (only used to fetch the cached tokenized split).
        seqlen:    non-overlapping window length; use the same value as the ppl eval so
                   the two numbers describe the same text.
        n_windows: evaluate only the first N windows (None = the whole test split).
        chunk:     positions per softmax block, to bound the [chunk, vocab] fp32 scratch.
        dump_path: optional .npz to receive the raw per-token vectors (kl_fwd, kl_rev,
                   teacher entropy, teacher top-1 margin, agreement flags), so that
                   further conditional readouts can be computed without re-quantizing.
    Returns:
        dict of floats/lists, plus provenance fields.
    """
    testenc = get_wikitext2_testenc(tokenizer)
    nsamples = testenc.numel() // seqlen
    assert nsamples > 0, f"seqlen={seqlen} longer than the tokenized test split"
    if n_windows is not None:
        nsamples = min(nsamples, int(n_windows))

    s_was_training, t_was_training = student.training, teacher.training
    student.eval()
    teacher.eval()

    # per-token vectors, kept rather than reduced, so the conditional readouts below and
    # anything in the dump are computed from the same single pass.
    v_fwd, v_rev, v_ent, v_marg, v_t1, v_t5 = [], [], [], [], [], []
    tot_ce = 0.0
    n_ce = 0

    it = range(nsamples)
    if verbose:
        it = tqdm(it, desc=f"KL(teacher||student) ({nsamples} x {seqlen})")
    for i in it:
        batch = testenc[:, i * seqlen:(i + 1) * seqlen].to(device)
        s_logits = student(batch).logits[0]
        t_logits = teacher(batch).logits[0]
        labels = batch[0]

        for a in range(0, s_logits.shape[0], chunk):
            b = min(a + chunk, s_logits.shape[0])
            ls = F.log_softmax(s_logits[a:b].float(), dim=-1)
            lt = F.log_softmax(t_logits[a:b].float(), dim=-1)
            pt = lt.exp()
            ps = ls.exp()

            v_fwd.append((pt * (lt - ls)).sum(-1).cpu())
            v_rev.append((ps * (ls - lt)).sum(-1).cpu())
            v_ent.append((-(pt * lt).sum(-1)).cpu())

            t_top2 = pt.topk(2, dim=-1)
            t_arg = t_top2.indices[:, 0]
            v_marg.append((t_top2.values[:, 0] - t_top2.values[:, 1]).cpu())
            v_t1.append((ls.argmax(-1) == t_arg).cpu())
            v_t5.append((ls.topk(5, dim=-1).indices == t_arg[:, None]).any(-1).cpu())

            # teacher next-token CE over the positions in this block that have a target
            hi = min(b, labels.shape[0] - 1)
            if hi > a:
                tgt = labels[a + 1:hi + 1]
                tot_ce += float(-lt[:hi - a].gather(1, tgt[:, None]).sum())
                n_ce += hi - a

    student.train(s_was_training)
    teacher.train(t_was_training)

    fwd = torch.cat(v_fwd).double()
    rev = torch.cat(v_rev).double()
    ent = torch.cat(v_ent).double()
    marg = torch.cat(v_marg).double()
    t1 = torch.cat(v_t1)
    t5 = torch.cat(v_t5)
    n_tok = fwd.numel()

    # ---- conditional readouts -------------------------------------------------------
    # Boundaries come from the TEACHER's entropy alone, so they are the same numbers for
    # every student compared against this teacher on this text.
    qs = torch.linspace(0, 1, N_DECILES + 1, dtype=torch.float64)[1:-1]
    edges = torch.quantile(ent, qs)
    bucket = torch.bucketize(ent, edges)
    decile_kl, decile_top1, decile_frac, decile_ent = [], [], [], []
    for d in range(N_DECILES):
        m = bucket == d
        k = int(m.sum())
        decile_frac.append(k / n_tok)
        decile_kl.append(float(fwd[m].mean()) if k else float("nan"))
        decile_top1.append(float(t1[m].double().mean()) if k else float("nan"))
        decile_ent.append(float(ent[m].mean()) if k else float("nan"))

    margin_kl, margin_top1, margin_frac = {}, {}, {}
    for thr in MARGIN_THRESHOLDS:
        m = marg > thr
        k = int(m.sum())
        margin_frac[str(thr)] = k / n_tok
        margin_kl[str(thr)] = float(fwd[m].mean()) if k else float("nan")
        margin_top1[str(thr)] = float(t1[m].double().mean()) if k else float("nan")

    tail_frac = {str(thr): float((fwd > thr).double().mean()) for thr in TAIL_THRESHOLDS}

    if dump_path:
        import numpy as np
        np.savez_compressed(
            dump_path,
            kl_fwd=fwd.float().numpy(), kl_rev=rev.float().numpy(),
            teacher_entropy=ent.float().numpy(), teacher_margin=marg.float().numpy(),
            top1=t1.numpy(), top5=t5.numpy(),
            decile_edges=edges.numpy())

    return {
        "kl_fwd": float(fwd.mean()),
        "kl_rev": float(rev.mean()),
        "top1_agree": float(t1.double().mean()),
        "top5_agree": float(t5.double().mean()),
        "ce_teacher": tot_ce / max(n_ce, 1),
        "kl_fwd_median": float(fwd.median()),
        "decile_edges": [float(x) for x in edges],
        "decile_ent": decile_ent,
        "decile_kl": decile_kl,
        "decile_top1": decile_top1,
        "decile_frac": decile_frac,
        "margin_kl": margin_kl,
        "margin_top1": margin_top1,
        "margin_frac": margin_frac,
        "tail_frac": tail_frac,
        "n_windows": nsamples,
        "n_tokens": n_tok,
        "seqlen": seqlen,
        "corpus": "wikitext2-raw-v1/test",
    }
