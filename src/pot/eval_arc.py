"""
ARC-Easy zero-shot accuracy, following the lm-evaluation-harness convention.

Why a local implementation instead of `pip install lm_eval`?
  * zero new dependencies, and it runs fully offline against the already-cached
    `allenai/ai2_arc` snapshot;
  * we control batching, so a full 2376-item run takes ~1 min on an A40 instead of
    the harness's slower generic path;
  * the harness pins its own transformers/datasets versions, which risks disturbing
    the pinned environment this repo depends on.
The scoring is a faithful reimplementation of `lm_eval`'s `arc_easy` task, so the
numbers are directly comparable to published ones:

  context      = "Question: {question}\\nAnswer:"
  continuation = " {choice_text}"
  score        = sum of log P(continuation tokens | context + preceding cont. tokens)
  acc          = argmax over choices of score
  acc_norm     = argmax over choices of score / len(continuation_string)   <- headline

We ALSO report `acc_norm_token` = argmax of score / n_continuation_tokens. That is the
convention used by this repo's own `benchmarks/downstream/eval_arc.py` (which reports
only that single number and calls it "accuracy"), so `acc_norm_token` is the metric to
use when comparing against the CoBALT paper's tables, while `acc_norm` is the one to
use when comparing against published lm-eval numbers. They are NOT the same metric.

Gotchas handled:
  * items have 3, 4 or 5 choices (2365/7/4 of them respectively in the test split);
  * `answerKey` is a letter ("A".."E") for most items but a digit ("1".."4") for ~97
    of them. We resolve the gold index via `choices['label'].index(answerKey)`, which
    is correct for both encodings; a letter<->digit fallback covers stragglers.
  * context/continuation are tokenized JOINTLY (tokenize(ctx+cont) then locate the
    boundary by tokenizing ctx alone) -- this is what the harness does and it matters
    for BPE tokenizers where the leading space merges across the boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

ARC = ("allenai/ai2_arc", "ARC-Easy")
_LETTERS = ["A", "B", "C", "D", "E"]
_DIGITS = ["1", "2", "3", "4", "5"]


def _gold_index(labels, answer_key):
    """Resolve the gold choice index, tolerating letter- vs digit-encoded answerKeys."""
    if answer_key in labels:
        return labels.index(answer_key)
    if answer_key in _LETTERS:                       # gold is a letter, labels are digits
        alt = _DIGITS[_LETTERS.index(answer_key)]
    elif answer_key in _DIGITS:                      # gold is a digit, labels are letters
        alt = _LETTERS[_DIGITS.index(answer_key)]
    else:
        raise ValueError(f"unparseable answerKey {answer_key!r} for labels {labels}")
    return labels.index(alt)


def build_arc_requests(dataset):
    """
    Flatten the dataset into per-choice scoring requests.
    Returns (requests, golds) where each request is
    (doc_index, ctx_str, cont_str) and golds[doc_index] is the correct choice ordinal.
    """
    requests, golds = [], []
    for di, doc in enumerate(dataset):
        texts = doc["choices"]["text"]
        labels = doc["choices"]["label"]
        golds.append(_gold_index(labels, doc["answerKey"]))
        ctx = f"Question: {doc['question']}\nAnswer:"
        for t in texts:
            requests.append((di, ctx, " " + t))
    return requests, golds


@torch.no_grad()
def _score_batch(model, tokenizer, batch, device):
    """
    Score a list of (ctx, cont) pairs. Returns a list of summed continuation logprobs.
    Left-padded so that every sequence ends at its own final token.
    """
    enc_full, enc_ctx = [], []
    for ctx, cont in batch:
        ids_full = tokenizer(ctx + cont, add_special_tokens=False).input_ids
        ids_ctx = tokenizer(ctx, add_special_tokens=False).input_ids
        # Guard against pathological retokenization shrinking the context.
        n_ctx = min(len(ids_ctx), len(ids_full) - 1)
        enc_full.append(ids_full)
        enc_ctx.append(n_ctx)

    maxlen = max(len(x) for x in enc_full)
    pad = tokenizer.pad_token_id or 0
    input_ids = torch.full((len(enc_full), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(enc_full), maxlen), dtype=torch.long)
    for i, ids in enumerate(enc_full):              # LEFT padding
        input_ids[i, maxlen - len(ids):] = torch.tensor(ids)
        attn[i, maxlen - len(ids):] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    logits = model(input_ids, attention_mask=attn).logits.float()
    logprobs = F.log_softmax(logits, dim=-1)

    scores = []
    for i, ids in enumerate(enc_full):
        start = maxlen - len(ids)                   # first real token of this row
        n_cont = len(ids) - enc_ctx[i]
        # token at position p is predicted by logits at p-1
        pred = logprobs[i, maxlen - n_cont - 1: maxlen - 1, :]
        tgt = input_ids[i, maxlen - n_cont: maxlen]
        assert start >= 0
        scores.append((pred.gather(-1, tgt.unsqueeze(-1)).sum().item(), n_cont))
    return scores


@torch.no_grad()
def eval_arc_easy(model, tokenizer, device="cuda", limit=None, batch_size=16,
                  verbose=True, dump_path=None):
    """
    Zero-shot ARC-Easy accuracy.

    Args:
        model, tokenizer: HF causal LM already on `device`.
        limit:      evaluate only the first N documents (fast iteration). None = all 2376.
        batch_size: number of (ctx, cont) pairs scored per forward pass.
        dump_path:  if given, ALSO write the raw per-item, per-choice scoring table to
                    this .npz (see `dump_per_item` for the schema). Purely additive --
                    the returned dict is bit-identical whether or not this is passed.
    Returns:
        {"acc": float, "acc_norm": float, "acc_norm_token": float, "n": int}
    """
    from datasets import load_dataset

    ds = load_dataset(*ARC, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    requests, golds = build_arc_requests(ds)
    was_training = model.training
    model.eval()

    scores = []
    rng = range(0, len(requests), batch_size)
    if verbose:
        rng = tqdm(rng, desc=f"ARC-Easy ({len(ds)} docs)")
    for s in rng:
        chunk = requests[s:s + batch_size]
        scores.extend(_score_batch(model, tokenizer,
                                   [(c, k) for _, c, k in chunk], device))

    # regroup per document: (raw_ll, ll/char, ll/token, nchar, ntok).
    # Elements 3 and 4 are the divisors, carried only so that `dump_per_item` can
    # record them; `_acc` indexes 0..2 only, so the reported metrics are unchanged.
    per_doc = [[] for _ in range(len(ds))]
    for (di, _ctx, cont), (sc, n_tok) in zip(requests, scores):
        nchar, ntok = max(len(cont), 1), max(n_tok, 1)
        per_doc[di].append((sc, sc / nchar, sc / ntok, nchar, ntok))

    def _acc(k):
        return sum(max(range(len(d)), key=lambda i: d[i][k]) == g
                   for d, g in zip(per_doc, golds)) / len(ds)

    if dump_path is not None:
        dump_per_item(dump_path, per_doc, golds)

    if was_training:
        model.train()
    return {"acc": _acc(0), "acc_norm": _acc(1), "acc_norm_token": _acc(2),
            "n": len(ds)}


# --------------------------------------------------------------------------------------
# Per-item score dumping (instrument work; strictly additive)
# --------------------------------------------------------------------------------------
# `acc`/`acc_norm`/`acc_norm_token` reduce a continuous per-choice log-probability to a
# 0/1 argmax, throwing away the margins. Higher-power readouts (margin, choice-NLL,
# Brier) need those margins, and item-level pairing across arms needs them per item.
# `dump_per_item` writes the raw table so any statistic can be computed offline WITHOUT
# re-running the model. It never touches the returned metrics.
#
# Schema (npz, n = #docs, C = max #choices over the split):
#   raw    (n, C) float64  summed continuation log-prob; NaN in padded slots
#   nchar  (n, C) int32    len(continuation string), 0 in padded slots
#   ntok   (n, C) int32    number of continuation tokens, 0 in padded slots
#   nch    (n,)   int32    number of real choices for this doc
#   gold   (n,)   int32    gold choice ordinal
# Note `acc_norm` divides by nchar and `acc_norm_token` by ntok; both divisors are
# stored so the three shipped statistics are exactly reproducible from the dump.

def dump_per_item(path, per_doc, golds):
    """Write the raw per-item, per-choice scoring table to `path` (.npz)."""
    import numpy as np
    n = len(per_doc)
    C = max(len(d) for d in per_doc)
    raw = np.full((n, C), np.nan, dtype=np.float64)
    nchar = np.zeros((n, C), dtype=np.int32)
    ntok = np.zeros((n, C), dtype=np.int32)
    nch = np.zeros(n, dtype=np.int32)
    for i, d in enumerate(per_doc):
        nch[i] = len(d)
        for j, entry in enumerate(d):
            raw[i, j] = entry[0]
            nchar[i, j] = entry[3]
            ntok[i, j] = entry[4]
    np.savez_compressed(path, raw=raw, nchar=nchar, ntok=ntok, nch=nch,
                        gold=np.asarray(golds, dtype=np.int32))
    return path
