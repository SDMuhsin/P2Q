"""
A multi-task zero-shot multiple-choice panel, used as a RANKING INSTRUMENT.

Why this exists
---------------
ARC-Easy `acc_norm` (2376 items) is the project's deliverable metric, and it cannot
rank artifacts. Two random column permutations of the identical recipe --
`SG_rand` (perm_seed 0) and `SG_rand_s2` (perm_seed 2) -- have `mean_layer_rel_mse`
agreeing to six decimals (0.028884 both) and score 0.6822 vs 0.5572, a 12.5 pp gap.
The binomial standard error on 2376 items is ~1.0 pp, so that gap is not item
sampling: the artifacts differ, or ARC-Easy's response to them is idiosyncratic.
Either way a single 2376-item benchmark is the wrong instrument for a 2.59 pp
question, and three sessions of promote-then-withdraw are the consequence.

This module scores a panel of SEVEN cached benchmarks through the EXACT scoring path
`pot.eval_arc` uses (`_score_batch` is imported, not reimplemented), so the panel and
the deliverable metric differ only in which documents are scored.

ARC-Easy is deliberately NOT in the panel.
    The panel is the selection instrument; ARC-Easy is held out as the test set and
    never influences a decision. This is the discipline the charter's C4/C6 were
    reaching for: the project's two voided headlines (0.6574, 0.6839) were both
    produced by selecting and reporting on the same numbers.

Panel composition (all lm-evaluation-harness conventions, all locally cached)
----------------------------------------------------------------------------
    task         split        docs used   headline    chance
    arc_c        test           1172      acc_norm    0.25
    hellaswag    validation     3000*     acc_norm    0.25
    piqa         validation     1838      acc_norm    0.50
    winogrande   validation     1267      acc         0.50
    openbookqa   test            500      acc_norm    0.25
    boolq        validation     1500*     acc         0.50
    siqa         validation     1954      acc         0.33
    * deterministically subsampled -- see `_subsample`.

Reported statistics, both fixed BEFORE any artifact was scored:
    panel_macro -- unweighted mean of the seven headline accuracies.  PRIMARY.
                   Unweighted so that HellaSwag's item count cannot dominate.
    panel_micro -- item-count-weighted mean of the same seven numbers.  SECONDARY.

Determinism
-----------
The project relies on `artifact -> score` being an exact function (verified three
times independently: identical re-quantizations reproduce ARC-Easy to spread
0.000000). This module preserves that:
  * subsampling uses a fixed `numpy.random.RandomState(_SUBSAMPLE_SEED)` and the
    selected indices are sorted, so the subset is a constant of the code;
  * requests are batched by a deterministic length-sorted, token-budgeted packer.
    The packing depends only on tokenized lengths -- i.e. only on the dataset and
    tokenizer, never on the model -- so every artifact sees byte-identical batches.
  * Qwen3 is a RoPE model and `_score_batch` left-pads with an attention mask, so a
    uniform shift of absolute positions is exactly invariant and padding-composition
    cannot perturb a score.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from tqdm import tqdm

from pot.eval_arc import _score_batch, _gold_index

_SUBSAMPLE_SEED = 1234

# Documents used per task. `None` = the whole split. The two caps exist only to bound
# runtime (HellaSwag is 10042 x 4 requests; BoolQ carries long passages); both are
# fixed constants of the code, never tuned.
_CAPS = {"hellaswag": 3000, "boolq": 1500, "mmlu": 3000}

# Which of acc / acc_norm is the task's headline, following lm-eval's own reporting.
# `acc_norm` (length-normalised) is used where continuations differ in length and the
# harness reports it; `acc` where the continuations are identical across choices
# (winogrande) or single tokens (boolq, siqa).
_HEADLINE = {
    "arc_c": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "openbookqa": "acc_norm",
    "boolq": "acc",
    "siqa": "acc",
    "mmlu": "acc",
}

PANEL_TASKS = ("arc_c", "hellaswag", "piqa", "winogrande", "openbookqa", "boolq", "siqa")

# Tasks scored and reported but DELIBERATELY NOT part of panel_macro/panel_micro.
# `panel_macro` is the project's pre-registered selection statistic and seven arms have
# already been ranked on it; folding an eighth task into that mean would silently
# redefine the instrument and make every logged panel number incomparable. MMLU is
# added here instead because the panel has no knowledge-heavy task -- all seven are
# commonsense or reading -- and 4-bit damage is expected to show differently there.
EXTRA_TASKS = ("mmlu",)

# Random-guess floor per task, and the measured bf16 reference for this model. Both are
# needed to put the panel and ARC-Easy on a COMPARABLE scale: a raw percentage-point
# spread means different things on a 0.25-chance 4-way task and a 0.50-chance binary
# one, so comparing "ARC-E moved 12.5 pp" against "the panel moved X pp" is meaningless
# until each is expressed as a fraction of the headroom that exists above chance.
CHANCE = {"arc_c": 0.25, "hellaswag": 0.25, "piqa": 0.50, "winogrande": 0.50,
          "openbookqa": 0.25, "boolq": 0.50, "siqa": 1.0 / 3.0, "mmlu": 0.25}

# Qwen3-1.7B bf16, measured by this module at the fixed default token budget.
#
# SUPERSEDED 2026-09-14. The SESSION 4 constants below no longer reproduce on this box:
# three independent measurements (run_baseline with and without `use_cache=False` /
# `use_fast=True`, and a direct eval_panel call loaded exactly as run_pot loads) agree
# with EACH OTHER exactly and disagree with the stored values on 5 of 7 tasks by up to
# 0.20 pp. So the deviation is deterministic, not the 0.42 pp batching noise, and none
# of the three candidate causes (use_cache, tokenizer, attn_implementation -- the
# default already IS sdpa) explains it. The most likely remaining cause is an
# environment or library change since SESSION 4, which cannot be recovered after the
# fact.
#
# BF16 is therefore set to the reproducible 2026-09-14 measurement, and the SESSION 4
# values are kept for provenance rather than deleted. Impact on the record is small:
# SESSION 4's headline damage comparison (strict_pot 0.1397 vs int4 0.1429) uses ONE
# reference for BOTH arms, so a shift of this size moves both damages together and
# leaves their 0.0032 difference essentially intact.
BF16_SESSION4 = {"arc_c": 0.4292, "hellaswag": 0.5880, "piqa": 0.7209,
                 "winogrande": 0.6101, "openbookqa": 0.3620, "boolq": 0.7700,
                 "siqa": 0.4498}
BF16 = {"arc_c": 0.4292, "hellaswag": 0.5873, "piqa": 0.7225, "winogrande": 0.6101,
        "openbookqa": 0.3600, "boolq": 0.7713, "siqa": 0.4483}
BF16_MMLU = 0.5460                     # measured 2026-09-14, tag MX_q17_bf16
BF16_ARCE_ACC_NORM = 0.704966          # pot.eval_arc, batch_size 16
CHANCE_ARCE = 0.25


# bf16 panel references, measured by this module at the fixed default token budget.
# One entry per model: a damage fraction is only meaningful against the SAME model's
# own bf16 headline, so a cross-model comparison must never fall back to another
# model's reference. Populate with `run_baseline.py --method none --tasks mc`.
# Measured 2026-09-13 by run_baseline.py --method none --tasks mc at
# --mc-token-budget 1024 (NOT the 4096 used for Qwen3-1.7B): this box could not fit the
# larger budget. Absolute values are therefore not comparable across the two models --
# the packing differs and the evaluator's batching noise floor is 0.42 pp. They ARE the
# correct reference for a damage fraction computed from arms scored at the SAME budget,
# which is what every Qwen3-0.6B arm here uses.
BF16_QWEN3_06B = {"arc_c": 0.3430, "hellaswag": 0.4650, "piqa": 0.6774,
                  "winogrande": 0.5620, "openbookqa": 0.3140, "boolq": 0.6380,
                  "siqa": 0.4012}
BF16_QWEN3_06B_ARCE_ACC_NORM = 0.566919      # ppl 20.9688; mmlu 0.3870

BF16_BY_MODEL = {"Qwen/Qwen3-1.7B": BF16, "Qwen/Qwen3-0.6B": BF16_QWEN3_06B}


def damage_fraction(headlines, ref=None):
    """Mean fraction of the above-chance bf16 headroom that has been destroyed.

    `headlines` maps task -> headline accuracy. 0.0 = bf16, 1.0 = chance. This is the
    statistic on which a panel number and an ARC-Easy number can be compared directly.

    `ref` is the bf16 reference to measure headroom against: either a task->accuracy
    mapping or a model name present in `BF16_BY_MODEL`. It defaults to Qwen3-1.7B's,
    which is correct ONLY for Qwen3-1.7B -- pass the right one for any other model.
    """
    if ref is None:
        ref = BF16
    elif isinstance(ref, str):
        if ref not in BF16_BY_MODEL:
            raise KeyError(
                f"no bf16 panel reference for {ref!r}; measure it with "
                f"run_baseline.py --model {ref} --method none --tasks mc and add it "
                f"to eval_mc.BF16_BY_MODEL. Falling back to another model's reference "
                f"would silently report a meaningless damage fraction.")
        ref = BF16_BY_MODEL[ref]
    vals = [(ref[t] - headlines[t]) / (ref[t] - CHANCE[t])
            for t in headlines if t in ref]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------------------
# Dataset loading + request construction.  One builder per task; each returns
# (requests, golds) with requests = [(doc_index, ctx_str, cont_str), ...], exactly the
# shape `pot.eval_arc.build_arc_requests` produces.
# --------------------------------------------------------------------------------------

def _load(task):
    from datasets import load_dataset
    spec = {
        "arc_c": (("allenai/ai2_arc", "ARC-Challenge"), "test"),
        "hellaswag": (("Rowan/hellaswag",), "validation"),
        "piqa": (("nthngdy/piqa",), "validation"),
        "winogrande": (("allenai/winogrande", "winogrande_xl"), "validation"),
        "openbookqa": (("allenai/openbookqa", "main"), "test"),
        "boolq": (("google/boolq",), "validation"),
        "siqa": (("lighteval/siqa",), "validation"),
        "mmlu": (("cais/mmlu", "all"), "test"),
    }[task]
    ds = load_dataset(*spec[0], split=spec[1])
    return _subsample(ds, _CAPS.get(task))


def _subsample(ds, cap):
    """Fixed-seed subsample, indices sorted so the subset is a constant of the code."""
    if cap is None or len(ds) <= cap:
        return ds
    idx = np.random.RandomState(_SUBSAMPLE_SEED).permutation(len(ds))[:cap]
    return ds.select(sorted(int(i) for i in idx))


def _hs_preprocess(text):
    """lm-eval's HellaSwag text cleanup, verbatim."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def build_requests(task, ds):
    requests, golds = [], []

    if task in ("arc_c", "openbookqa"):
        qfield = "question" if task == "arc_c" else "question_stem"
        for di, doc in enumerate(ds):
            golds.append(_gold_index(doc["choices"]["label"], doc["answerKey"]))
            # ARC prefixes "Question:/Answer:"; OpenBookQA's harness task uses the bare
            # stem as the context. Both are lm-eval's own doc_to_text.
            ctx = (f"Question: {doc['question']}\nAnswer:" if task == "arc_c"
                   else doc["question_stem"])
            for t in doc["choices"]["text"]:
                requests.append((di, ctx, " " + t))

    elif task == "hellaswag":
        for di, doc in enumerate(ds):
            golds.append(int(doc["label"]))
            ctx_ab = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            query = _hs_preprocess(doc["activity_label"] + ": " + ctx_ab)
            for e in doc["endings"]:
                requests.append((di, query, " " + _hs_preprocess(e)))

    elif task == "piqa":
        for di, doc in enumerate(ds):
            golds.append(int(doc["label"]))
            ctx = f"Question: {doc['goal']}\nAnswer:"
            for t in (doc["sol1"], doc["sol2"]):
                requests.append((di, ctx, " " + t))

    elif task == "winogrande":
        # Partial evaluation: the CONTINUATION is shared and the CONTEXT varies, so the
        # comparison is over P(tail | prefix+option). acc_norm is identical to acc here
        # (same continuation string) and is reported only for completeness.
        for di, doc in enumerate(ds):
            golds.append({"1": 0, "2": 1}[doc["answer"]])
            cut = doc["sentence"].index("_")
            tail = doc["sentence"][cut + 1:].strip()
            for opt in (doc["option1"], doc["option2"]):
                requests.append((di, doc["sentence"][:cut] + opt, " " + tail))

    elif task == "boolq":
        for di, doc in enumerate(ds):
            golds.append(1 if doc["answer"] else 0)          # choices are ["no", "yes"]
            ctx = f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:"
            for t in ("no", "yes"):
                requests.append((di, ctx, " " + t))

    elif task == "siqa":
        for di, doc in enumerate(ds):
            golds.append(int(doc["label"]) - 1)               # labels are "1".."3"
            ctx = f"{doc['context']}\nQuestion: {doc['question']}\nAnswer:"
            for t in (doc["answerA"], doc["answerB"], doc["answerC"]):
                requests.append((di, ctx, " " + t))

    elif task == "mmlu":
        # lm-eval's 0-shot hendrycksTest prompt: the subject-primed header, the four
        # choices spelled out in the context, and a single-letter continuation. Scoring
        # a letter rather than the answer text is what makes acc the headline (the
        # continuations are all one token, so length normalisation is a no-op).
        letters = ("A", "B", "C", "D")
        for di, doc in enumerate(ds):
            golds.append(int(doc["answer"]))
            subject = doc["subject"].replace("_", " ")
            lines = [f"The following are multiple choice questions (with answers) "
                     f"about {subject}.", "", doc["question"]]
            lines += [f"{lt}. {ch}" for lt, ch in zip(letters, doc["choices"])]
            lines.append("Answer:")
            ctx = "\n".join(lines)
            for lt in letters:
                requests.append((di, ctx, " " + lt))

    else:
        raise ValueError(f"unknown task {task!r}")

    return requests, golds


# --------------------------------------------------------------------------------------
# Deterministic length-sorted, token-budgeted batching.
# --------------------------------------------------------------------------------------

def _plan_batches(tokenizer, requests, token_budget):
    """
    Group request indices into batches of roughly constant padded-token cost.

    Returns a list of lists of request indices. The plan is a pure function of the
    tokenized lengths, so it is identical for every model -- which is what keeps
    `artifact -> score` an exact function across artifacts.
    """
    lens = [len(tokenizer(c + k, add_special_tokens=False).input_ids)
            for _, c, k in requests]
    # Stable sort by length: ties keep dataset order, so the plan is fully determined.
    order = sorted(range(len(requests)), key=lambda i: (lens[i], i))
    batches, cur = [], []
    for i in order:
        trial = len(cur) + 1
        if cur and trial * lens[i] > token_budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches, lens


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

@torch.no_grad()
def eval_task(model, tokenizer, task, device="cuda", token_budget=4096, verbose=True,
              dump_dir=None):
    """Score one panel task. Returns {"acc", "acc_norm", "acc_norm_token", "n"}."""
    ds = _load(task)
    requests, golds = build_requests(task, ds)
    plan, _ = _plan_batches(tokenizer, requests, token_budget)

    was_training = model.training
    model.eval()

    scores = [None] * len(requests)
    it = tqdm(plan, desc=f"{task} ({len(ds)} docs)") if verbose else plan
    for batch_idx in it:
        chunk = [(requests[i][1], requests[i][2]) for i in batch_idx]
        for i, sc in zip(batch_idx, _score_batch(model, tokenizer, chunk, device)):
            scores[i] = sc

    per_doc = [[] for _ in range(len(ds))]
    for (di, _ctx, cont), (sc, n_tok) in zip(requests, scores):
        nchar, ntok = max(len(cont), 1), max(n_tok, 1)
        per_doc[di].append((sc, sc / nchar, sc / ntok, nchar, ntok))

    def _acc(k):
        return sum(max(range(len(d)), key=lambda i: d[i][k]) == g
                   for d, g in zip(per_doc, golds)) / len(ds)

    if dump_dir is not None:
        import os
        from pot.eval_arc import dump_per_item
        os.makedirs(dump_dir, exist_ok=True)
        dump_per_item(os.path.join(dump_dir, f"{task}.npz"), per_doc, golds)

    if was_training:
        model.train()
    return {"acc": _acc(0), "acc_norm": _acc(1), "acc_norm_token": _acc(2), "n": len(ds)}


@torch.no_grad()
def eval_panel(model, tokenizer, device="cuda", tasks=PANEL_TASKS, token_budget=4096,
               verbose=True, dump_dir=None, extra_tasks=()):
    """
    Score the whole panel.

    `extra_tasks` are scored and returned under "extra", and are EXCLUDED from
    panel_macro / panel_micro / n_items by construction -- those three statistics are
    computed from `tasks` alone, so an extra task cannot move a number that earlier
    arms were already ranked on. See EXTRA_TASKS.

    Returns
        {"tasks": {task: {acc, acc_norm, acc_norm_token, n, headline}},
         "panel_macro": float,      # unweighted mean of headline accuracies  (PRIMARY)
         "panel_micro": float,      # item-weighted mean of the same          (secondary)
         "n_items": int,
         "extra": {task: {...}}}    # reported only, never folded into the above
    """
    def _score(t):
        r = eval_task(model, tokenizer, t, device=device, token_budget=token_budget,
                      verbose=verbose, dump_dir=dump_dir)
        r["headline"] = r[_HEADLINE[t]]
        if verbose:
            print(f"  {t:<11s} {_HEADLINE[t]:<9s} = {r['headline']:.4f}  (n={r['n']})")
        return r

    out = {t: _score(t) for t in tasks}
    heads = [out[t]["headline"] for t in tasks]
    ns = [out[t]["n"] for t in tasks]
    res = {
        "tasks": out,
        "panel_macro": float(np.mean(heads)),
        "panel_micro": float(np.average(heads, weights=ns)),
        "n_items": int(sum(ns)),
    }
    if extra_tasks:
        res["extra"] = {t: _score(t) for t in extra_tasks}
    return res
