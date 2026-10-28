"""
WikiText-2 perplexity evaluation.

Convention matches the repo's existing `src/ecaq_ppl_test.py::evaluate_ppl`, which is
itself the canonical GPTQ/AWQ/SINQ convention, so numbers here are directly comparable
to published PTQ literature and to the rest of this repo:

  * dataset:   wikitext / wikitext-2-raw-v1, `test` split
  * text:      "\\n\\n".join(testdata['text'])  (the WHOLE split, not a subsample)
  * tokenize:  once, no truncation
  * windows:   non-overlapping chunks of `seqlen`; the trailing partial chunk is dropped
  * loss:      next-token CE within each window (window i does NOT see window i-1)
  * ppl:       exp( sum_i loss_i * seqlen / (nsamples * seqlen) )

Note on the `* seqlen` factor: each window contributes `seqlen - 1` predictions but the
canonical implementation weights every window by `seqlen`. Since all windows have equal
length this is just a constant that cancels in the mean, so the result is identical to
a plain mean-of-window-losses. We keep the exact form for bit-for-bit comparability.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

WIKITEXT2 = ("wikitext", "wikitext-2-raw-v1")

_CACHE = {}


def get_wikitext2_testenc(tokenizer, cache=True):
    """Tokenize the full WikiText-2 raw test split once. Returns a [1, N] LongTensor."""
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    if cache and key in _CACHE:
        return _CACHE[key]
    from datasets import load_dataset

    testdata = load_dataset(*WIKITEXT2, split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids
    if cache:
        _CACHE[key] = testenc
    return testenc


@torch.no_grad()
def eval_wikitext_ppl(model, tokenizer, seqlen=2048, device="cuda", verbose=True):
    """
    WikiText-2 perplexity of `model`.

    Args:
        model:     a HF causal LM (already on `device`); set to eval mode internally.
        tokenizer: matching tokenizer.
        seqlen:    non-overlapping window length (2048 is the literature default).
        device:    where to put the input batches.
    Returns:
        float perplexity.
    """
    testenc = get_wikitext2_testenc(tokenizer)
    nsamples = testenc.numel() // seqlen
    assert nsamples > 0, f"seqlen={seqlen} longer than the tokenized test split"

    was_training = model.training
    model.eval()
    loss_fct = nn.CrossEntropyLoss()

    nlls = []
    it = range(nsamples)
    if verbose:
        it = tqdm(it, desc=f"wikitext2 ppl (seqlen={seqlen})")
    for i in it:
        batch = testenc[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(batch).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    if was_training:
        model.train()
    return ppl.item()
