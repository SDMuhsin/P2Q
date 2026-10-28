"""End-to-end P2Q driver: layer-sequential GPTQ with a shift-free codebook.

Standard GPTQ pipeline: capture the decoder-layer-0 inputs once, then walk layers one at
a time, building each Linear's Hessian from that layer's real inputs, quantizing, and
re-running the (now quantized) layer to produce the next layer's inputs. One pass per
layer, not one per layer per matrix.

Usage:
  python3 src/pot/run_p2q.py --codebook strict_pot --group 128
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pot.codebooks import get_codebook
from pot.gptq_pot import PoTGPTQ
from pot.verify_shiftfree import verify_codebook_artifact
from pot.eval_ppl import eval_wikitext_ppl
from pot.eval_arc import eval_arc_easy

DEV = 'cuda'
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def find_linears(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)
            and n.split('.')[-1] in TARGETS}


def accounted_bpw(codebook_bits, group, K, n_entries, weights_per_layer,
                  scale_bits=16, zero_bits=0, col_scale=False):
    """Fully accounted bits per weight: codes + group scale + zero-pt + col scale + codebook."""
    terms = {
        'codes': codebook_bits,
        'group_scale': scale_bits / group,
        'zero_point': zero_bits / group,
        'col_scale': (16.0 / K) if col_scale else 0.0,
        'codebook': n_entries * 16.0 / weights_per_layer,
    }
    terms['total'] = sum(terms.values())
    return terms


@torch.no_grad()
def get_calib(tok, nsamples, seqlen):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    enc = tok("\n\n".join(ds['text']), return_tensors='pt').input_ids
    g = torch.Generator().manual_seed(42)
    out = []
    for _ in range(nsamples):
        i = torch.randint(0, enc.shape[1] - seqlen - 1, (1,), generator=g).item()
        out.append(enc[:, i:i + seqlen])
    return out


@torch.no_grad()
def quantize_model(model, batches, cb, args):
    layers = model.model.layers
    model.model.embed_tokens.to(DEV)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to(DEV)

    hidden = model.config.hidden_size
    n = len(batches)
    inps = torch.zeros(n, args.seqlen, hidden, dtype=torch.bfloat16, device=DEV)
    cache = {'i': 0, 'kw': None}

    class Catcher(nn.Module):
        def __init__(self, mod):
            super().__init__(); self.module = mod
            # transformers 4.55 reads this off the layer object itself
            self.attention_type = getattr(mod, 'attention_type', 'full_attention')
        def forward(self, hidden_states, **kw):
            inps[cache['i']] = hidden_states
            cache['i'] += 1
            cache['kw'] = kw
            raise ValueError

    layers[0] = Catcher(layers[0]).to(DEV)
    for b in batches:
        try:
            model(b.to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].module.cpu()
    kw = cache['kw']
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    stats = []
    for li in range(len(layers)):
        t0 = time.time()
        layer = layers[li].to(DEV)
        subset = find_linears(layer)
        gq = {nm: PoTGPTQ(m) for nm, m in subset.items()}
        hs = [m.register_forward_hook(
                (lambda nm: lambda mod, inp, out: gq[nm].add_batch(inp[0].detach()))(nm))
              for nm, m in subset.items()]
        for j in range(n):
            layer(inps[j].unsqueeze(0), **kw)
        for h in hs:
            h.remove()

        for nm, m in subset.items():
            r = gq[nm].quantize(cb, group=args.group, act_order=args.act_order,
                                rescale_in_loop=not args.no_rescale_in_loop,
                                pot_scale=args.pot_scale)
            m.weight.data = r['W_hat'].to(m.weight.dtype)
            if li == 0:
                ok, diag = verify_codebook_artifact(r['W_hat'], r['scales'], cb,
                                                    group=r['group'])
                stats.append(('verify', f'layer0.{nm}', bool(ok), diag.get('n_violations', 0)))
            gq[nm].free()

        for j in range(n):
            outs[j] = layer(inps[j].unsqueeze(0), **kw)[0]
        layers[li] = layer.cpu()
        del layer, gq, subset
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(f'  layer {li:2d}/{len(layers)}  {time.time()-t0:.1f}s', flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--codebook', default='strict_pot')
    ap.add_argument('--group', type=int, default=128)
    ap.add_argument('--nsamples', type=int, default=64)
    ap.add_argument('--seqlen', type=int, default=2048)
    ap.add_argument('--act-order', action='store_true')
    ap.add_argument('--no-rescale-in-loop', action='store_true')
    ap.add_argument('--pot-scale', action='store_true')
    ap.add_argument('--rtn', action='store_true', help='no compensation (H=I) baseline')
    ap.add_argument('--tasks', default='ppl,arc_easy')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='results/pot/pot_runs.jsonl')
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    model.eval()
    cb = get_codebook(a.codebook, device=DEV)
    print(f'codebook {a.codebook}: {len(cb)} levels, shift_free={cb.is_shift_free}', flush=True)

    batches = get_calib(tok, a.nsamples, a.seqlen)
    t0 = time.time()
    stats = quantize_model(model, batches, cb, a)
    t_quant = time.time() - t0
    model.to(DEV)

    K = model.config.hidden_size
    wpl = K * K
    bpw = accounted_bpw(4.0, a.group, K, len(cb), wpl,
                        scale_bits=8 if a.pot_scale else 16, zero_bits=0)
    verified = all(s[2] for s in stats) and len(stats) > 0

    rec = dict(model=a.model, codebook=a.codebook, group=a.group, nsamples=a.nsamples,
               seqlen=a.seqlen, act_order=a.act_order,
               rescale_in_loop=not a.no_rescale_in_loop, pot_scale=a.pot_scale,
               tag=a.tag, bpw=bpw, verified=verified,
               n_verify_checks=len(stats), t_quant_s=round(t_quant, 1))

    tasks = a.tasks.split(',')
    if 'ppl' in tasks:
        rec['wikitext2_ppl'] = eval_wikitext_ppl(model, tok, seqlen=a.seqlen, device=DEV)
    if 'arc_easy' in tasks:
        rec.update({f'arc_easy_{k}': v for k, v in
                    eval_arc_easy(model, tok, device=DEV, limit=a.limit).items()})

    rec['bpw_ok'] = bpw['total'] <= 4.5
    rec['success'] = bool(verified and rec['bpw_ok'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'a') as f:
        f.write(json.dumps(rec, default=str) + '\n')
    print(json.dumps(rec, indent=2, default=str))
    if not rec['success']:
        print('!! RUN NOT SUCCESSFUL: verifier or bpw gate failed', file=sys.stderr)


main()
