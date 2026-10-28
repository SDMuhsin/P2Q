"""
The channel that tr(E E^T) misses.

mean_layer_rel_mse moved only +1.04% under rotation while ARC-E fell 33 pp, so the
damage is not in unweighted weight MSE.  What the model actually feels is the
ACTIVATION-WEIGHTED error  sum_j H_jj ||E[:,j]||^2,  H_jj = E[x_j^2].

Hypothesis (PoT-specific): a log grid's ABSOLUTE error is proportional to |w|.  In the
native basis, LLM weights on high-variance (outlier) input channels are systematically
SMALL, so a log grid automatically places small absolute error exactly where the
activations are large -- free outlier protection it gets for nothing.  A uniform grid's
error is ~Delta/2 independent of |w|, so it has no such alignment to lose.  A Hadamard
mixes channels and destroys the alignment; it should cost PoT far more than int4 on the
H-weighted metric even though the unweighted metric barely moves.

Reports per layer, for {none, hadamard} x {strict_pot, int4_sym}:
  relmse_I  = ||E||^2 / ||W||^2                         (what the runs log)
  relmse_H  = sum_j H_jj||E_j||^2 / sum_j H_jj||W_j||^2 (what the model feels)
  amp       = relmse_H / relmse_I   -- >1 means error sits on loud channels
  corr_mw   = corr( log H_jj , log ||W[:,j]||^2 ) -- the alignment itself
"""
import argparse, json, os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, "/workspace/CoBALT/src"); sys.path.insert(0, "/workspace/CoBALT")
from pot.rotation import rotate_qwen3

SUF = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def q_pot(Xg):
    s = Xg.abs().amax(1, keepdim=True).clamp_min(1e-30)
    Xn = Xg / s
    r = torch.log2(Xn.abs().clamp_min(1e-45)).neg().round().clamp(0, 7)
    return torch.sign(Xn) * (2.0 ** (-r)) * s


def q_int4(Xg):
    s = Xg.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return (Xg / s * 7).round().clamp(-7, 7) / 7 * s


def err(W, g, fn):
    sh = W.shape
    return (fn(W.reshape(-1, g)) - W.reshape(-1, g)).reshape(sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", default="none")
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/workspace/CoBALT/benchmarks")
    from benchmark_suite import get_calibration_data

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                                 torch_dtype=torch.bfloat16).to(dev)
    model.eval(); model.config.use_cache = False
    if a.rotate != "none":
        rotate_qwen3(model, mode=a.rotate, seed=0, dev=dev, verbose=False)

    calib = get_calibration_data(tok, a.nsamples, a.seqlen, "wikitext2")

    # accumulate diag(H) = sum x_j^2 on the INPUT of every target linear
    acc, hooks = {}, []
    def mk(name):
        def h(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            s = (x * x).sum(0)
            acc[name] = acc.get(name, torch.zeros_like(s)) + s
        return h
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and any(n.endswith(s) for s in SUF):
            hooks.append(m.register_forward_hook(mk(n)))
    with torch.no_grad():
        for i in range(calib.shape[0]):
            model(calib[i:i+1].to(dev))
    for h in hooks:
        h.remove()

    rows = []
    with torch.no_grad():
        for n, m in model.named_modules():
            if not (isinstance(m, torch.nn.Linear) and n in acc):
                continue
            W = m.weight.detach().float().cpu()
            Hd = acc[n].cpu().clamp_min(1e-30)
            wcol = (W * W).sum(0).clamp_min(1e-30)
            lc = torch.corrcoef(torch.stack([Hd.log(), wcol.log()]))[0, 1].item()
            row = {"layer": n, "suffix": next(s for s in SUF if n.endswith(s)),
                   "corr_logH_logW": lc}
            for kind, fn in (("pot", q_pot), ("int4", q_int4)):
                E = err(W, a.group, fn)
                ecol = (E * E).sum(0)
                row[f"relmse_I_{kind}"] = (ecol.sum() / wcol.sum()).item()
                row[f"relmse_H_{kind}"] = ((Hd * ecol).sum() / (Hd * wcol).sum()).item()
                row[f"amp_{kind}"] = row[f"relmse_H_{kind}"] / row[f"relmse_I_{kind}"]
            rows.append(row)
    json.dump({"rotate": a.rotate, "rows": rows}, open(a.out, "w"), indent=1)
    ks = ["relmse_I_pot", "relmse_H_pot", "amp_pot", "relmse_I_int4", "relmse_H_int4",
          "amp_int4", "corr_logH_logW"]
    print(a.rotate, {k: round(sum(r[k] for r in rows) / len(rows), 6) for k in ks})


main()
