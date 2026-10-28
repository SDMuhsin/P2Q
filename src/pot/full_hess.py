"""Full tr(E H E^T)/tr(W H W^T) with the TRUE Hessian H = X^T X (off-diagonals kept),
for {none,hadamard} x {strict_pot,int4_sym}.  diag(H) alone may understate how much
rotation costs, because rotation makes E dense and cross-channel correlated."""
import argparse, json, sys, torch
sys.path.insert(0, "/workspace/CoBALT/src"); sys.path.insert(0, "/workspace/CoBALT")
sys.path.insert(0, "/workspace/CoBALT/benchmarks")
from pot.rotation import rotate_qwen3
from benchmark_suite import get_calibration_data
SUF = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
       "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
def q_pot(Xg):
    s=Xg.abs().amax(1,keepdim=True).clamp_min(1e-30); Xn=Xg/s
    r=torch.log2(Xn.abs().clamp_min(1e-45)).neg().round().clamp(0,7)
    return torch.sign(Xn)*(2.0**(-r))*s
def q_int4(Xg):
    s=Xg.abs().amax(1,keepdim=True).clamp_min(1e-30)
    return (Xg/s*7).round().clamp(-7,7)/7*s
ap=argparse.ArgumentParser(); ap.add_argument("--rotate",default="none")
ap.add_argument("--nsamples",type=int,default=8); ap.add_argument("--group",type=int,default=64)
ap.add_argument("--layers",default="0,9,18,27"); ap.add_argument("--out",required=True)
a=ap.parse_args(); want={int(x) for x in a.layers.split(",")}
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B",use_fast=True)
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",torch_dtype=torch.bfloat16).to("cuda")
model.eval(); model.config.use_cache=False
if a.rotate!="none": rotate_qwen3(model,mode=a.rotate,seed=0,dev="cuda",verbose=False)
calib=get_calibration_data(tok,a.nsamples,2048,"wikitext2")
Hs={}; hooks=[]
def mk(n):
    def h(m,i,o):
        x=i[0].detach().reshape(-1,i[0].shape[-1]).float()
        Hs[n]=Hs.get(n,torch.zeros(x.shape[1],x.shape[1],device=x.device))+x.T@x
    return h
for n,m in model.named_modules():
    if isinstance(m,torch.nn.Linear) and any(n.endswith(s) for s in SUF):
        if int(n.split("layers.")[1].split(".")[0]) in want: hooks.append(m.register_forward_hook(mk(n)))
with torch.no_grad():
    for i in range(calib.shape[0]): model(calib[i:i+1].to("cuda"))
for h in hooks: h.remove()
rows=[]
with torch.no_grad():
    for n,m in model.named_modules():
        if n not in Hs: continue
        W=m.weight.detach().float(); H=Hs[n]
        den=(W@H*W).sum().item(); row={"layer":n,"suffix":next(s for s in SUF if n.endswith(s))}
        for k,fn in (("pot",q_pot),("int4",q_int4)):
            E=(fn(W.reshape(-1,a.group))-W.reshape(-1,a.group)).reshape(W.shape)
            row[f"trEHE_{k}"]=(E@H*E).sum().item()/den
            row[f"trEE_{k}"]=(E*E).sum().item()/(W*W).sum().item()
        rows.append(row)
json.dump({"rotate":a.rotate,"rows":rows},open(a.out,"w"),indent=1)
ks=["trEE_pot","trEHE_pot","trEE_int4","trEHE_int4"]
print(a.rotate,{k:round(sum(r[k] for r in rows)/len(rows),6) for k in ks})
