"""CONTROL: rotated but UNQUANTIZED model, evaluated through the SAME eval path
run_pot.py uses (pot.eval_arc.eval_arc_easy + pot.eval_ppl).  bf16 reference:
wikitext2 ppl 16.7185, arc_easy acc 0.7264, acc_norm 0.7050.
If ARC collapses here while ppl is fine, the ARC eval path is broken under rotation."""
import argparse, sys, torch
sys.path.insert(0,"/workspace/CoBALT/src"); sys.path.insert(0,"/workspace/CoBALT")
from pot.rotation import rotate_qwen3
from pot.eval_arc import eval_arc_easy
from pot.eval_ppl import eval_wikitext_ppl, get_wikitext2_testenc
from transformers import AutoModelForCausalLM, AutoTokenizer
ap=argparse.ArgumentParser(); ap.add_argument("--rotate",default="hadamard"); a=ap.parse_args()
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B",use_fast=True)
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",torch_dtype=torch.bfloat16).to("cuda")
m.eval(); m.config.use_cache=False
if a.rotate!="none": rotate_qwen3(m,mode=a.rotate,seed=0,dev="cuda",verbose=False)
ppl=eval_wikitext_ppl(m,tok,seqlen=2048,device="cuda")
arc=eval_arc_easy(m,tok,device="cuda")
print(f"RESULT rotate={a.rotate} UNQUANTIZED ppl={float(ppl):.4f} acc={arc['acc']:.4f} acc_norm={arc['acc_norm']:.4f}")
