"""
Step 2 of PoT-PTQ (arXiv:2507.11959): the *data-dependent* learned scale residual.

Where this sits
---------------
PoT-PTQ quantizes to `W = S . sign(W) . 2^E` in two steps:

  step 1  a data-agnostic per-group scale grid search (`scale_search.search_group_scale`,
          already implemented, and used inside the GPTQ sweep by `gptq_pot.PoTGPTQ`);
  step 2  THIS FILE -- a learned multiplicative residual on that scale,
              S_hat = S * (1 + Gamma),      Gamma initialised to 0,
          optimised by Adam to minimise the BLOCK OUTPUT reconstruction error
              || F(W, X) - F(W_tilde(Gamma), X) ||^2  +  (lam/2) ||Gamma||_F^2
          i.e. OmniQuant-style block reconstruction restricted to the scales.

We were missing step 2 entirely. This file adds it, and (unlike PoT-PTQ, which has no
error compensation at all) stacks it ON TOP of the GPTQ output rather than on RTN.

The invariant, and why learning a scale cannot break it
------------------------------------------------------
ONLY the scale is learned. Never the codes, never the weights. The stored artifact is
rebuilt, every single step and again at the end, as

    W_hat = levels[code] * S_hat            with `levels` the frozen strict-PoT codebook

so every weight is a codeword times a scale by *construction*, exactly as before -- the
learning moves the scale the lattice is anchored to, not the lattice. The one thing that
does change is which codeword each weight is assigned to, and that is a discrete
re-assignment recomputed under `torch.no_grad()` each step (straight-through: the
gradient sees only `S_hat`, never the assignment). `verify_strict_pot` from
`verify_shiftfree.py` is called on the finished artifact, and is the only thing that
licenses the result -- see `learn_scale_residual(..., verify=True)`, which is the default.

Which target the codes are re-assigned against -- and a measured surprise
------------------------------------------------------------------------
PoT-PTQ recomputes `E = clamp(round(log2(|W|/S_hat)))` at every step, because their step 1
artifact is plain RTN: the only thing tying a code to a weight is the current scale, so the
assignment must follow the scale. Stacking on GPTQ breaks that reasoning, and MEASURABLY.

  retarget='none' (DEFAULT) -- freeze the codes GPTQ chose; learn a pure scale refinement.
  retarget='gptq' -- re-round the already-compensated column values GPTQ actually saw
        (`out['W_tgt']`, from `quantize(return_targets=True)`).
  retarget='orig' -- re-round the original W (what PoT-PTQ does; correct for an RTN
        artifact, and the right setting with --method rtn).

Measured, Qwen3-1.7B layer 0, strict PoT g128, layer scope, block relative output error:

    GPTQ baseline                      0.015453
    + learned scale, retarget='gptq'   0.031547   (2.04x WORSE)

and the same ordering on a synthetic fixture at every learning rate tried (1e-4..1e-2),
monotonically worse the further Gamma moves. The reason is structural, not a tuning
failure. GPTQ's rounding decisions and its error feedback are one object: column j's
compensated value `W_tgt[:, j]` was produced by the specific codes chosen for columns
< j. Re-round column j against `W_tgt` under a new scale and the code changes while the
feedback that justified it does not, so the residual the sweep already pushed forward is
now accounting for a rounding decision that no longer exists. The artifact drifts toward
an uncompensated (RTN-quality) solution -- which is exactly what 'orig' converges to from
above (0.0316 there too). Freezing the codes keeps GPTQ's solution self-consistent and
lets Gamma do the only thing it can still do honestly: re-centre the reconstruction level
of an assignment that is already fixed.

The principled way to get both would be to ALTERNATE -- learn Gamma, then re-run the GPTQ
sweep with the learned scale (`rescale_in_loop=False`, scales := S*(1+Gamma)) so the
feedback is recomputed for the new codes. That is not implemented here.

Scope
-----
  scope='layer' -- per-linear reconstruction. `|| (W - W_hat) X ||_F^2 = tr(E H E^T)` with
        H the Gram matrix GPTQ already accumulated, so this needs NO activations and no
        forward pass at all: it is exact, and costs one [in, in] matmul per step.
  scope='block' -- whole-decoder-block reconstruction against the block's fp16/bf16 output.
        This is what PoT-PTQ (and OmniQuant) actually do; it sees the nonlinearities, the
        norms and the residual, so it can trade error between the linears of a block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebooks import Codebook, nearest_codeword
from .verify_shiftfree import verify_codebook_artifact, verify_strict_pot

__all__ = ["learn_scale_residual", "ScaledPoTLinear", "rebuild_w_hat"]


def _levels(C):
    return C.levels if isinstance(C, Codebook) else C


# --------------------------------------------------------------------------- module
class ScaledPoTLinear(nn.Module):
    """A drop-in stand-in for one quantized nn.Linear whose SCALE is a leaf parameter.

    Holds the frozen codebook `levels`, the frozen target `W_tgt` (or the frozen codes),
    the frozen base scale `S0` [out, n_groups], the frozen `g_idx` [in] and `col_scale`,
    and the single learnable tensor `gamma` (same shape as `S0`), plus an optional
    per-input-column `gamma_col` [in].

    `forward` rebuilds the weight from those parameters exactly as an inference kernel
    would, so what the loss sees is literally the artifact.
    """

    def __init__(self, out: dict, levels: torch.Tensor, bias=None,
                 retarget: str = "none", learn_col: bool = False,
                 clamp: float = 0.75, device=None):
        super().__init__()
        dev = device or out["W_hat"].device
        self.register_buffer("levels", levels.to(dev, torch.float32), persistent=False)
        self.register_buffer("S0", out["scales"].to(dev, torch.float32).clone(),
                             persistent=False)
        self.register_buffer("g_idx", out["g_idx"].to(dev).long(), persistent=False)
        cs = out.get("col_scale")
        self.register_buffer("col_scale",
                             None if cs is None else cs.to(dev, torch.float32).clone(),
                             persistent=False)
        self.retarget = retarget
        self.clamp = float(clamp)

        if retarget == "none":
            self.register_buffer("codes", out["codes"].to(dev).long(), persistent=False)
            self.target = None
        else:
            key = "W_tgt" if retarget == "gptq" else "W_orig"
            if key not in out or out[key] is None:
                raise KeyError(
                    f"retarget={retarget!r} needs out[{key!r}]; run "
                    f"PoTGPTQ.quantize(..., return_targets=True) or pass W_orig")
            self.register_buffer("target", out[key].to(dev, torch.float32).clone(),
                                 persistent=False)
            self.codes = None

        n_in = out["W_hat"].shape[1]
        self.gamma = nn.Parameter(torch.zeros_like(self.S0))
        self.gamma_col = (nn.Parameter(torch.zeros(n_in, device=dev))
                          if learn_col else None)
        self.bias = bias

    # ------------------------------------------------------------------ the artifact
    def scales(self) -> torch.Tensor:
        """S_hat = S0 * (1 + Gamma), clamped away from 0 so it stays a legal scale."""
        g = self.gamma.clamp(-self.clamp, self.clamp)
        return self.S0 * (1.0 + g)

    def eff_col_scale(self):
        """The per-input-column factor actually stored, including any learned part."""
        if self.gamma_col is None:
            return self.col_scale
        c = 1.0 + self.gamma_col.clamp(-self.clamp, self.clamp)
        return c if self.col_scale is None else self.col_scale * c

    def full_scale(self) -> torch.Tensor:
        """[out, in] effective scale, built in the SAME association order as
        `verify_shiftfree.expand_scales` (row-group first, column factor outermost)."""
        S = self.scales()[:, self.g_idx]
        c = self.eff_col_scale()
        if c is not None:
            S = S * c.view(1, -1)
        return S

    def weight_hat(self) -> torch.Tensor:
        S = self.full_scale()
        if self.retarget == "none":
            idx = self.codes
        else:
            with torch.no_grad():                       # straight-through assignment
                idx = nearest_codeword(self.target / S.detach(), self.levels)
            self._last_idx = idx
        return self.levels[idx] * S

    def forward(self, x):
        return F.linear(x, self.weight_hat().to(x.dtype), self.bias)

    # ------------------------------------------------------------------ finalisation
    @torch.no_grad()
    def finalize(self) -> dict:
        """Freeze the learned scale into a storable artifact.

        Returns a dict with the same keys `PoTGPTQ.quantize` returns, so it drops straight
        back into the driver and into the verifier. `W_hat` is recomputed from the STORED
        `codes` and `scales` -- not carried over from the graph -- so the thing verified is
        the thing an inference kernel would reconstruct.
        """
        S_grp = self.scales().detach().float()
        c = self.eff_col_scale()
        c = None if c is None else c.detach().float()
        S = S_grp[:, self.g_idx]
        if c is not None:
            S = S * c.view(1, -1)
        if self.retarget == "none":
            idx = self.codes
        else:
            idx = nearest_codeword(self.target / S, self.levels)
        W_hat = self.levels[idx] * S
        return {"W_hat": W_hat, "codes": idx.to(torch.uint8), "scales": S_grp,
                "g_idx": self.g_idx, "col_scale": c}


@torch.no_grad()
def rebuild_w_hat(out: dict, levels: torch.Tensor) -> torch.Tensor:
    """`levels[codes] * expand(scales)` -- the reference dequantization."""
    S = out["scales"][:, out["g_idx"].long()]
    if out.get("col_scale") is not None:
        S = S * out["col_scale"].view(1, -1)
    return levels[out["codes"].long()] * S


# ------------------------------------------------------------------------- helpers
def _get_parent(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


def _relative(num, den):
    if torch.is_tensor(num):
        num = float(num.detach())
    if torch.is_tensor(den):
        den = float(den.detach())
    return float(num) / float(max(float(den), 1e-30))


# --------------------------------------------------------------------- layer scope
def _layer_loss(shim: ScaledPoTLinear, W_orig: torch.Tensor, H: torch.Tensor):
    """|| (W - W_hat) X ||_F^2 = tr(E H E^T), with H = sum_t X_t^T X_t (up to a constant).

    Exact output reconstruction error for one linear, with no forward pass.
    """
    E = shim.weight_hat() - W_orig
    return torch.einsum("oi,ij,oj->", E, H, E)


# --------------------------------------------------------------------------- driver
def learn_scale_residual(layer_or_block,
                         X=None,
                         quant_state=None,
                         *,
                         codebook=None,
                         scope: str = "block",
                         Y_ref=None,
                         forward_kwargs: dict | None = None,
                         iters: int = 100,
                         lr: float = 5e-3,
                         lam: float = 0.0,
                         batch: int = 1,
                         retarget: str = "none",
                         learn_col: bool = False,
                         clamp: float = 0.75,
                         device: str = "cuda",
                         dtype=torch.float32,
                         verify: bool = True,
                         seed: int = 0,
                         verbose: bool = False):
    """Learn `Gamma` in `S_hat = S * (1 + Gamma)` by block-output reconstruction.

    Args:
        layer_or_block: the nn.Module to reconstruct. A whole decoder layer when
            `scope='block'`; a decoder layer or a single nn.Linear when `scope='layer'`.
        X:  calibration inputs. `scope='block'`: [N, T, hidden] (CPU is fine, minibatched
            onto `device`). `scope='layer'`: unused -- the Gram matrix in
            `quant_state[name]['H']` carries all the information, exactly.
        quant_state: `{linear_name: out_dict}` as returned by `PoTGPTQ.quantize`, each
            augmented with 'W_orig' (the unquantized fp32 weight) and, for scope='layer',
            'H' (the accumulated Gram). Pass a single out_dict if `layer_or_block` is one
            nn.Linear. Modified in place with the refined 'W_hat'/'codes'/'scales'.
        codebook: the Codebook (needed for the levels and for verification).
        scope: 'block' (whole-decoder-block reconstruction, PoT-PTQ's choice) or 'layer'
            (per-linear, exact, activation-free).
        Y_ref: reference block outputs [N, T, hidden] computed with the ORIGINAL weights.
            Required for scope='block'. (In the sequential driver this is precisely the
            `outs` tensor computed before quantizing the block.)
        forward_kwargs: extra kwargs the decoder layer needs (position_embeddings, masks).
        iters, lr, lam: Adam steps, learning rate, and the (lam/2)||Gamma||_F^2 penalty.
        batch: sequences per step (scope='block').
        retarget: 'none' (default; keep GPTQ's codes) | 'gptq' | 'orig'. See the module
            docstring -- re-rounding on top of a GPTQ artifact measurably hurts.
        learn_col: also learn a per-input-column residual (costs 16 bits / in-feature).
        clamp: |Gamma| is clamped to this, so S_hat stays strictly positive and bounded.
        verify: run `verify_shiftfree.verify_strict_pot` (or the add2 verifier) on every
            refined artifact and RAISE if it does not pass. Leave this on.

    Returns:
        dict with per-linear before/after reconstruction error, the loss trace, and the
        verification report.
    """
    assert scope in ("block", "layer"), scope
    assert retarget in ("gptq", "orig", "none"), retarget
    torch.manual_seed(seed)
    dev = torch.device(device)

    single = isinstance(layer_or_block, nn.Linear)
    if single:
        assert isinstance(quant_state, dict) and "W_hat" in quant_state
        states = {"": quant_state}
    else:
        states = quant_state
    assert states, "quant_state is empty"

    if codebook is None:
        raise ValueError("pass the Codebook used to quantize (needed for verification)")
    levels = _levels(codebook).to(dev, torch.float32)

    # ------------------------------------------------------------------ build shims
    shims = {}
    originals = {}
    for name, out in states.items():
        if single:
            lin = layer_or_block
        else:
            parent, attr = _get_parent(layer_or_block, name)
            lin = getattr(parent, attr)
        bias = None if lin.bias is None else lin.bias.data.to(dev, dtype)
        shims[name] = ScaledPoTLinear(out, levels, bias=bias, retarget=retarget,
                                      learn_col=learn_col, clamp=clamp,
                                      device=dev).to(dev)
        originals[name] = lin

    params = []
    for s in shims.values():
        params.append(s.gamma)
        if s.gamma_col is not None:
            params.append(s.gamma_col)
    opt = torch.optim.Adam(params, lr=lr)

    def gamma_penalty():
        if lam == 0.0:
            return torch.zeros((), device=dev)
        t = sum((s.gamma ** 2).sum() for s in shims.values())
        if learn_col:
            t = t + sum((s.gamma_col ** 2).sum() for s in shims.values())
        return 0.5 * lam * t

    trace = []
    info = {"scope": scope, "retarget": retarget, "iters": iters, "lr": lr, "lam": lam,
            "learn_col": learn_col, "linears": {}}

    # =============================================================== layer scope
    if scope == "layer":
        Hs, Ws, denom = {}, {}, {}
        for name, out in states.items():
            H = out.get("H")
            if H is None:
                raise KeyError(f"scope='layer' needs quant_state[{name!r}]['H']")
            Hs[name] = H.to(dev, torch.float32)
            Ws[name] = out["W_orig"].to(dev, torch.float32)
            denom[name] = float(torch.einsum("oi,ij,oj->", Ws[name], Hs[name], Ws[name]))
            info["linears"][name] = {
                "err_before": _relative(_layer_loss(shims[name], Ws[name], Hs[name]),
                                        denom[name])}
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            loss = sum(_layer_loss(shims[n], Ws[n], Hs[n]) / denom[n] for n in shims)
            total = loss + gamma_penalty()
            total.backward()
            opt.step()
            trace.append(float(loss))
            if verbose and (it % max(1, iters // 10) == 0 or it == iters - 1):
                print(f"    [scale_learn/layer] it {it:4d}  rel_out_err {float(loss):.6g}",
                      flush=True)
        with torch.no_grad():
            for name in shims:
                info["linears"][name]["err_after"] = _relative(
                    _layer_loss(shims[name], Ws[name], Hs[name]), denom[name])

    # =============================================================== block scope
    else:
        assert not single, "scope='block' expects a decoder block, not a single Linear"
        assert X is not None and Y_ref is not None, "scope='block' needs X and Y_ref"
        fk = dict(forward_kwargs or {})
        fk = {k: (v.to(dev, dtype) if torch.is_tensor(v) and v.is_floating_point()
                  else (tuple(t.to(dev, dtype) for t in v)
                        if isinstance(v, tuple) and all(torch.is_tensor(t) for t in v)
                        else (v.to(dev) if torch.is_tensor(v) else v)))
              for k, v in fk.items()}

        block = layer_or_block
        orig_dtype = next(block.parameters()).dtype
        orig_req = [(pp, pp.requires_grad) for pp in block.parameters()]
        block.to(dev, dtype)
        for pp in block.parameters():
            pp.requires_grad_(False)
        for name, sh in shims.items():
            parent, attr = _get_parent(block, name)
            setattr(parent, attr, sh)

        N = X.shape[0]
        order = torch.randperm(N)
        ptr = 0

        def next_batch():
            nonlocal ptr, order
            if ptr + batch > N:
                order = torch.randperm(N)
                ptr = 0
            sl = order[ptr:ptr + batch]
            ptr += batch
            return X[sl].to(dev, dtype), Y_ref[sl].to(dev, dtype)

        # before / after are measured on a FIXED held-out-ish slice for comparability
        n_ev = min(N, max(4, batch))
        Xe = X[:n_ev].to(dev, dtype)
        Ye = Y_ref[:n_ev].to(dev, dtype)

        @torch.no_grad()
        def eval_err():
            num = den = 0.0
            for j in range(n_ev):
                y = block(Xe[j:j + 1], **fk)
                y = y[0] if isinstance(y, tuple) else y
                num += float(((y - Ye[j:j + 1]) ** 2).sum())
                den += float((Ye[j:j + 1] ** 2).sum())
            return _relative(num, den)

        info["block_err_before"] = eval_err()
        for it in range(iters):
            xb, yb = next_batch()
            opt.zero_grad(set_to_none=True)
            y = block(xb, **fk)
            y = y[0] if isinstance(y, tuple) else y
            loss = ((y - yb) ** 2).sum() / (yb ** 2).sum().clamp_min(1e-30)
            (loss + gamma_penalty()).backward()
            opt.step()
            trace.append(float(loss))
            if verbose and (it % max(1, iters // 10) == 0 or it == iters - 1):
                print(f"    [scale_learn/block] it {it:4d}  rel_out_err {float(loss):.6g}",
                      flush=True)
        info["block_err_after"] = eval_err()

        # restore the real Linears (weights are written below) and the block's original
        # dtype -- the sequential driver keeps forwarding through this layer in bf16.
        for name, lin in originals.items():
            parent, attr = _get_parent(block, name)
            setattr(parent, attr, lin)
        block.to(orig_dtype)
        for pp, rq in orig_req:
            pp.requires_grad_(rq)

    # ------------------------------------------------------- finalize + VERIFY
    reports = []
    for name, sh in shims.items():
        new = sh.finalize()
        out = states[name]
        # the artifact must be exactly codes * scales; rebuild it independently and
        # require bit equality before we let it anywhere near the model.
        ref = rebuild_w_hat({**new, "codes": new["codes"]}, levels)
        assert torch.equal(ref, new["W_hat"]), (
            f"{name}: dequant from stored params disagrees with the learned artifact")

        gam = sh.gamma.detach().clamp(-clamp, clamp)
        info["linears"].setdefault(name, {})
        info["linears"][name].update({
            "gamma_absmean": float(gam.abs().mean()),
            "gamma_absmax": float(gam.abs().max()),
            "code_change_frac": float((new["codes"] != out["codes"]).float().mean()),
        })
        if "W_orig" in out:
            Wo = out["W_orig"].to(new["W_hat"].device, torch.float32)
            info["linears"][name]["rel_mse_before"] = out.get("rel_mse")
            info["linears"][name]["rel_mse_after"] = _relative(
                ((new["W_hat"] - Wo) ** 2).sum(), (Wo ** 2).sum())

        if verify:
            vkw = dict(group=out.get("group"), g_idx=new["g_idx"],
                       col_scale=new["col_scale"])
            if getattr(codebook, "shift_free", "none") == "strict_pot":
                # the immutable condition, checked by the existing verifier -- no
                # tolerance, bit-equality against an exhibited +-2^e witness.
                ok, diag = verify_strict_pot(new["W_hat"], new["scales"], **vkw)
            else:
                ok, diag = verify_codebook_artifact(
                    new["W_hat"], new["scales"], codebook, **vkw)
            reports.append({"linear": name, "ok": ok, "kind": diag.get("kind"),
                            "n_violations": diag.get("n_violations")})
            if ok is False:
                raise AssertionError(
                    f"learn_scale_residual BROKE THE POWER-OF-TWO INVARIANT on {name}: "
                    f"{diag.get('n_violations')} violations, e.g. {diag['examples'][:2]}")

        out.update({k: new[k] for k in ("W_hat", "codes", "scales", "g_idx", "col_scale")})
        out["gamma"] = gam.cpu()
        # write the refined weight back into the live module
        lin = originals[name]
        lin.weight.data = new["W_hat"].to(lin.weight.dtype).to(lin.weight.device)

    info["verify"] = reports
    info["verify_ok"] = (all(r["ok"] is True for r in reports) if reports else None)
    info["loss_trace"] = trace
    info["loss_first"] = trace[0] if trace else None
    info["loss_last"] = trace[-1] if trace else None
    return info
