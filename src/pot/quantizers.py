"""
Minimal weight-only quantizers for a single [out_features, in_features] matrix.

All quantizers here are *group-wise along the INPUT dimension*, i.e. the weight is
reshaped to [out * (in // group), group] and each row of that view gets its own
scale (and zero point). This is the layout every 4-bit inference kernel expects
(GPTQ / AWQ / bitsandbytes NF4 / SINQ with axis=1), because at GEMM time a group of
`group` contiguous input channels shares one scale.

Each function returns `(W_hat, meta)`:
    W_hat : dequantized weight, SAME shape and SAME dtype as the input W.
    meta  : dict with the quantization parameters and a couple of error metrics.

Conventions
-----------
asymmetric:  q = clamp(round(W / s) + z, 0, 2^b - 1);  W_hat = (q - z) * s
             s = (max - min) / (2^b - 1),  z = round(-min / s)
symmetric:   q = clamp(round(W / s), -2^(b-1), 2^(b-1) - 1);  W_hat = q * s
             s = max|W| / (2^(b-1) - 1)

Effective bits per weight (ignoring packing overhead) is
    bits + 2 * scale_bits / group        (asym, scale + zero stored in fp16)
    bits + 1 * scale_bits / group        (sym)
`meta['effective_bits']` reports this assuming fp16 params.

`group <= 0` or `group >= in_features` means per-output-channel (one group per row).
"""

from __future__ import annotations

import torch

__all__ = ["rtn_int_asym", "rtn_int_sym", "quantize_weight", "QUANTIZERS"]


def _reshape_groups(W: torch.Tensor, group: int):
    """[out, in] -> ([out * n_groups, group], group)."""
    assert W.dim() == 2, f"expected a 2-D [out, in] weight, got {tuple(W.shape)}"
    out_f, in_f = W.shape
    if group is None or group <= 0 or group > in_f:
        group = in_f
    assert in_f % group == 0, (
        f"in_features={in_f} not divisible by group={group}; "
        "pick a group size that divides the input dim"
    )
    return W.reshape(-1, group), group


def _finish(W: torch.Tensor, W_hat: torch.Tensor, meta: dict):
    W_hat = W_hat.reshape(W.shape).to(W.dtype)
    err = (W_hat.float() - W.float())
    denom = W.float().norm().clamp_min(1e-12)
    meta["mse"] = err.pow(2).mean().item()
    meta["rel_fro_err"] = (err.norm() / denom).item()
    return W_hat, meta


def _effective_bits(bits: int, group: int, n_params: int, scale_bits: int = 16):
    return bits + n_params * scale_bits / group


@torch.no_grad()
def rtn_int_asym(W: torch.Tensor, bits: int = 4, group: int = 128, scale_bits: int = 16):
    """
    Standard asymmetric integer round-to-nearest. The "what everyone ships" baseline
    (bitsandbytes int4, GPTQ's RTN reference, HQQ's init, ...).

    min/max of each group are mapped to 0 / 2^bits - 1, so the group's dynamic range is
    used exactly. Costs one scale + one zero point per group.
    """
    if bits >= 16:
        return W.clone(), {"method": "rtn_int_asym", "bits": bits, "group": group,
                           "effective_bits": 16.0, "mse": 0.0, "rel_fro_err": 0.0}
    Wg, group = _reshape_groups(W, group)
    Wg = Wg.float()
    qmax = 2 ** bits - 1

    wmin = Wg.min(dim=1, keepdim=True).values.clamp_max(0)
    wmax = Wg.max(dim=1, keepdim=True).values.clamp_min(0)
    scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
    zero = torch.round(-wmin / scale)

    q = torch.clamp(torch.round(Wg / scale) + zero, 0, qmax)
    W_hat = (q - zero) * scale

    meta = {
        "method": "rtn_int_asym", "bits": bits, "group": group,
        "effective_bits": _effective_bits(bits, group, 2, scale_bits),
        "scale": scale, "zero": zero,
    }
    return _finish(W, W_hat, meta)


@torch.no_grad()
def rtn_int_sym(W: torch.Tensor, bits: int = 4, group: int = 128, scale_bits: int = 16):
    """
    Symmetric integer round-to-nearest: no zero point, grid centred on 0.

    Cheaper metadata (one fp scale per group) and cheaper kernels (no zero-point
    subtraction), but wastes ~1 code of range whenever a group is not centred on 0,
    which for LLM weights costs a small but real amount of accuracy vs asymmetric.
    Uses the signed range [-2^(b-1), 2^(b-1)-1] with scale set from the positive side.
    """
    if bits >= 16:
        return W.clone(), {"method": "rtn_int_sym", "bits": bits, "group": group,
                           "effective_bits": 16.0, "mse": 0.0, "rel_fro_err": 0.0}
    Wg, group = _reshape_groups(W, group)
    Wg = Wg.float()
    qpos = 2 ** (bits - 1) - 1
    qneg = -(2 ** (bits - 1))

    absmax = Wg.abs().max(dim=1, keepdim=True).values
    scale = (absmax / qpos).clamp_min(1e-8)

    q = torch.clamp(torch.round(Wg / scale), qneg, qpos)
    W_hat = q * scale

    meta = {
        "method": "rtn_int_sym", "bits": bits, "group": group,
        "effective_bits": _effective_bits(bits, group, 1, scale_bits),
        "scale": scale,
    }
    return _finish(W, W_hat, meta)


QUANTIZERS = {
    "rtn_int_asym": rtn_int_asym,
    "rtn_int_sym": rtn_int_sym,
}


def quantize_weight(W: torch.Tensor, method: str, bits: int = 4, group: int = 128):
    """Dispatch by name. `method='none'` is a pass-through."""
    if method in (None, "none"):
        return W, {"method": "none", "bits": 16, "group": group,
                   "effective_bits": 16.0, "mse": 0.0, "rel_fro_err": 0.0}
    if method not in QUANTIZERS:
        raise ValueError(f"unknown method {method!r}; have {sorted(QUANTIZERS)}")
    return QUANTIZERS[method](W, bits=bits, group=group)
