#!/usr/bin/env python3
"""
Incoherence processing by orthogonal rotation (QuaRot / SpinQuant style) for Qwen3.

Why this preserves the immutable P2Q condition
----------------------------------------------
The identity used everywhere below is  ``W x = (W Q) (Q^T x)`` for orthogonal ``Q``.
We quantize ``W Q``, not ``W``.  Nothing about the *codebook* changes: the stored
weights are still exactly ``code * scale`` with ``code`` drawn from the strict
power-of-two set ``{+-2^e}``, so the shift-free verifier in
``pot/verify_shiftfree.py`` still passes bit-for-bit on the artifact.

The counterpart ``Q^T x`` on the activation side is *not* a per-weight multiply:

  * Every rotation fused in this file is fused **into the weights of an adjacent
    linear layer at quantization time** -- it costs nothing at all at inference.
  * The single rotation that cannot be fused (the ``down_proj`` input, which sits
    behind the SwiGLU nonlinearity) is a **Hadamard transform**: the fast
    Walsh-Hadamard transform is `n log n` additions and subtractions plus ONE
    scalar multiply by ``1/sqrt(n)`` for the whole vector, which can itself be a
    shift when ``n`` is an even power of two.  There is no per-weight multiply,
    and no multiply at all inside the matmul.  The randomized variant multiplies
    by a ``+-1`` diagonal, which is a sign flip.
  * Rotation therefore leaves the "weight x activation is an exponent-add" claim
    intact.  It only changes *which* matrix we hand to the quantizer.

Qwen3-specific hazards, and how each is handled
-----------------------------------------------
1. ``lm_head`` is TIED to ``embed_tokens`` (same ``data_ptr``).  The residual-stream
   rotation acts on both as ``W <- W Q`` -- the *same* operation -- so tying alone is
   not a problem.  What IS a problem is the final RMSNorm gain: to rotate the residual
   stream you must fold ``model.norm``'s gain into ``lm_head``, and that is not an
   operation ``embed_tokens`` wants.  Untying to allow it costs
   ``151936 x 2048 x 2 B = 0.622 GB`` in bf16, against a 0.749 GB quantized decoder at
   4.25 bpw -- i.e. **+83% on the entire weight budget**, which would make any "4.1 bpw"
   headline dishonest.
   **So by default we do NOT untie.**  ``embed_tokens``/``lm_head`` are left completely
   untouched (still tied, still sharing storage) and R1 is applied ONLINE at the two
   points where the residual stream is entered and left:
       * after the embedding lookup:  ``h <- h Q1``
       * before ``lm_head``:          ``x <- (x Q1^T) * g_final``
   with ``model.norm.weight`` set to ones and its gain ``g_final`` carried into that
   second hook.  That is two Hadamards (adds/subtracts) and one length-2048 vector
   scale per token, negligible beside the 151936-row ``lm_head`` matmul itself, and it
   is mathematically identical to the fused-and-untied version.  Neither
   ``embed_tokens`` nor ``lm_head`` is quantized, so nothing is lost by not fusing
   into them.
   ``untie_lm_head=True`` restores the fused variant as an explicitly costed option;
   it must never be the default or the headline.  ``untied_lm_head`` in the returned
   info dict says which was used.
2. ``q_norm`` / ``k_norm`` are per-head RMSNorms of size ``head_dim`` applied AFTER
   ``q_proj``/``k_proj`` and BEFORE RoPE.  A rotation folded out of the *output* of
   ``q_proj``/``k_proj`` does **not** commute through them (RMSNorm is not linear,
   and its learned per-channel gain is not rotation-equivariant).  We therefore
   **never rotate the output side of q_proj/k_proj**.  Only their *input* side is
   rotated, which is upstream of the norms and provably safe.  This is why there is
   no "R3" (KV-cache) rotation here.
3. ``config.head_dim`` (128) is read from the config, never computed as
   ``hidden_size // num_heads``.
4. GQA 16:8 -- ``k_proj``/``v_proj`` are half width.  The per-head value rotation
   uses the SAME 128x128 Hadamard for every head, so the query-head -> kv-head
   mapping introduced by ``repeat_kv`` is irrelevant to correctness.

Which rotations are applied
---------------------------
R1  residual stream, size ``hidden_size`` (2048).  Fused into every
    ``q/k/v/gate/up_proj`` (input) and every ``o_proj``/``down_proj`` (output); applied
    ONLINE at the embedding output and the ``lm_head`` input (see hazard 1 above),
    unless ``untie_lm_head=True``, in which case it is fused there too.  Requires the
    two per-layer RMSNorm gains
    (``input_layernorm``, ``post_attention_layernorm``, ``model.norm``) to be folded
    into the following linears first, because ``diag(g) Q != Q diag(g)``.
    RMSNorm's normalization itself commutes with an orthogonal map (it preserves
    ``||x||``), so folding the gain is the only thing needed.
R2  per-head value/output rotation, size ``head_dim`` (128).  Fused into ``v_proj``
    (output, per head) and ``o_proj`` (input, per head).  Attention is linear in V,
    so this commutes with the attention weights exactly.  Fully fused, no runtime op.
R4  ``down_proj`` input, size ``intermediate_size`` (6144).  This one sits behind
    SwiGLU and CANNOT be fused; it is applied ONLINE as a fast Hadamard transform
    (implemented here as a matmul for validation clarity; see module docstring for
    the multiply-free argument).

Hadamard construction
---------------------
``hadamard_matrix(n)`` builds an exact (orthonormal) Hadamard matrix:
  * ``n`` a power of two -> Sylvester construction.
  * otherwise -> ``n = m * 2^k`` with ``m`` a Paley order (``m - 1`` prime and
    ``= 3 mod 4``) -> ``kron(H_m_paley, H_{2^k}_sylvester)``.
    6144 = 12 * 512, and 11 is prime and 3 mod 4, so 6144 is exact.
  * otherwise it RAISES.  ``allow_random=True`` opts into a random orthogonal
    fallback, and the construction name returned is ``"random_orthogonal"`` so the
    caller can say which was used.  Nothing here ever silently returns a
    non-orthogonal matrix: every construction is asserted orthonormal.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "hadamard_matrix",
    "hadamard_construction",
    "random_hadamard",
    "fuse_input_rotation",
    "fuse_output_rotation",
    "fold_rmsnorm_into_linears",
    "rotate_qwen3",
]


# --------------------------------------------------------------------- Hadamard
def _is_prime(p: int) -> bool:
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    i = 3
    while i * i <= p:
        if p % i == 0:
            return False
        i += 2
    return True


def _sylvester(k: int, dtype=torch.float64) -> torch.Tensor:
    """Unnormalized +-1 Sylvester Hadamard of order 2^k."""
    H = torch.ones(1, 1, dtype=dtype)
    for _ in range(k):
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def _paley1(q: int, dtype=torch.float64) -> torch.Tensor:
    """Unnormalized +-1 Paley type-I Hadamard of order q+1, for prime q = 3 (mod 4)."""
    assert _is_prime(q) and q % 4 == 3, f"Paley I needs prime q = 3 mod 4, got {q}"
    residues = {(i * i) % q for i in range(1, q)}
    chi = torch.zeros(q, dtype=dtype)
    for a in range(q):
        chi[a] = 0.0 if a == 0 else (1.0 if a in residues else -1.0)
    idx = (torch.arange(q).unsqueeze(1) - torch.arange(q).unsqueeze(0)) % q
    Q = chi[idx].to(dtype)
    n = q + 1
    S = torch.zeros(n, n, dtype=dtype)
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    S[1:, 1:] = Q
    return torch.eye(n, dtype=dtype) + S


def _factor(n: int):
    """Return (m, k) with n = m * 2^k, m the smallest usable Paley base (or 1)."""
    k = 0
    m = n
    while m % 2 == 0:
        m //= 2
        k += 1
    if m == 1:
        return 1, k
    # try to move powers of two from the 2^k part into the base until the base is a
    # valid Paley order (base % 4 == 0 and base-1 prime = 3 mod 4)
    for j in range(k + 1):
        base = m * (2 ** j)
        if base % 4 == 0 and _is_prime(base - 1) and (base - 1) % 4 == 3:
            return base, k - j
    return None, None


def hadamard_construction(n: int) -> str:
    """Name of the construction ``hadamard_matrix(n)`` would use, without building it."""
    m, k = _factor(n)
    if m is None:
        return "none"
    if m == 1:
        return f"sylvester_2^{k}"
    return f"kron(paley1_{m}, sylvester_2^{k})"


def hadamard_matrix(n: int, dtype=torch.float64, allow_random: bool = False,
                    seed: int = 0):
    """
    Exact orthonormal Hadamard matrix of order ``n`` (i.e. ``H_pm / sqrt(n)``).

    Returns ``(H, kind)`` where ``kind`` names the construction actually used, so the
    caller can report it.  Raises unless ``allow_random`` when no exact construction
    is known -- we never silently substitute a non-Hadamard matrix.

    Every returned matrix is asserted orthonormal to 1e-9.
    """
    m, k = _factor(n)
    if m is None:
        if not allow_random:
            raise ValueError(
                f"no exact Hadamard construction known for n={n} "
                f"(n = odd*2^k with odd part not reachable as a Paley order). "
                f"Pass allow_random=True to fall back to a random orthogonal matrix."
            )
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(n, n, generator=g, dtype=torch.float64)
        H, _ = torch.linalg.qr(A)
        kind = "random_orthogonal"
    elif m == 1:
        H = _sylvester(k, dtype=torch.float64) / math.sqrt(n)
        kind = f"sylvester_2^{k}"
    else:
        Hm = _paley1(m - 1) / math.sqrt(m)
        Hs = _sylvester(k, dtype=torch.float64) / math.sqrt(2 ** k)
        H = torch.kron(Hm, Hs)
        kind = f"kron(paley1_{m}, sylvester_2^{k})"
    err = (H @ H.T - torch.eye(n, dtype=torch.float64)).abs().max().item()
    assert err < 1e-9, f"hadamard_matrix({n}) [{kind}] not orthonormal: err={err:.3e}"
    return H.to(dtype), kind


def random_hadamard(n: int, seed: int = 0, dtype=torch.float64,
                    allow_random: bool = False):
    """
    Randomized Hadamard ``H @ diag(d)``, ``d`` uniform in ``{-1,+1}`` -- the variant
    QuaRot actually uses.  Still orthogonal, and still add/subtract + sign flips at
    inference (a ``+-1`` diagonal is a sign flip, not a multiply).

    Returns ``(Q, kind)``.
    """
    H, kind = hadamard_matrix(n, dtype=torch.float64, allow_random=allow_random,
                              seed=seed)
    g = torch.Generator().manual_seed(seed + 12345)
    d = (torch.randint(0, 2, (n,), generator=g, dtype=torch.float64) * 2 - 1)
    Q = H * d.unsqueeze(0)          # H @ diag(d)
    err = (Q @ Q.T - torch.eye(n, dtype=torch.float64)).abs().max().item()
    assert err < 1e-9, f"random_hadamard({n}) not orthonormal: err={err:.3e}"
    return Q.to(dtype), "random_" + kind


# ------------------------------------------------------------------ fusion helpers
def _matmul_chunked(W: torch.Tensor, Q: torch.Tensor, dev, chunk=16384) -> torch.Tensor:
    """W @ Q in float32 on `dev`, row-chunked so the embedding matrix fits easily."""
    out = torch.empty_like(W)
    Qd = Q.to(dev, torch.float32)
    for i in range(0, W.shape[0], chunk):
        blk = W[i:i + chunk].to(dev, torch.float32)
        out[i:i + chunk] = (blk @ Qd).to(W.dtype).to(W.device)
    return out


@torch.no_grad()
def fuse_input_rotation(lin: nn.Linear, Q: torch.Tensor, dev="cuda") -> None:
    """
    Rotate the INPUT of ``lin``: ``W <- W Q``.

    ``lin`` then expects ``Q^T x`` instead of ``x``.  Exactly output preserving when
    the producer of ``x`` has the matching output rotation fused (or when ``Q^T x`` is
    applied online).  Bias is untouched (it is on the output side).
    """
    assert lin.weight.shape[1] == Q.shape[0], (lin.weight.shape, Q.shape)
    lin.weight.data = _matmul_chunked(lin.weight.data, Q, dev)


@torch.no_grad()
def fuse_output_rotation(lin: nn.Linear, Q: torch.Tensor, dev="cuda") -> None:
    """
    Rotate the OUTPUT of ``lin``: ``W <- Q^T W`` (and ``b <- Q^T b``).

    The consumer of ``lin``'s output must have the matching input rotation fused.
    """
    assert lin.weight.shape[0] == Q.shape[0], (lin.weight.shape, Q.shape)
    W = lin.weight.data
    Qd = Q.to(dev, torch.float32)
    lin.weight.data = (Qd.T @ W.to(dev, torch.float32)).to(W.dtype).to(W.device)
    if lin.bias is not None:
        b = lin.bias.data
        lin.bias.data = (Qd.T @ b.to(dev, torch.float32)).to(b.dtype).to(b.device)


@torch.no_grad()
def fold_rmsnorm_into_linears(norm: nn.Module, linears, dev="cuda") -> None:
    """
    Absorb an RMSNorm's learned gain into every Linear that consumes its output, and
    set the gain to 1.

    RMSNorm(x) = g * x / rms(x).  The normalization ``x / rms(x)`` commutes with any
    orthogonal ``Q`` because ``rms(Q^T x) = rms(x)``; the *gain* does not, since
    ``diag(g) Q != Q diag(g)``.  So the gain must move downstream before rotating.
    After this call the norm is a pure normalizer and the rotation commutes with it.

    Exact in exact arithmetic: ``W (g * u) = (W diag(g)) u``.
    """
    g = norm.weight.data.detach().float()
    for lin in linears:
        assert lin.weight.shape[1] == g.numel(), (lin.weight.shape, g.numel())
        W = lin.weight.data
        Wf = W.to(dev, torch.float32) * g.to(dev).unsqueeze(0)
        lin.weight.data = Wf.to(W.dtype).to(W.device)
    norm.weight.data = torch.ones_like(norm.weight.data)


class _OnlineRotate:
    """Forward-PRE-hook: applies ``x <- x @ Q`` to the input of a Linear.

    At inference this is a fast Walsh-Hadamard transform: adds/subtracts plus one
    scalar ``1/sqrt(n)``.  It is written as a matmul here only so the equivalence is
    obvious and easy to check; it introduces no per-weight multiply either way.

    One instance is shared by every layer, and the matrix is cached per device, so the
    6144x6144 fp32 matrix is materialized once, not 28 times.

    NB: PyTorch runs forward-pre-hooks BEFORE forward hooks and passes the *modified*
    args on, so a GPTQ ``add_batch`` forward hook on the same Linear correctly sees the
    rotated activation.
    """

    def __init__(self, Q):
        self._Q_cpu = Q.to(torch.float32)
        self._cache = {}

    def _q(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = self._Q_cpu.to(device)
        return self._cache[key]

    def __call__(self, module, args):
        x = args[0]
        Q = self._q(x.device)
        xr = (x.to(torch.float32) @ Q).to(x.dtype)
        return (xr,) + tuple(args[1:])


class _OnlinePostRotate:
    """Forward-POST-hook: applies ``h <- h @ Q`` to a module's OUTPUT.

    Used on ``embed_tokens`` so the residual stream is rotated without touching the
    (tied) embedding weight.  One Hadamard per token: adds/subtracts plus one scalar.
    """

    def __init__(self, Q):
        self._Q_cpu = Q.to(torch.float32)
        self._cache = {}

    def _q(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = self._Q_cpu.to(device)
        return self._cache[key]

    def __call__(self, module, args, output):
        Q = self._q(output.device)
        return (output.to(torch.float32) @ Q).to(output.dtype)


class _OnlineUnrotateAndGain:
    """Forward-PRE-hook for ``lm_head``: ``x <- (x @ Q^T) * g``.

    Undoes the residual-stream rotation and applies the final RMSNorm gain that was
    lifted out of ``model.norm`` (whose weight is left at ones).  Exactly equivalent to
    folding both into ``lm_head.weight``, but it leaves ``lm_head`` tied to
    ``embed_tokens``, saving 0.622 GB on Qwen3-1.7B.  One Hadamard plus one length-n
    vector scale per token -- negligible next to the vocab-sized matmul that follows,
    and NOT a per-weight multiply.
    """

    def __init__(self, Q, g):
        self._Qt_cpu = Q.to(torch.float32).T.contiguous()
        self._g_cpu = g.to(torch.float32)
        self._cache = {}

    def _qg(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = (self._Qt_cpu.to(device), self._g_cpu.to(device))
        return self._cache[key]

    def __call__(self, module, args):
        x = args[0]
        Qt, g = self._qg(x.device)
        xr = ((x.to(torch.float32) @ Qt) * g).to(x.dtype)
        return (xr,) + tuple(args[1:])


# ------------------------------------------------------------------ Qwen3 driver
@torch.no_grad()
def rotate_qwen3(model, mode: str = "hadamard", seed: int = 0, dev: str = "cuda",
                 do_r1: bool = True, do_r2: bool = True, do_r4: bool = True,
                 untie_lm_head: bool = False, verbose: bool = True):
    """
    Fuse QuaRot-style rotations into a Qwen3 causal LM, in place.

    ``mode``: ``"hadamard"`` (deterministic) or ``"random_hadamard"`` (Hadamard times a
    random +-1 diagonal, the QuaRot default).  ``"none"`` is a no-op.

    ``untie_lm_head`` (default False): when True, clone ``lm_head.weight`` away from the
    tied ``embed_tokens`` so R1 and the final-norm gain can be FUSED into both.  This
    costs +0.622 GB in bf16 on Qwen3-1.7B (vocab 151936 x 2048), which is +83% on top of
    the whole 4.25 bpw quantized decoder -- so it is off by default and the online
    variant, which is mathematically identical, is used instead.

    Returns an info dict (constructions used, which paths were rotated, whether the
    lm_head was untied).  The returned model is mathematically equivalent to the
    input model; see ``scripts``/the validation run for the empirical check.
    """
    if mode in (None, "none"):
        return {"mode": "none", "rotated": []}
    assert mode in ("hadamard", "random_hadamard"), mode

    def mk(n):
        if mode == "hadamard":
            return hadamard_matrix(n, dtype=torch.float64)
        return random_hadamard(n, seed=seed, dtype=torch.float64)

    cfg = model.config
    hidden = cfg.hidden_size
    head_dim = cfg.head_dim              # 128 for Qwen3-1.7B; NOT hidden//num_heads
    inter = cfg.intermediate_size
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads

    mm = model.model
    layers = mm.layers
    info = {"mode": mode, "seed": seed, "rotated": [], "not_rotated": [],
            "constructions": {}, "untied_lm_head": False,
            "head_dim": head_dim, "hidden": hidden, "intermediate": inter,
            "num_heads": n_q, "num_kv_heads": n_kv}

    # ---- lm_head / embed_tokens: tied by default, see hazard 1 --------------
    tied = (model.lm_head.weight.data_ptr() == mm.embed_tokens.weight.data_ptr())
    info["lm_head_tied_on_entry"] = tied
    if untie_lm_head and tied and do_r1:
        # EXPLICITLY COSTED OPTION, not the default: +0.622 GB bf16 on Qwen3-1.7B.
        model.lm_head.weight = nn.Parameter(
            mm.embed_tokens.weight.data.clone(), requires_grad=False)
        model.config.tie_word_embeddings = False
        if hasattr(model, "_tied_weights_keys"):
            model._tied_weights_keys = []
        info["untied_lm_head"] = True
        info["untie_cost_bytes"] = int(model.lm_head.weight.numel()
                                       * model.lm_head.weight.element_size())
        assert model.lm_head.weight.data_ptr() != mm.embed_tokens.weight.data_ptr()

    # ---- R1: residual stream ------------------------------------------------
    if do_r1:
        Q1, kind1 = mk(hidden)
        info["constructions"]["R1(%d)" % hidden] = kind1
        # 1a. fold the per-layer RMSNorm gains forward into their consumers
        for lyr in layers:
            fold_rmsnorm_into_linears(
                lyr.input_layernorm,
                [lyr.self_attn.q_proj, lyr.self_attn.k_proj, lyr.self_attn.v_proj], dev)
            fold_rmsnorm_into_linears(
                lyr.post_attention_layernorm,
                [lyr.mlp.gate_proj, lyr.mlp.up_proj], dev)
        # 1b. rotate the decoder linears: producers on the output side, consumers on
        #     the input side. These are the matrices we actually quantize.
        for lyr in layers:
            for lin in (lyr.self_attn.q_proj, lyr.self_attn.k_proj,
                        lyr.self_attn.v_proj, lyr.mlp.gate_proj, lyr.mlp.up_proj):
                fuse_input_rotation(lin, Q1, dev)
            fuse_output_rotation(lyr.self_attn.o_proj, Q1, dev)
            fuse_output_rotation(lyr.mlp.down_proj, Q1, dev)
        # 1c. enter and leave the residual stream.
        if info["untied_lm_head"]:
            # fused variant: only reachable with untie_lm_head=True (+0.622 GB)
            fold_rmsnorm_into_linears(mm.norm, [model.lm_head], dev)
            mm.embed_tokens.weight.data = _matmul_chunked(
                mm.embed_tokens.weight.data, Q1, dev)
            fuse_input_rotation(model.lm_head, Q1, dev)
            info["rotated"].append(
                "R1 residual stream; embed_tokens/lm_head FUSED (lm_head UNTIED, "
                "+%.3f GB bf16)" % (info["untie_cost_bytes"] / 1e9))
        else:
            # default: leave the tied embed_tokens/lm_head weights untouched and do the
            # two endpoints online. Mathematically identical, 0 extra bytes stored.
            g_final = mm.norm.weight.data.detach().float().clone()
            mm.norm.weight.data = torch.ones_like(mm.norm.weight.data)
            emb_hook = _OnlinePostRotate(Q1)
            head_hook = _OnlineUnrotateAndGain(Q1, g_final)
            model._p2q_embed_rot = emb_hook
            model._p2q_head_unrot = head_hook
            model._p2q_embed_handle = mm.embed_tokens.register_forward_hook(emb_hook)
            model._p2q_head_handle = model.lm_head.register_forward_pre_hook(head_hook)
            info["rotated"].append(
                "R1 residual stream (q/k/v/gate/up input, o/down output fused); "
                "embed_tokens output and lm_head input rotated ONLINE so the tied "
                "embed/lm_head weights are untouched (0 extra bytes); final RMSNorm "
                "gain carried into the lm_head hook")
    else:
        info["not_rotated"].append("R1 residual stream (disabled)")

    # ---- R2: per-head value / attention-output ------------------------------
    if do_r2:
        Q2, kind2 = mk(head_dim)
        info["constructions"]["R2(%d)" % head_dim] = kind2
        Q2d = Q2.to(dev, torch.float32)
        for lyr in layers:
            attn = lyr.self_attn
            Wv = attn.v_proj.weight.data
            assert Wv.shape[0] == n_kv * head_dim, Wv.shape
            Wvf = Wv.to(dev, torch.float32).view(n_kv, head_dim, -1)
            Wvf = torch.einsum("dk,hdc->hkc", Q2d, Wvf)          # Q2^T @ block
            attn.v_proj.weight.data = Wvf.reshape(n_kv * head_dim, -1).to(Wv.dtype).to(Wv.device)
            assert attn.v_proj.bias is None, "v_proj bias unsupported for R2"
            Wo = attn.o_proj.weight.data
            assert Wo.shape[1] == n_q * head_dim, Wo.shape
            Wof = Wo.to(dev, torch.float32).view(-1, n_q, head_dim)
            Wof = torch.einsum("rhd,dk->rhk", Wof, Q2d)          # block @ Q2
            attn.o_proj.weight.data = Wof.reshape(Wo.shape).to(Wo.dtype).to(Wo.device)
        info["rotated"].append(
            "R2 per-head value path (v_proj output rows, o_proj input columns), "
            "same %dx%d Hadamard on every head" % (head_dim, head_dim))
    else:
        info["not_rotated"].append("R2 value/output head rotation (disabled)")

    # ---- R4: down_proj input (online) ---------------------------------------
    if do_r4:
        Q4, kind4 = mk(inter)
        info["constructions"]["R4(%d)" % inter] = kind4
        _hook4 = _OnlineRotate(Q4)
        model._p2q_online_rotation = _hook4
        for lyr in layers:
            fuse_input_rotation(lyr.mlp.down_proj, Q4, dev)
            h = lyr.mlp.down_proj.register_forward_pre_hook(_hook4)
            lyr.mlp._p2q_rot_handle = h
        info["rotated"].append(
            "R4 down_proj input, ONLINE Hadamard of size %d (behind SwiGLU, "
            "cannot be fused; add/sub + one scalar 1/sqrt(n))" % inter)
    else:
        info["not_rotated"].append("R4 down_proj input rotation (disabled)")

    # ---- what we deliberately did NOT rotate --------------------------------
    info["not_rotated"].append(
        "q_proj/k_proj OUTPUT side: blocked by Qwen3's per-head q_norm/k_norm "
        "(RMSNorm applied after the projection, before RoPE). A rotation folded out "
        "of q/k does not commute through them, so it is not equivalence preserving "
        "and is skipped. Their INPUT side is rotated by R1.")

    if verbose:
        print("[rotation] mode=%s" % mode, flush=True)
        for k, v in info["constructions"].items():
            print("[rotation]   %s -> %s" % (k, v), flush=True)
        for s in info["rotated"]:
            print("[rotation]   APPLIED: %s" % s, flush=True)
        for s in info["not_rotated"]:
            print("[rotation]   SKIPPED: %s" % s, flush=True)
        print("[rotation]   untied lm_head: %s%s" % (
            info["untied_lm_head"],
            "  (+%.3f GB bf16)" % (info["untie_cost_bytes"] / 1e9)
            if info["untied_lm_head"] else ""), flush=True)
    return info
