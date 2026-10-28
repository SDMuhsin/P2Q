"""Unit tests for src/pot/rotation.py -- the algebra, in float64, without a GPU."""
import os, sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from pot.rotation import (hadamard_matrix, random_hadamard, fuse_input_rotation,
                          fuse_output_rotation, fold_rmsnorm_into_linears)

SIZES = [1, 2, 4, 12, 20, 24, 128, 1024, 2048, 6144]


def test_hadamard_orthonormal():
    for n in SIZES:
        H, kind = hadamard_matrix(n)
        assert H.shape == (n, n), kind
        err = (H @ H.T - torch.eye(n, dtype=H.dtype)).abs().max().item()
        assert err < 1e-9, (n, kind, err)


def test_hadamard_entries_are_pm_one_over_sqrt_n():
    """Every entry is +-1/sqrt(n): that is what makes the transform add/subtract only."""
    for n in [4, 12, 128, 6144]:
        H, _ = hadamard_matrix(n)
        assert torch.allclose(H.abs(), torch.full_like(H, n ** -0.5), atol=1e-12)


def test_random_hadamard_orthonormal_and_seeded():
    Q1, _ = random_hadamard(256, seed=3)
    Q2, _ = random_hadamard(256, seed=3)
    Q3, _ = random_hadamard(256, seed=4)
    assert torch.equal(Q1, Q2)
    assert not torch.equal(Q1, Q3)
    assert (Q1 @ Q1.T - torch.eye(256, dtype=Q1.dtype)).abs().max() < 1e-12


def test_no_silent_non_orthogonal_fallback():
    try:
        hadamard_matrix(2 * 1025)          # 2050 = 2 * 5^2 * 41, no known construction
    except ValueError:
        pass
    else:
        raise AssertionError("expected a raise, not a silent fallback")
    H, kind = hadamard_matrix(2050, allow_random=True)
    assert kind == "random_orthogonal"
    assert (H @ H.T - torch.eye(2050, dtype=H.dtype)).abs().max() < 1e-9


def test_fuse_input_output_rotation_is_identity_pair():
    torch.manual_seed(0)
    n, m = 128, 64
    Q, _ = random_hadamard(n, seed=1)
    Q = Q.float()
    prod = nn.Linear(32, n, bias=False).float()      # producer
    cons = nn.Linear(n, m, bias=False).float()       # consumer
    x = torch.randn(7, 32)
    ref = cons(prod(x))
    fuse_output_rotation(prod, Q, dev="cpu")
    fuse_input_rotation(cons, Q, dev="cpu")
    assert torch.allclose(ref, cons(prod(x)), atol=1e-4)


def test_fold_rmsnorm_is_exact():
    torch.manual_seed(0)

    class RMS(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(n) + 0.5)
            self.variance_epsilon = 1e-6

        def forward(self, x):
            v = x.pow(2).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(v + self.variance_epsilon))

    n = 64
    norm = RMS(n).float()
    lin = nn.Linear(n, 32, bias=False).float()
    x = torch.randn(5, n)
    ref = lin(norm(x))
    fold_rmsnorm_into_linears(norm, [lin], dev="cpu")
    assert torch.allclose(norm.weight, torch.ones(n))
    assert torch.allclose(ref, lin(norm(x)), atol=1e-5)


def test_rmsnorm_normalizer_commutes_with_rotation():
    """The gain must be folded, but x/rms(x) itself is rotation-equivariant."""
    n = 128
    Q, _ = random_hadamard(n, seed=2)
    x = torch.randn(9, n, dtype=torch.float64)
    f = lambda z: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(f(x @ Q), f(x) @ Q, atol=1e-10)


def test_online_endpoints_equal_the_fused_untied_version():
    """The default (no-untie) path must be *identical* to fusing R1 + the final norm
    gain into an untied lm_head. This is the whole justification for not paying the
    0.622 GB untie cost on Qwen3-1.7B."""
    from pot.rotation import _OnlinePostRotate, _OnlineUnrotateAndGain
    torch.manual_seed(0)
    n, V = 64, 40
    Q, _ = random_hadamard(n, seed=7)
    Q = Q.float()
    g = torch.rand(n) + 0.5                       # final RMSNorm gain
    E = torch.randn(V, n)                         # tied embed / lm_head weight
    h = torch.randn(3, n)                         # residual stream (unrotated)

    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    ref = (g * rms(h)) @ E.T                      # original: norm(h) @ lm_head^T

    # (a) fused + untied: lm_head weight becomes (E * g) @ Q, norm gain -> 1
    fused = rms(h @ Q) @ ((E * g) @ Q).T
    assert torch.allclose(ref, fused, atol=1e-4)

    # (b) online, weights untouched and still tied: x <- (x @ Q^T) * g before lm_head
    hook = _OnlineUnrotateAndGain(Q, g)
    x = rms(h @ Q)                                # what model.norm emits, gain = 1
    online = hook(None, (x,))[0] @ E.T
    assert torch.allclose(ref, online, atol=1e-4)
    assert torch.allclose(fused, online, atol=1e-4)

    # and the embedding-side hook rotates the stream without touching E
    post = _OnlinePostRotate(Q)
    assert torch.allclose(post(None, None, h), h @ Q, atol=1e-5)


def test_deterministic_hadamard_mode_is_seed_independent():
    """`--rotate hadamard` must carry NO randomness, so a run of it has no
    rotation-seed variance at all. (The calibration draw is a separate, independent
    source of variance and is NOT removed by this.)"""
    for n in (128, 2048, 6144):
        a, ka = hadamard_matrix(n, seed=0)
        b, kb = hadamard_matrix(n, seed=7)
        assert torch.equal(a, b) and ka == kb, n
        # and the randomized variant genuinely does depend on the seed
        ra, _ = random_hadamard(n, seed=0)
        rb, _ = random_hadamard(n, seed=7)
        assert not torch.equal(ra, rb), n
