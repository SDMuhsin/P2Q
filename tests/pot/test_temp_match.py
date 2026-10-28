import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pot.temp_match import _kl_at, apply_temperature


def test_kl_is_zero_at_matching_temperature_and_convex_around_it():
    torch.manual_seed(0)
    zt = [torch.randn(64, 100) * 3]
    zs = [zt[0] / 1.15]                       # student is exactly a cooler teacher
    kls = {a: _kl_at(a, zt, zs, "cpu") for a in (1.0, 1.10, 1.15, 1.20, 1.3)}
    assert kls[1.15] < 1e-5
    assert kls[1.10] < kls[1.0] and kls[1.20] < kls[1.3]


def test_apply_temperature_scales_logits_exactly():
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.model = torch.nn.Module(); s.model.norm = torch.nn.RMSNorm(8)
            s.model.norm.weight.data = torch.rand(8) + 0.5
            s.head = torch.nn.Linear(8, 5, bias=False)
        def forward(s, h): return s.head(s.model.norm(h))
    m = M(); h = torch.randn(3, 8)
    z0 = m(h).clone(); apply_temperature(m, 1.3)
    assert torch.allclose(m(h), 1.3 * z0, rtol=1e-5, atol=1e-6)
