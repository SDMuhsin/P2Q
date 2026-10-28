"""Unit tests for src/pot/quantizers.py.  Run: python3 src/pot/test_quantizers.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pot.quantizers import rtn_int_asym, rtn_int_sym, QUANTIZERS


def main():
    torch.manual_seed(0)
    W = torch.randn(512, 1024, dtype=torch.bfloat16)
    ok = True

    for name, fn in QUANTIZERS.items():
        # 1. bits=16 is a pass-through (lossless).
        W16, m16 = fn(W, bits=16, group=128)
        assert torch.equal(W16, W), f"{name}: bits=16 not lossless"

        # 2. error decreases monotonically with bits.
        errs = []
        for b in (2, 3, 4, 5, 6, 8):
            W_hat, meta = fn(W, bits=b, group=128)
            assert W_hat.shape == W.shape and W_hat.dtype == W.dtype, f"{name}: shape/dtype"
            errs.append(meta["rel_fro_err"])
        mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
        print(f"{name:14s} rel_fro_err by bits [2,3,4,5,6,8] = "
              + ", ".join(f"{e:.4f}" for e in errs) + f"  monotone={mono}")
        ok &= mono

        # 3. smaller groups -> lower error.
        e_big, _ = None, None
        eg = [fn(W, bits=4, group=g)[1]["rel_fro_err"] for g in (1024, 256, 128, 64, 32)]
        mono_g = all(eg[i] > eg[i + 1] for i in range(len(eg) - 1))
        print(f"{name:14s} rel_fro_err @4b by group [1024,256,128,64,32] = "
              + ", ".join(f"{e:.4f}" for e in eg) + f"  monotone={mono_g}")
        ok &= mono_g

        # 4. 8-bit should be very accurate.
        _, m8 = fn(W, bits=8, group=128)
        assert m8["rel_fro_err"] < 0.01, f"{name}: 8-bit rel err {m8['rel_fro_err']}"

    # 5. asymmetric beats symmetric at the same bits/group on skewed data.
    Wskew = torch.randn(256, 512).abs() + 0.5     # strictly positive -> sym wastes half the grid
    ea = rtn_int_asym(Wskew, 4, 128)[1]["rel_fro_err"]
    es = rtn_int_sym(Wskew, 4, 128)[1]["rel_fro_err"]
    print(f"skewed data @4b: asym={ea:.4f}  sym={es:.4f}  (asym should win)")
    ok &= ea < es

    # 6. effective bits accounting.
    _, m = rtn_int_asym(W, 4, 128)
    assert abs(m["effective_bits"] - (4 + 32 / 128)) < 1e-9
    _, m = rtn_int_sym(W, 4, 128)
    assert abs(m["effective_bits"] - (4 + 16 / 128)) < 1e-9

    print("\nALL TESTS PASSED" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
