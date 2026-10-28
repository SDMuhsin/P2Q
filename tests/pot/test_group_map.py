"""Anchor tests for --group-map: per-matrix group size with honest bpw accounting.

The whole point of --group-map is that `account_bpw` must not care whether the groups
it is summing came from one scalar --group or from many different per-matrix group
sizes -- it only ever sums n_groups_total = sum_over_matrices(rows_i * n_groups_i).
The anchor test below is the one the brief calls for: a --group-map whose every entry
resolves to the SAME group as a plain --group run must reproduce that run's bpw
bit-for-bit, because it must resolve to literally the same n_groups_total.
"""
import pytest

from pot.run_pot import account_bpw, parse_group_map, resolve_group


# A handful of Qwen3-1.7B-shaped linears (rows=out_features, cols=in_features) so the
# arithmetic below is not a toy -- these are the real per-layer matrix shapes.
LINEARS = {
    "self_attn.q_proj": (2048, 2048),
    "self_attn.k_proj": (1024, 2048),
    "self_attn.v_proj": (1024, 2048),
    "self_attn.o_proj": (2048, 2048),
    "mlp.gate_proj": (6144, 2048),
    "mlp.up_proj": (6144, 2048),
    "mlp.down_proj": (2048, 6144),
}


def _n_groups_total(group_of):
    """group_of: name -> group size. Mirrors exactly what run_pot.py accumulates:
    n_groups_total += W_hat.shape[0] * n_groups, n_groups = cols // group."""
    n_weights = 0
    n_groups_total = 0
    for name, (rows, cols) in LINEARS.items():
        g = group_of(name)
        assert cols % g == 0, f"{name}: group {g} does not divide cols {cols}"
        n_groups_total += rows * (cols // g)
        n_weights += rows * cols
    return n_weights, n_groups_total


def test_uniform_group_map_matches_scalar_group_bit_for_bit():
    """--group-map 'default=64' must give the identical bpw as --group 64."""
    scalar_group = 64
    n_weights_a, n_groups_a = _n_groups_total(lambda name: scalar_group)

    gm = parse_group_map("default=64")
    n_weights_b, n_groups_b = _n_groups_total(lambda name: resolve_group(name, gm, 999))

    assert n_weights_a == n_weights_b
    assert n_groups_a == n_groups_b  # exact int equality, not just close

    bpw_a, brk_a = account_bpw(n_weights=n_weights_a, n_groups_total=n_groups_a,
                                n_rows_total=0, code_bits=4, group=scalar_group,
                                scale_bits=16, zero_bits=0, col_scale_bits=16,
                                col_scale_entries=0, codebook_entries=16)
    bpw_b, brk_b = account_bpw(n_weights=n_weights_b, n_groups_total=n_groups_b,
                                n_rows_total=0, code_bits=4, group=None,
                                scale_bits=16, zero_bits=0, col_scale_bits=16,
                                col_scale_entries=0, codebook_entries=16)
    assert bpw_a == bpw_b
    assert brk_a == brk_b
    # matches the incumbent S_g64_ao's reported bpw exactly
    assert round(bpw_a, 4) == 4.2500


def test_uniform_group_map_matches_scalar_group_at_g128():
    """Same anchor at g=128, the other uniform arm actually run in the log (4.1250)."""
    n_weights_a, n_groups_a = _n_groups_total(lambda name: 128)
    gm = parse_group_map("default=128")
    n_weights_b, n_groups_b = _n_groups_total(lambda name: resolve_group(name, gm, 999))
    assert (n_weights_a, n_groups_a) == (n_weights_b, n_groups_b)
    bpw, _ = account_bpw(n_weights=n_weights_b, n_groups_total=n_groups_b,
                          n_rows_total=0, code_bits=4, group=None,
                          scale_bits=16, zero_bits=0, col_scale_bits=16,
                          col_scale_entries=0, codebook_entries=16)
    assert round(bpw, 4) == 4.1250


def test_group_map_resolution_longest_suffix_and_default():
    gm = parse_group_map("default=64,mlp.gate_proj=128,mlp.down_proj=32,proj=16")
    # exact / most-specific suffix wins over the shorter 'proj' catch-all
    assert resolve_group("mlp.gate_proj", gm, 999) == 128
    assert resolve_group("mlp.down_proj", gm, 999) == 32
    # matches only the generic 'proj' suffix
    assert resolve_group("self_attn.q_proj", gm, 999) == 16
    assert resolve_group("self_attn.k_proj", gm, 999) == 16
    # no suffix matches at all -> default
    assert resolve_group("something_else", gm, 999) == 64


def test_group_map_requires_default():
    with pytest.raises(ValueError):
        parse_group_map("mlp.gate_proj=128")


def test_empty_group_map_falls_back_to_scalar_group():
    assert parse_group_map("") == {}
    assert parse_group_map(None) == {}
    assert resolve_group("mlp.gate_proj", {}, 128) == 128


def test_nonuniform_allocation_honest_accounting_example():
    """The B_map_mlp_realloc candidate: gate/up coarsened to 128, down_proj sharpened
    to 32, attention left at the incumbent's 64 -- must land at <= 4.2500 with an
    exact, auditable breakdown (this mirrors the actual arm run in this session)."""
    group_of = {
        "self_attn.q_proj": 64, "self_attn.k_proj": 64, "self_attn.v_proj": 64,
        "self_attn.o_proj": 64, "mlp.gate_proj": 128, "mlp.up_proj": 128,
        "mlp.down_proj": 32,
    }
    n_weights, n_groups_total = _n_groups_total(lambda name: group_of[name])
    bpw, breakdown = account_bpw(n_weights=n_weights, n_groups_total=n_groups_total,
                                  n_rows_total=0, code_bits=4, group=None,
                                  scale_bits=16, zero_bits=0, col_scale_bits=16,
                                  col_scale_entries=0, codebook_entries=16)
    # the codebook itself costs an intrinsic ~1.8e-7 bpw regardless of group allocation
    # (16 fp16 entries amortized over the whole model) -- the incumbent S_g64_ao's own
    # logged bpw is 4.250000181652251 for exactly this reason, so the tolerance below
    # is real headroom for that fixed term, not slack for the reallocation itself.
    assert bpw <= 4.2500 + 1e-3
    assert round(bpw, 4) == 4.2500  # exact zero-sum reallocation, see report
    assert breakdown["codes"] == 4.0
