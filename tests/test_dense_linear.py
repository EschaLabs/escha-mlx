"""Dense (single-stream) escha linear: numerics, metadata guards, kernel parity.

The codec itself is already gated by the format-level goldens in
tests/test_metal.py / test_ref_decode.py. What is new here is everything the
dense path adds on top of it:

  * the folded transform vectors — a dense export ships the end-to-end scales
    s_in/s_out separately from rin/rout, and ``ref.fold_scales`` multiplies them
    in at load time;
  * the additive bias the end-to-end stage leaves behind;
  * the single-stream Metal kernels, which must stay bit-identical to the
    expert kernels they are a compile-time variant of.

The cross-runtime gate is ``dense_linear_golden``: real coded data from a
shipped checkpoint together with the reference output shipped for it. That
reference and this one round at different points by design (see
``ref.fold_scales``), so it is a tolerance check — the *bit*-exact gates are
between this package's own reference and its kernels.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_metal, needs_mlx

# Cross-runtime agreement bar. The reference path rounds x*s_in to f16
# before the transform and rounds again before applying s_out; this one folds
# both into the transform vectors and rounds once. That is a sub-ulp difference
# per weight, so the two agree to fp16 rounding on the output — measured
# ~6e-4 relative on full-size shipped tensors.
REL_TOL = 3e-3


def _ref():
    from escha_mlx import ref
    return ref


# --------------------------------------------------------------- fold_scales


def test_fold_scales_is_the_product_in_f32():
    ref = _ref()
    rin = np.array([2.0, 4.0], dtype=np.float16)
    rout = np.array([3.0], dtype=np.float16)
    s_in = np.array([0.5, 0.25], dtype=np.float32)
    s_out = np.array([2.0], dtype=np.float32)
    ri, ro = ref.fold_scales(rin, rout, s_in, s_out)
    assert ri.dtype == np.float32 and ro.dtype == np.float32
    assert np.array_equal(ri, np.array([1.0, 1.0], dtype=np.float32))
    assert np.array_equal(ro, np.array([6.0], dtype=np.float32))


def test_fold_scales_without_end_to_end_scales_is_a_cast():
    """MoE exports carry no s_in/s_out; folding must then change nothing."""
    ref = _ref()
    rin = np.array([2.0, 4.0], dtype=np.float16)
    rout = np.array([3.0], dtype=np.float16)
    ri, ro = ref.fold_scales(rin, rout, None, None)
    assert np.array_equal(ri, rin.astype(np.float32))
    assert np.array_equal(ro, rout.astype(np.float32))


# ------------------------------------------------------- reference numerics


def test_reference_matches_deploy_runtime(dense_linear_golden):
    """ref.dense_linear vs the shipped reference output on real coded data."""
    ref = _ref()
    g = dense_linear_golden
    rin, rout = ref.fold_scales(g["rin"], g["rout"], g["s_in"], g["s_out"])
    y = ref.dense_linear(g["x"], g["code"], rin, rout, g["K"], bias=g["bias"])
    want = g["deploy"].astype(np.float32)
    got = y.astype(np.float32)
    rel = np.abs(got - want).mean() / np.abs(want).mean()
    assert rel < REL_TOL, f"K={g['K']} relative disagreement {rel:.2e}"
    assert np.corrcoef(got.ravel(), want.ravel())[0, 1] > 0.9999


def test_bias_is_actually_applied(dense_linear_golden):
    """A dropped bias is the quietest possible failure: still finite, still
    correlated, just wrong. Pin that it moves the output by exactly the bias."""
    ref = _ref()
    g = dense_linear_golden
    rin, rout = ref.fold_scales(g["rin"], g["rout"], g["s_in"], g["s_out"])
    with_b = ref.dense_linear(g["x"], g["code"], rin, rout, g["K"], bias=g["bias"])
    no_b = ref.dense_linear(g["x"], g["code"], rin, rout, g["K"], bias=None)
    assert not np.array_equal(with_b, no_b)
    delta = with_b.astype(np.float32) - no_b.astype(np.float32)
    assert np.abs(delta - g["bias"].astype(np.float32)).max() < 1e-3


def test_end_to_end_scales_are_actually_applied(dense_linear_golden):
    """Likewise for s_in/s_out: they are ~1.0 +- 0.02, so ignoring them leaves a
    plausible-looking output that no smoke test would catch."""
    ref = _ref()
    g = dense_linear_golden
    folded = ref.dense_linear(
        g["x"], g["code"], *ref.fold_scales(g["rin"], g["rout"], g["s_in"], g["s_out"]),
        g["K"], bias=g["bias"])
    dropped = ref.dense_linear(
        g["x"], g["code"], *ref.fold_scales(g["rin"], g["rout"], None, None),
        g["K"], bias=g["bias"])
    want = g["deploy"].astype(np.float32)
    rel_folded = np.abs(folded.astype(np.float32) - want).mean() / np.abs(want).mean()
    rel_dropped = np.abs(dropped.astype(np.float32) - want).mean() / np.abs(want).mean()
    assert rel_folded < REL_TOL
    assert rel_dropped > 10 * rel_folded, (
        "dropping s_in/s_out must be visibly wrong against the deploy reference; "
        f"folded {rel_folded:.2e} vs dropped {rel_dropped:.2e}")


# ------------------------------------------------------------ module wiring


@needs_mlx
def test_module_matches_reference(dense_linear_golden, monkeypatch):
    """EschaLinear's portable path must be bit-exact against ref.dense_linear."""
    monkeypatch.setenv("ESCHA_MLX_LINEAR", "ops")
    import mlx.core as mx
    from escha_mlx import dense

    ref = _ref()
    g = dense_linear_golden
    lin = dense.build({"escha_code": g["code"], "escha_rin": g["rin"],
                       "escha_rout": g["rout"], "escha_s_in": g["s_in"],
                       "escha_s_out": g["s_out"], "bias": g["bias"]})
    assert lin.K == g["K"] and lin._w.IC == 128 and lin._w.OC == 128
    got = np.array(lin(mx.array(g["x"])).astype(mx.float32))
    rin, rout = ref.fold_scales(g["rin"], g["rout"], g["s_in"], g["s_out"])
    want = ref.dense_linear(g["x"], g["code"], rin, rout, g["K"], bias=g["bias"])
    assert np.array_equal(got.astype(np.float16).view(np.uint16),
                          want.view(np.uint16))


@needs_mlx
def test_module_preserves_leading_dimensions(dense_linear_golden, monkeypatch):
    monkeypatch.setenv("ESCHA_MLX_LINEAR", "ops")
    import mlx.core as mx
    from escha_mlx import dense

    g = dense_linear_golden
    lin = dense.build({"escha_code": g["code"], "escha_rin": g["rin"],
                       "escha_rout": g["rout"]})
    x = mx.array(g["x"]).reshape(2, 4, 128)
    y = lin(x)
    assert y.shape == (2, 4, 128) and y.dtype == mx.float16
    flat = lin(mx.array(g["x"]))
    assert np.array_equal(np.array(y.reshape(8, 128)), np.array(flat))


@needs_mlx
def test_prepacked_code_is_equivalent(dense_linear_golden, monkeypatch):
    """The streaming loader packs each code stream on arrival; that must build
    the same weight as handing the raw int16 tensor over."""
    monkeypatch.setenv("ESCHA_MLX_LINEAR", "ops")
    import mlx.core as mx
    from escha_mlx import dense

    g = dense_linear_golden
    common = {"escha_rin": g["rin"], "escha_rout": g["rout"],
              "escha_s_in": g["s_in"], "escha_s_out": g["s_out"], "bias": g["bias"]}
    raw = dense.build({"escha_code": g["code"], **common})
    packed = dense.build({"escha_code": dense.pack_code(g["code"]), **common})
    assert raw.K == packed.K
    x = mx.array(g["x"])
    assert np.array_equal(np.array(raw(x)), np.array(packed(x)))


# ------------------------------------------------------------ metadata guards


@needs_mlx
@pytest.mark.parametrize("field,bad", [(1, 7), (4, 999), (5, 999)])
def test_config_disagreement_is_rejected(dense_linear_golden, field, bad):
    """A K or a shape that disagrees with the stream means the metadata and the
    codes came from different exports — which decodes to plausible noise."""
    from escha_mlx import dense

    g = dense_linear_golden
    cfg = np.array([16, g["K"], 2, 1, 128, 128], dtype=np.int32)
    cfg[field] = bad
    with pytest.raises(ValueError, match="disagrees"):
        dense.EschaWeight(g["code"], g["rin"], g["rout"], config=cfg)


@needs_mlx
def test_unknown_codebook_is_rejected(dense_linear_golden):
    from escha_mlx import dense

    g = dense_linear_golden
    cfg = np.array([16, g["K"], 2, 0, 128, 128], dtype=np.int32)
    with pytest.raises(ValueError, match="codebook"):
        dense.EschaWeight(g["code"], g["rin"], g["rout"], config=cfg)


@needs_mlx
def test_matching_config_is_accepted(dense_linear_golden):
    from escha_mlx import dense

    g = dense_linear_golden
    cfg = np.array([16, g["K"], 2, 1, 128, 128], dtype=np.int32)
    w = dense.EschaWeight(g["code"], g["rin"], g["rout"], config=cfg)
    assert (w.K, w.IC, w.OC) == (g["K"], 128, 128)


@needs_mlx
def test_transform_vector_length_mismatch_is_rejected(dense_linear_golden):
    from escha_mlx import dense

    g = dense_linear_golden
    with pytest.raises(ValueError, match="do not match"):
        dense.EschaWeight(g["code"], g["rin"][:64], g["rout"])


@needs_mlx
@pytest.mark.parametrize("tk,tn", [(9, 8), (8, 9)])
def test_non_hadamard_dimensions_are_rejected(dense_linear_golden, tk, tn):
    """16-aligned but not 128-aligned: the kernels would silently drop the
    remainder of the last block rather than fail."""
    from escha_mlx import dense

    g = dense_linear_golden
    K = g["K"]
    code = np.zeros((tk, tn, 16 * K), dtype=np.int16)
    with pytest.raises(ValueError, match="multiples of 128"):
        dense.EschaWeight(code, np.ones(tk * 16, np.float16), np.ones(tn * 16, np.float16))


@needs_mlx
def test_incomplete_group_is_rejected(dense_linear_golden):
    from escha_mlx import dense

    g = dense_linear_golden
    assert dense.REQUIRED <= dense.LEAVES
    with pytest.raises(KeyError):
        dense.build({"escha_code": g["code"], "escha_rin": g["rin"]})


@needs_mlx
def test_linear_mode_rejects_nonsense(monkeypatch):
    from escha_mlx import dense

    monkeypatch.setenv("ESCHA_MLX_LINEAR", "turbo")
    with pytest.raises(ValueError, match="ESCHA_MLX_LINEAR"):
        dense.linear_mode()


# --------------------------------------------------------------- kernel parity


# A deliberately RECTANGULAR, MULTI-BLOCK shape. 128x128 -- the golden corner --
# is the one geometry at which every dense-vs-MoE address expression coincides
# by construction (see test_dense_kernels_match_expert_kernels), and at which
# blk, ocb and any TK/TN confusion are all invisible because they are zero or
# equal. IC != OC and both > 128 make each of those observable.
BIG_IC, BIG_OC = 256, 384


def _synth(K, rng, ic=BIG_IC, oc=BIG_OC):
    """A random coded linear at a rectangular multi-block shape.

    Random codes are legitimate input: the trellis is tail-biting, so every bit
    string decodes to a valid weight (there is no invalid code to construct).
    """
    return {
        "escha_code": rng.integers(-32768, 32768, size=(ic // 16, oc // 16, 16 * K),
                                   dtype=np.int16),
        "escha_rin": (rng.standard_normal(ic) * 0.05).astype(np.float16),
        "escha_rout": (rng.standard_normal(oc) * 0.05).astype(np.float16),
        "escha_s_in": (1.0 + rng.standard_normal(ic) * 0.02).astype(np.float32),
        "escha_s_out": (1.0 + rng.standard_normal(oc) * 0.02).astype(np.float32),
        "bias": (rng.standard_normal(oc) * 0.01).astype(np.float16),
    }


@needs_mlx
@pytest.mark.parametrize("K", [2, 3])
def test_module_matches_reference_rectangular(K, monkeypatch):
    """The module against the reference at IC != OC, both multi-block.

    Runs on the portable path, so it gates the address arithmetic that the
    128x128 golden cannot reach even where Metal is unavailable.
    """
    monkeypatch.setenv("ESCHA_MLX_LINEAR", "ops")
    import mlx.core as mx
    from escha_mlx import dense

    ref = _ref()
    rng = np.random.default_rng(100 + K)
    g = _synth(K, rng)
    lin = dense.build(g)
    assert (lin._w.IC, lin._w.OC, lin.K) == (BIG_IC, BIG_OC, K)
    x = (rng.standard_normal((5, BIG_IC)) * 0.4).astype(np.float16)
    got = np.array(lin(mx.array(x)))
    rin, rout = ref.fold_scales(g["escha_rin"], g["escha_rout"],
                                g["escha_s_in"], g["escha_s_out"])
    want = ref.dense_linear(x, g["escha_code"], rin, rout, K, bias=g["bias"])
    assert np.array_equal(got.view(np.uint16), want.view(np.uint16))


@needs_metal
@pytest.mark.parametrize("K", [2, 3])
def test_dense_kernels_match_expert_kernels(K):
    """The dense kernels are a compile-time variant of the expert kernels: same
    decode, same accumulation order, minus the row_expert indirection. They must
    therefore be BIT-identical to the expert kernels -- which is what lets the
    dense path inherit the expert kernels' accumulated gate history.

    For that inheritance to mean anything the reference must be NON-DEGENERATE.
    Run against E=1 with row_expert=0 the MoE prologue computes
    `base = code + 0*(TK*TN*wpt)` and `roff = 0*dim + blk*128 + tid`, which are
    arithmetically identical to the dense `base = code` and
    `roff = blk*128 + tid` -- so the comparison would prove only that both
    sources compile. Here the real stream sits at expert index 1 behind a
    garbage expert 0 and row_expert is all ones, so both `e*` terms are
    non-zero, and the shape is rectangular and multi-block so blk, ocb and any
    TK/TN transposition are live.
    """
    import mlx.core as mx
    from escha_mlx import msl

    rng = np.random.default_rng(200 + K)
    ic, oc = BIG_IC, BIG_OC
    tk, tn = ic // 16, oc // 16
    code_i16 = rng.integers(-32768, 32768, size=(tk, tn, 16 * K), dtype=np.int16)
    garbage = rng.integers(-32768, 32768, size=(tk, tn, 16 * K), dtype=np.int16)
    code = mx.array(msl.code_to_u32(code_i16))                       # [TK,TN,8K]
    stacked = mx.array(msl.code_to_u32(np.stack([garbage, code_i16])))  # [2,...]

    m = 6
    xh = mx.array((rng.standard_normal((m, ic)) * 0.3).astype(np.float16))
    rin = (rng.standard_normal(ic) * 0.1).astype(np.float32)
    rout = (rng.standard_normal(oc) * 0.1).astype(np.float32)
    junk_in = (rng.standard_normal(ic) * 5.0).astype(np.float32)
    junk_out = (rng.standard_normal(oc) * 5.0).astype(np.float32)
    ones = mx.ones((m,), dtype=mx.int32)

    d_mid = msl.dense_gemv(xh, code, K, ic, oc)
    m_mid = msl.moe_gemv(xh, stacked, ones, K, ic, oc)
    assert np.array_equal(np.array(d_mid), np.array(m_mid))

    d_in = msl.dense_scaled_had(xh, mx.array(rin), msl.ref.RS)
    m_in = msl.scaled_had(xh, mx.array(np.stack([junk_in, rin])), ones, msl.ref.RS)
    assert np.array_equal(np.array(d_in), np.array(m_in))

    d_out = msl.dense_scaled_had_out(d_mid, mx.array(rout), msl.ref.RS)
    m_out = msl.scaled_had_out(m_mid, mx.array(np.stack([junk_out, rout])), ones,
                               msl.ref.RS)
    assert np.array_equal(np.array(d_out), np.array(m_out))


@needs_metal
@pytest.mark.parametrize("K", [2, 3])
def test_dense_gemv_against_decoded_weight(K):
    """An oracle check independent of the MoE kernels entirely.

    dense_gemv must equal `xh @ decode_tiles(code)`. decode_tiles is gated
    against the committed codec goldens, so this ties the dense GEMV's
    addressing to the format rather than to a sibling kernel -- it catches a
    TK/TN swap or an IC/OC confusion that a dense-vs-MoE comparison could
    never see, because both kernels share that text.
    """
    import mlx.core as mx
    from escha_mlx import msl

    rng = np.random.default_rng(300 + K)
    ic, oc = BIG_IC, BIG_OC
    code_i16 = rng.integers(-32768, 32768, size=(ic // 16, oc // 16, 16 * K),
                            dtype=np.int16)
    code = mx.array(msl.code_to_u32(code_i16))
    xh = (rng.standard_normal((4, ic)) * 0.3).astype(np.float16)
    w = np.array(msl.decode_tiles(code, K, ic, oc))          # [IC, OC] f16
    want = xh.astype(np.float32) @ w.astype(np.float32)
    got = np.array(msl.dense_gemv(mx.array(xh), code, K, ic, oc))
    assert got.shape == (4, oc)
    # f32 reduction order differs from numpy's; the bar is rounding, and a
    # transposition or stride bug is orders of magnitude outside it.
    assert np.abs(got - want).max() < 1e-2 * max(1.0, np.abs(want).max())


@needs_metal
def test_fused_module_matches_reference(dense_linear_golden):
    """The whole fused linear against the portable reference."""
    import mlx.core as mx
    from escha_mlx import dense

    ref = _ref()
    g = dense_linear_golden
    lin = dense.build({"escha_code": g["code"], "escha_rin": g["rin"],
                       "escha_rout": g["rout"], "escha_s_in": g["s_in"],
                       "escha_s_out": g["s_out"], "bias": g["bias"]})
    got = np.array(lin(mx.array(g["x"])).astype(mx.float32))
    rin, rout = ref.fold_scales(g["rin"], g["rout"], g["s_in"], g["s_out"])
    want = ref.dense_linear(g["x"], g["code"], rin, rout, g["K"],
                            bias=g["bias"]).astype(np.float32)
    assert np.abs(got - want).max() < 1e-2
    deploy = g["deploy"].astype(np.float32)
    rel = np.abs(got - deploy).mean() / np.abs(deploy).mean()
    assert rel < REL_TOL


@needs_metal
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("m,R", [(1, 2), (2, 2), (5, 2), (8, 4), (9, 4),
                                 (16, 8), (17, 8), (64, 8), (63, 8)])
def test_row_blocked_gemm_matches_per_row_gemv(K, m, R):
    """The row-blocked dense GEMM changes the loop nest, not the accumulation
    order, so it must be BIT-identical to the per-row kernel.

    The odd row counts are the point: they leave the tail group partly padding,
    which is where the store guard and the staging clamp live. m=1 with R=2
    means the FIRST group is already partly padding. Rectangular multi-block so
    the group's row addressing is observable.
    """
    import mlx.core as mx
    from escha_mlx import msl

    rng = np.random.default_rng(400 + K + m)
    ic, oc = BIG_IC, BIG_OC
    code = mx.array(msl.code_to_u32(
        rng.integers(-32768, 32768, size=(ic // 16, oc // 16, 16 * K), dtype=np.int16)))
    xh = mx.array((rng.standard_normal((m, ic)) * 0.3).astype(np.float16))
    want = msl.dense_gemv(xh, code, K, ic, oc)
    got = msl.dense_gemm_rows(xh, code, K, ic, oc, R)
    assert got.shape == (m, oc)
    assert np.array_equal(np.array(got), np.array(want))


@needs_metal
def test_row_blocked_gemm_does_not_touch_rows_outside_the_batch():
    """The dense GEMM has no padding sink row: the tail group's padding slots
    stage a real row and are dropped by a store guard. If that guard were wrong
    they would write over a live row instead."""
    import mlx.core as mx
    from escha_mlx import msl

    rng = np.random.default_rng(500)
    ic, oc, K = BIG_IC, BIG_OC, 2
    code = mx.array(msl.code_to_u32(
        rng.integers(-32768, 32768, size=(ic // 16, oc // 16, 16 * K), dtype=np.int16)))
    xh = mx.array((rng.standard_normal((7, ic)) * 0.3).astype(np.float16))
    full = np.array(msl.dense_gemm_rows(xh, code, K, ic, oc, 4))
    # every row must equal the per-row kernel's answer for THAT row alone
    for i in range(7):
        one = np.array(msl.dense_gemv(xh[i:i + 1], code, K, ic, oc))
        assert np.array_equal(full[i:i + 1], one), f"row {i} corrupted"


@needs_metal
def test_blocked_path_is_reachable_from_the_module(dense_linear_golden):
    """The policy must actually route a prefill-sized batch to the blocked
    kernel, and the module's output must not depend on which one ran."""
    import mlx.core as mx
    from escha_mlx import dense, msl

    assert msl.dense_block_r(1) == 1
    assert msl.dense_block_r(256) > 1

    g = dense_linear_golden
    lin = dense.build({"escha_code": g["code"], "escha_rin": g["rin"],
                       "escha_rout": g["rout"], "escha_s_in": g["s_in"],
                       "escha_s_out": g["s_out"], "bias": g["bias"]})
    rng = np.random.default_rng(12)
    x = mx.array((rng.standard_normal((96, 128)) * 0.5).astype(np.float16))
    many = np.array(lin(x))
    one_at_a_time = np.concatenate(
        [np.array(lin(x[i:i + 1])) for i in range(x.shape[0])], axis=0)
    assert np.array_equal(many, one_at_a_time)


@needs_mlx
def test_dense_block_r_env_override(monkeypatch):
    from escha_mlx import msl

    monkeypatch.delenv("ESCHA_MLX_DENSE_BLOCK_R", raising=False)
    assert msl.dense_block_r_pin() is None
    monkeypatch.setenv("ESCHA_MLX_DENSE_BLOCK_R", "1")
    assert msl.dense_block_r(4096, msl.dense_block_r_pin()) == 1
    monkeypatch.setenv("ESCHA_MLX_DENSE_BLOCK_R", "16")
    assert msl.dense_block_r(1, msl.dense_block_r_pin()) == 16
    monkeypatch.setenv("ESCHA_MLX_DENSE_BLOCK_R", "0")
    with pytest.raises(ValueError, match="ESCHA_MLX_DENSE_BLOCK_R"):
        msl.dense_block_r_pin()


@needs_mlx
def test_dense_block_r_is_latched_at_construction(dense_linear_golden, monkeypatch):
    """The pin is resolved once, so flipping the env mid-process cannot change
    an already-built module — the invariant every other knob here holds to."""
    from escha_mlx import dense, msl

    monkeypatch.setenv("ESCHA_MLX_LINEAR", "ops")
    monkeypatch.delenv("ESCHA_MLX_DENSE_BLOCK_R", raising=False)
    g = dense_linear_golden
    lin = dense.build({"escha_code": g["code"], "escha_rin": g["rin"],
                       "escha_rout": g["rout"]})
    before = lin._block_r_pin
    monkeypatch.setenv("ESCHA_MLX_DENSE_BLOCK_R", "9")
    assert lin._block_r_pin == before
    assert msl.dense_block_r_pin() == 9      # a NEW module would pick it up


@needs_mlx
def test_dense_block_r_never_pads_and_is_bounded():
    """R <= m always (a dense group issuing MACs for rows that do not exist is
    pure waste), R is a power of two (each value is a separate compilation), and
    R > 1 from m = 2 (a ladder that stayed at 1 for small batches would read the
    whole coded stream once per row at exactly the concurrency a server runs)."""
    from escha_mlx import msl

    rs = {m: msl.dense_block_r(m) for m in range(1, 600)}
    assert all(r <= m for m, r in rs.items())
    assert set(rs.values()) <= {1, 2, 4, 8}
    assert rs[1] == 1 and rs[2] == 2 and rs[4] == 4 and rs[8] == 8
    assert all(rs[m] == msl.DENSE_R_MAX for m in range(msl.DENSE_R_MAX, 600))
    assert all(rs[m] <= rs[m + 1] for m in range(1, 599))    # monotone


@needs_mlx
def test_coded_bytes_sees_streams_that_parameters_cannot(dense_linear_golden, monkeypatch):
    """The coded stream is deliberately off the parameter tree, so any byte
    ledger that walks parameters() alone values a coded linear at its bias —
    zero when there is none. bench/roofline.py depends on this helper."""
    monkeypatch.setenv("ESCHA_MLX_LINEAR", "ops")
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from escha_mlx import dense

    g = dense_linear_golden
    lin = dense.build({"escha_code": g["code"], "escha_rin": g["rin"],
                       "escha_rout": g["rout"]})          # no bias
    assert [k for k, _ in tree_flatten(lin.parameters())] == []

    class Holder(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a, self.stack = a, [b, b]

    want = sum(a.nbytes for a in lin._w.arrays())
    assert dense.coded_bytes(lin) == want
    # nested, and each distinct module counted once even when aliased in a list
    other = dense.build({"escha_code": g["code"], "escha_rin": g["rin"],
                         "escha_rout": g["rout"]})
    assert dense.coded_bytes(Holder(lin, other)) == 2 * want
    assert dense.coded_bytes([lin, other]) == 2 * want
    assert dense.coded_bytes(nn.Linear(4, 4)) == 0
