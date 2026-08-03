"""Determinism of the routed-expert accumulation.

The MoE tail used to be `mx.zeros((t,H)).at[row_token].add(contrib)` -- an
atomic f32 scatter-add with top_k duplicate indices per token.  Atomics commit
in arrival order, so the per-token sum order varied between runs; compounded
over 40 layers it flipped greedy argmax on near-ties, and decoding the same
prompt twice produced different text.  It is now a fixed-order segmented
reduction (`contrib.reshape(t, top_k, -1).sum(axis=1)`).

These tests pin the three things that fix depends on:
  1. the row layout really is token-major (else reshape sums the wrong groups),
  2. the reduction agrees with the scatter it replaced (same addends),
  3. repeated evaluation is bit-identical (the actual guarantee).

Determinism here is a correctness requirement, not a nicety: without it no
kernel A/B can be settled by comparing outputs, and every eval number carries
an invisible run-to-run term.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_mlx

pytestmark = needs_mlx

TOP_K = 4
HID = 64
T = 5


def _fake_block(monkeypatch=None):
    """An EschaSparseMoeBlock-shaped object exposing only what we test.

    Building a real block needs packed expert streams; the accumulation tail is
    pure MLX and independent of the codec, so we exercise it directly.
    """
    import mlx.core as mx

    class B:
        top_k = TOP_K

        def _rows(self, xf, ids):
            from escha_mlx.moe import EschaSparseMoeBlock
            return EschaSparseMoeBlock._rows(self, xf, ids)

    return B()


def test_rows_layout_is_token_major():
    """reshape(t, top_k, -1) is only valid if rows are grouped by token.

    If `_rows` ever switched to slot-major (tile instead of repeat), the
    segmented sum would silently combine contributions from DIFFERENT tokens --
    wrong output, no error.  This is the invariant that makes it safe.
    """
    import mlx.core as mx

    blk = _fake_block()
    xf = mx.zeros((T, HID))
    ids = mx.array(np.arange(T * TOP_K, dtype=np.int32).reshape(T, TOP_K))
    row_expert, row_token = blk._rows(xf, ids)

    want_token = np.repeat(np.arange(T), TOP_K)
    assert np.array_equal(np.array(row_token), want_token), (
        "row layout is not token-major; the segmented sum in _expert_path is "
        "invalid — see tests/test_moe_determinism.py")
    # row_expert must follow the same order, so contiguous blocks of top_k rows
    # belong to one token.
    assert np.array_equal(np.array(row_expert),
                          np.arange(T * TOP_K, dtype=np.int32))


def test_segmented_sum_matches_scatter_add():
    """Same addends as the scatter it replaced (order differs, hence tolerance)."""
    import mlx.core as mx

    rng = np.random.default_rng(5)
    contrib = mx.array(rng.standard_normal((T * TOP_K, HID)).astype(np.float32))
    row_token = mx.array(np.repeat(np.arange(T), TOP_K).astype(np.int32))

    new = contrib.reshape(T, TOP_K, -1).sum(axis=1)
    old = mx.zeros((T, HID), dtype=mx.float32).at[row_token].add(contrib)
    mx.eval(new, old)

    a, b = np.array(new), np.array(old)
    # f32 addition is not associative, so this is a rounding-level check, not
    # bit-identity. An exact-in-f64 check confirms neither is drifting.
    assert np.abs(a - b).max() <= 1e-5 * max(np.abs(b).max(), 1e-6), (
        np.abs(a - b).max())
    exact = np.array(contrib).astype(np.float64).reshape(T, TOP_K, HID).sum(axis=1)
    assert np.abs(a.astype(np.float64) - exact).max() < 1e-4


def test_segmented_sum_is_bit_reproducible():
    """The reduction must give bit-identical results across evaluations.

    The old scatter-add did not: this same assertion on `.at[].add()` is what
    the run-to-run token divergence reduced to.
    """
    import mlx.core as mx

    rng = np.random.default_rng(6)
    # Values spanning several exponents make any reassociation visible.
    raw = (rng.standard_normal((T * TOP_K, HID))
           * 10.0 ** rng.integers(-4, 4, size=(T * TOP_K, 1))).astype(np.float32)
    contrib = mx.array(raw)

    ref = None
    for _ in range(24):
        y = contrib.reshape(T, TOP_K, -1).sum(axis=1)
        mx.eval(y)
        a = np.array(y)
        if ref is None:
            ref = a
        else:
            assert np.array_equal(a.view(np.uint32), ref.view(np.uint32)), (
                "segmented sum is not bit-reproducible")


@pytest.mark.parametrize("t,top_k", [(1, 8), (3, 8), (17, 8), (64, 2)])
def test_segmented_sum_shapes(t, top_k):
    """Shape-agnostic: the reshape must hold for bs1 through prefill widths."""
    import mlx.core as mx

    rng = np.random.default_rng(t * 100 + top_k)
    contrib = mx.array(rng.standard_normal((t * top_k, HID)).astype(np.float32))
    row_token = mx.array(np.repeat(np.arange(t), top_k).astype(np.int32))
    new = contrib.reshape(t, top_k, -1).sum(axis=1)
    old = mx.zeros((t, HID), dtype=mx.float32).at[row_token].add(contrib)
    mx.eval(new, old)
    assert new.shape == (t, HID)
    assert np.abs(np.array(new) - np.array(old)).max() <= 2e-5 * max(
        np.abs(np.array(old)).max(), 1e-6)


def test_no_float_scatter_add_remains():
    """Guard the whole package: any new f32 `.at[].add()` reintroduces the bug.

    The three surviving scatters in build_groups are int32, where atomic
    ordering cannot change the result (integer addition is associative).  A new
    float one would be a silent determinism regression, so it must be a
    deliberate edit to this test.
    """
    from pathlib import Path

    src_dir = Path(__file__).parent.parent / "escha_mlx"
    hits = []
    for f in sorted(src_dir.glob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]          # prose about the old bug
            if ".at[" in code and ".add(" in code:
                hits.append((f.name, i, line.strip()))
    non_int = [h for h in hits if "int32" not in h[2]]
    assert not non_int, (
        "non-int32 scatter-add found — atomics make f32 accumulation "
        f"run-to-run nondeterministic: {non_int}")
    assert len(hits) == 3, (
        f"scatter-add count changed ({len(hits)}); confirm each is integer "
        f"and update this test: {hits}")
