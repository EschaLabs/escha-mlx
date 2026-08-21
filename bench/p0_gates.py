"""P0 derisk gates for escha on Apple Silicon — run FIRST on a new machine.

Run on the target Mac:
    python bench/p0_gates.py [--model <checkpoint_dir>]

  G0.1  decode bit-exactness (hash + LUT) vs the committed goldens
  G0.2  fused GEMV: value check vs reference + DRAM-side GB/s microbench
  G0.3  Q8 repack round-trip
  G0.4  memory envelope report

Every gate is exception-isolated: a Metal compile error in one kernel still
lets the rest of the report print (capture the WHOLE output either way).
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE.parent / "tests" / "data" / "codec"
sys.path.insert(0, str(HERE.parent))

RESULTS: list[bool] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    RESULTS.append(ok)


def section(fn):
    def run(*a, **kw):
        try:
            fn(*a, **kw)
        except Exception as e:
            gate(fn.__name__, False, f"EXCEPTION: {e!r}")
            traceback.print_exc()
    return run


def _goldens():
    p2 = np.fromfile(DATA / "packed_gu_e0_k2.i16", dtype=np.int16).reshape(128, 64, 32)
    e2 = np.fromfile(DATA / "expected_gu_e0_k2.f16", dtype=np.float16).reshape(2048, 1024)
    p3 = np.fromfile(DATA / "packed_down_e0_k3.i16", dtype=np.int16).reshape(32, 128, 48)
    e3 = np.fromfile(DATA / "expected_down_e0_k3.f16", dtype=np.float16).reshape(512, 2048)
    return p2, e2, p3, e3


@section
def g01_decode():
    import mlx.core as mx
    from escha_mlx import msl
    print("\nG0.1 decode bit-exactness")
    p2, e2, p3, e3 = _goldens()
    for lut in (False, True):
        os.environ["ESCHA_MLX_LUT"] = "1" if lut else "0"
        msl._lut_array.cache_clear()
        for K, packed, expected in ((2, p2, e2), (3, p3, e3)):
            try:
                got = np.array(msl.decode_tiles(
                    mx.array(msl.code_to_u32(packed)), K, *expected.shape))
                ok = np.array_equal(got.view(np.uint16), expected.view(np.uint16))
                gate(f"K{K} decode ({'LUT' if lut else 'hash'})", ok)
            except Exception as e:
                gate(f"K{K} decode ({'LUT' if lut else 'hash'})", False, repr(e)[:120])
    os.environ["ESCHA_MLX_LUT"] = "0"
    msl._lut_array.cache_clear()


@section
def g02_gemv():
    import mlx.core as mx
    from escha_mlx import msl, ref
    print("\nG0.2 fused GEMV (value gate + DRAM-side microbench)")
    p2, e2, p3, e3 = _goldens()
    rng = np.random.default_rng(0)
    for K, packed, expected in ((2, p2, e2), (3, p3, e3)):
        ic, oc = expected.shape
        E = 64
        flat = packed.reshape(-1, packed.shape[-1])
        codes = np.stack([packed] + [
            flat[rng.permutation(len(flat))].reshape(packed.shape) for _ in range(E - 1)])
        code_mx = mx.array(codes.reshape(E, ic // 16, oc // 16, -1)
                           .view(np.uint16).view(np.uint32))
        m = 8
        xh = mx.array((rng.standard_normal((m, ic)) * 0.05).astype(np.float16))

        # value gate: expert 0 (the golden) + expert E-1 (stride check)
        re_val = np.array([0, E - 1] * 4, dtype=np.int32)
        mid = np.array(msl.moe_gemv(xh, code_mx, mx.array(re_val), K, ic, oc))
        w_last = ref.reconstruct_fast(codes[E - 1], ic, oc, K).astype(np.float32)
        xh_np = np.array(xh)
        ok = True
        for r in range(m):
            w_r = expected.astype(np.float32) if re_val[r] == 0 else w_last
            want = xh_np[r].astype(np.float32) @ w_r
            if np.abs(mid[r] - want).max() > 2e-3 * max(np.abs(want).max(), 1e-6):
                ok = False
        gate(f"K{K} gemv values (expert 0 + {E-1})", ok)

        # DRAM-side timing: fresh experts each call so streams miss the SLC
        res = [mx.array(rng.integers(0, E, m).astype(np.int32)) for _ in range(50)]
        mx.eval(msl.moe_gemv(xh, code_mx, res[0], K, ic, oc))
        t0 = time.perf_counter()
        outs = [msl.moe_gemv(xh, code_mx, re, K, ic, oc) for re in res]
        mx.eval(outs)
        dt = (time.perf_counter() - t0) / len(res)
        gbps = m * (ic * oc * K / 8) / dt / 1e9
        gate(f"K{K} gemv perf", True,
             f"{dt*1e6:.0f} us/call, ~{gbps:.0f} GB/s trellis stream (DRAM-side)")
    print("  (target: >=~35% of chip peak BW on M4-class; <15% = investigate)")


@section
def g02b_dense():
    """Dense (single-stream) kernels: values, expert-parity, row-blocking.

    The dense kernels are a compile-time variant of the expert kernels above,
    so this gate is deliberately NON-DEGENERATE: the parity check puts the real
    stream at expert index 1 behind a garbage expert 0, at a rectangular
    multi-block shape.  Run against E=1 with row_expert=0 the two kernels
    compute identical addresses by construction and the comparison would prove
    nothing at all.
    """
    import mlx.core as mx
    from escha_mlx import msl
    print("\nG0.2b dense GEMV / row-blocked GEMM (single-stream variant)")
    rng = np.random.default_rng(1)
    ic, oc = 256, 384                      # IC != OC, both multi-block
    for K in (2, 3):
        real = rng.integers(-32768, 32768, (ic // 16, oc // 16, 16 * K), dtype=np.int16)
        junk = rng.integers(-32768, 32768, (ic // 16, oc // 16, 16 * K), dtype=np.int16)
        code = mx.array(msl.code_to_u32(real))
        stacked = mx.array(msl.code_to_u32(np.stack([junk, real])))
        m = 6
        xh = mx.array((rng.standard_normal((m, ic)) * 0.3).astype(np.float16))

        # 1. against the decoded weight — ties addressing to the FORMAT, not to
        #    a sibling kernel that shares the same text.
        w = np.array(msl.decode_tiles(code, K, ic, oc)).astype(np.float32)
        want = np.array(xh).astype(np.float32) @ w
        got = np.array(msl.dense_gemv(xh, code, K, ic, oc))
        gate(f"K{K} dense gemv vs decoded weight",
             np.abs(got - want).max() < 1e-2 * max(np.abs(want).max(), 1e-6))

        # 2. bit-identical to the expert kernel at a NON-ZERO expert index
        ones = mx.ones((m,), dtype=mx.int32)
        gate(f"K{K} dense == expert kernel (expert 1)",
             np.array_equal(np.array(msl.dense_gemv(xh, code, K, ic, oc)),
                            np.array(msl.moe_gemv(xh, stacked, ones, K, ic, oc))))
        rin = (rng.standard_normal(ic) * 0.1).astype(np.float32)
        rout = (rng.standard_normal(oc) * 0.1).astype(np.float32)
        junk_in = (rng.standard_normal(ic) * 5).astype(np.float32)
        junk_out = (rng.standard_normal(oc) * 5).astype(np.float32)
        gate(f"K{K} dense input transform == expert",
             np.array_equal(
                 np.array(msl.dense_scaled_had(xh, mx.array(rin), msl.ref.RS)),
                 np.array(msl.scaled_had(xh, mx.array(np.stack([junk_in, rin])),
                                         ones, msl.ref.RS))))
        mid = msl.dense_gemv(xh, code, K, ic, oc)
        gate(f"K{K} dense output transform == expert",
             np.array_equal(
                 np.array(msl.dense_scaled_had_out(mid, mx.array(rout), msl.ref.RS)),
                 np.array(msl.scaled_had_out(mid, mx.array(np.stack([junk_out, rout])),
                                             ones, msl.ref.RS))))

        # 3. row-blocking must not change a bit, including a partial tail group
        ok = True
        for rows, R in ((5, 2), (9, 4), (17, 8)):
            x2 = mx.array((rng.standard_normal((rows, ic)) * 0.3).astype(np.float16))
            if not np.array_equal(
                    np.array(msl.dense_gemm_rows(x2, code, K, ic, oc, R)),
                    np.array(msl.dense_gemv(x2, code, K, ic, oc))):
                ok = False
        gate(f"K{K} row-blocked GEMM == per-row (partial tail groups)", ok)


@section
def g03_q8():
    import numpy as np
    import mlx.core as mx
    from escha_mlx import quant
    print("\nG0.3 MLX affine-Q8 repack")
    # Validate EVERY group size the loader can select (fit_group picks the
    # largest that divides K), not just the old 64 default.
    for g in (32, 64, quant.DEFAULT_GROUP):
        quant._VALIDATED.discard(g)
        quant.validate_pack(g)
        gate(f"repack round-trip bit-exact (group {g})", True)

    # The default's whole justification is that group size cannot change a
    # value: the escha scale is per-output-channel, so pack_q8 stores one
    # constant per row. Assert that on-device rather than trusting the algebra.
    rng = np.random.default_rng(0)
    w8 = rng.integers(-128, 128, size=(8, 1024), dtype=np.int8)
    scale = (rng.random(8).astype(np.float32) * 0.05 + 1e-3).astype(np.float16)
    x = mx.array(rng.standard_normal((4, 1024)).astype(np.float16))
    outs = [np.array(quant.make_linear(w8, scale, g)(x)) for g in (64, 128)]
    same = np.array_equal(outs[0].view(np.uint16), outs[1].view(np.uint16))
    gate("group 64 == group 128 bit-identical", same,
         f"default={quant.DEFAULT_GROUP} (-140 MB vs 64)")


@section
def g04_memory(model: str | None):
    print("\nG0.4 memory envelope")
    try:
        import mlx.core as mx
        ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()) / 1e9
        # iogpu.wired_limit_mb is 0 unless explicitly overridden, so the old
        # `ram * 2/3` fallback under-reported this box by 1.9 GB (M4 24 GB gives
        # 3/4, not 2/3).  Metal's own figure is authoritative -- ask it.
        wl = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True,
                            text=True, timeout=5).stdout.strip()
        override = int(wl) / 1e3 if wl and int(wl) else None
        # Metal's figure is what set_wired_limit is actually checked against, so
        # it is the authoritative cap; the sysctl value is only what was asked
        # for (they differ -- 21000 MB requested reads back as a 22.02 GB cap).
        metal = mx.device_info().get("max_recommended_working_set_size", 0) / 1e9
        print(f"  RAM {ram:.1f} GB, GPU working-set cap {metal:.2f} GB"
              + (f" (iogpu.wired_limit_mb override: {override:.2f} GB requested)"
                 if override else " (no sysctl override)"))
        wired_now = mx.set_wired_limit(0)   # query without changing behaviour
        mx.set_wired_limit(wired_now)       # ... and put it back
        print(f"  MLX wired limit currently {wired_now/1e9:.2f} GB "
              f"(default 0 = nothing wired)")
        if wired_now == 0:
            print(f"  NOTE: a working set within ~2 GB of the cap NEEDS wiring — "
                  f"unwired, B=80 at 19.28 GB measured 5.9 tok/s vs 136.6 wired "
                  f"(23x, silent). Set ESCHA_MLX_WIRED_GB. Below ~18 GB it is a "
                  f"wash. See bring-up doc §10.3.")
        # The first three values were measured by the pre-2026-08-09
        # baseline.py, whose _gb divided by 1024**3; converted here to decimal
        # GB (x1.0737). The B=16/48/64 peaks come from the sweep harnesses
        # (sweep_block_r / sweep_gdn_state, bring-up doc §10.3/§12.4), which
        # always divided by 1e9 — decimal as recorded.
        print(f"  measured on M4 24 GB @ Q8 group 128: 12.25 GB resident, "
              f"12.92 GB peak at short ctx, 13.36 GB at ISL 512, "
              f"13.67 GB at B=16, 16.49 GB at B=48, 17.89 GB at B=64")
    except Exception as e:
        print(f"  (sysctl probe failed: {e!r})")
    if model:
        print(f"  baseline for comparison: stock mlx-lm 4-bit on this box")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    import mlx.core as mx
    print(f"escha-mlx P0 gates — {platform.platform()} — mlx {mx.__version__} "
          f"metal={mx.metal.is_available()}")
    if not mx.metal.is_available():
        print("!! Metal not available — G0.1/G0.2 require an Apple Silicon Mac")

    g01_decode()
    g02_gemv()
    g02b_dense()
    g03_q8()
    g04_memory(args.model)

    print(f"\n{'ALL GATES PASS' if RESULTS and all(RESULTS) else '>>> GATE FAILURES — see above'}")
    sys.exit(0 if RESULTS and all(RESULTS) else 1)


if __name__ == "__main__":
    main()
