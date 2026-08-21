"""Does M4 have a faster fp16-ACCUMULATE matrix path, and is it worth the error?

Three inner products over identical decoded weights, interleaved A/B/C/A to
cancel thermal drift:

  scalar   -- today's bit-exact row-blocked GEMM (R=16)
  mat-f32  -- simdgroup MMA, float8x8 accumulator.  On M4 the half x half
              product is EXACT (22 mantissa bits into f32's 24), so this is a
              reassociation of an f32 sum, not a precision downgrade.
  mat-f16  -- simdgroup MMA, half8x8 accumulator: rounds the running sum to
              fp16, and the error grows with reduction length.

On GPUs whose matrix units run fp16-accumulate at roughly twice the
fp32-accumulate rate, mat-f16 is the variant worth its error budget.  If it is
not meaningfully faster than mat-f32 here, then this GPU has no such bonus and
the risky variant buys nothing -- which settles whether the precision question
is even worth asking on M4.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from escha_mlx import msl  # noqa: E402


def build_acc(K: int, half_acc: bool):
    wpt = 8 * K
    extract = msl._substitute_fetch(msl._EXTRACT_K2 if K == 2 else msl._EXTRACT_K3)
    stores = []
    for j in range(8):
        fi = j >> 1
        row = f"(lane & 3u) * 2u + {j & 1}u + {(fi & 1) * 8}u"
        col = f"2u * ((l0 >> 3) + {4 if j >= 4 else 0}u) + c_off"
        stores.append(f"        s_w[sg * 256u + ({row}) * 16u + ({col})] = cba_decode(s{j});")
    ct = "simdgroup_half8x8" if half_acc else "simdgroup_float8x8"
    zero = "simdgroup_half8x8(0.0h)" if half_acc else "simdgroup_float8x8(0.0f)"
    ot = "half" if half_acc else "float"
    src = f"""
    uint tid  = thread_position_in_threadgroup.x;
    uint ocb  = thread_position_in_grid.x >> 8;
    uint grp  = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;
    if (grp * 16u >= (uint)M) return;

    threadgroup half s_x[256];
    threadgroup half s_w[2048];

    {ct} C00 = {zero}, C01 = {zero}, C10 = {zero}, C11 = {zero};
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xr = tid >> 4, xc = tid & 15u;
    uint srow = min(grp * 16u + xr, (uint)M - 1u);

    for (uint kt = 0; kt < TK; ++kt) {{
        s_x[tid] = xh[(ulong)srow * IC + kt * 16u + xc];
        device const uint* wp = code + ((ulong)kt * TN + ocb * 8u + sg) * {wpt}u;
{extract}
{chr(10).join(stores)}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 A00, A01, A10, A11, B00, B01, B10, B11;
        simdgroup_load(A00, s_x, 16);          simdgroup_load(A01, s_x + 8u, 16);
        simdgroup_load(A10, s_x + 128u, 16);   simdgroup_load(A11, s_x + 136u, 16);
        threadgroup const half* wb = s_w + sg * 256u;
        simdgroup_load(B00, wb, 16);           simdgroup_load(B01, wb + 8u, 16);
        simdgroup_load(B10, wb + 128u, 16);    simdgroup_load(B11, wb + 136u, 16);
        simdgroup_multiply_accumulate(C00, A00, B00, C00);
        simdgroup_multiply_accumulate(C00, A01, B10, C00);
        simdgroup_multiply_accumulate(C01, A00, B01, C01);
        simdgroup_multiply_accumulate(C01, A01, B11, C01);
        simdgroup_multiply_accumulate(C10, A10, B00, C10);
        simdgroup_multiply_accumulate(C10, A11, B10, C10);
        simdgroup_multiply_accumulate(C11, A10, B01, C11);
        simdgroup_multiply_accumulate(C11, A11, B11, C11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    device {ot}* o = mid + (ulong)(grp * 16u) * OC + ocb * 128u + sg * 16u;
    simdgroup_store(C00, o, OC);            simdgroup_store(C01, o + 8u, OC);
    simdgroup_store(C10, o + 8u * OC, OC);  simdgroup_store(C11, o + 8u * OC + 8u, OC);
"""
    return mx.fast.metal_kernel(
        name=f"probe_sgmat_{'f16' if half_acc else 'f32'}_k{K}",
        input_names=["xh", "code"], output_names=["mid"],
        header=msl._SGMAT_HEADER, source=src)


def run(kern, xh, cu, K, IC, OC, half_acc):
    m = xh.shape[0]
    (mid,) = kern(inputs=[xh, cu.reshape(-1)],
                  template=[("TK", IC // 16), ("TN", OC // 16), ("IC", IC),
                            ("OC", OC), ("M", m)],
                  grid=(256 * (OC // 128), (m + 15) // 16, 1),
                  threadgroup=(256, 1, 1), output_shapes=[(m, OC)],
                  output_dtypes=[mx.float16 if half_acc else mx.float32])
    return mid


def timeit(fn, iters=8, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rng = np.random.default_rng(0)
    shapes = [("mlp.gate K2", 5120, 17408, 2), ("mlp.down K3", 17408, 5120, 3),
              ("gdn.qkv  K2", 5120, 10240, 2)]
    print(f"m={M}: scalar vs matrix-f32acc vs matrix-f16acc "
          f"(medians of 3 interleaved passes)\n")
    print(f"{'shape':13s} {'scalar ms':>10s} {'matF32 ms':>10s} {'matF16 ms':>10s} "
          f"{'f32 gain':>9s} {'f16 gain':>9s} {'f32 dev':>9s} {'f16 dev':>9s}")
    for name, IC, OC, K in shapes:
        code = rng.integers(-32768, 32767, size=(IC // 16, OC // 16, 16 * K), dtype=np.int16)
        cu = mx.array(msl.code_to_u32(code))
        xh = mx.array((rng.standard_normal((M, IC)) * 0.3).astype(np.float16))
        mx.eval(cu, xh)
        kf32, kf16 = build_acc(K, False), build_acc(K, True)
        fs = lambda: msl.dense_gemm_rows(xh, cu, K, IC, OC, 16)          # noqa: E731
        f32 = lambda: run(kf32, xh, cu, K, IC, OC, False)                # noqa: E731
        f16 = lambda: run(kf16, xh, cu, K, IC, OC, True)                 # noqa: E731
        ts, t32, t16 = [], [], []
        for _ in range(3):                       # interleave to cancel drift
            ts.append(timeit(fs)); t32.append(timeit(f32)); t16.append(timeit(f16))
        ts, t32, t16 = statistics.median(ts), statistics.median(t32), statistics.median(t16)
        ref = fs(); a32 = f32(); a16 = f16()
        mx.eval(ref, a32, a16)
        scale = float(mx.abs(ref).mean())
        d32 = float(mx.abs(ref - a32).mean()) / scale
        d16 = float(mx.abs(ref - a16.astype(mx.float32)).mean()) / scale
        print(f"{name:13s} {ts*1000:10.2f} {t32*1000:10.2f} {t16*1000:10.2f} "
              f"{ts/t32:8.2f}x {ts/t16:8.2f}x {d32:9.1e} {d16:9.1e}")


if __name__ == "__main__":
    main()
