"""Prototype: simdgroup-matrix dense GEMM vs the scalar row-blocked GEMM.

Measurement only -- NOT wired into the runtime.  Answers two questions with
one kernel: how much faster is the matrix path on M4 base, and how far do its
outputs move from the bit-exact scalar kernel.

The M4 MMA computes half x half products at FULL precision into an f32
accumulator (probed empirically: (1+2^-10)^2 comes back exact, and 2048
accumulations of 2^-12 are exact).  So the deviation measured here is pure
SUMMATION REORDERING of an f32 sum -- the same class as the split-K path this
runtime already ships -- not a precision downgrade.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from escha_mlx import msl  # noqa: E402

HDR = ("#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
       "using namespace metal;\n" + msl._HEADER)


def build(K: int):
    """R=16 row-blocked GEMM whose inner product is 8 simdgroup MMAs per tile.

    Weights are decoded exactly as the scalar kernel decodes them, then staged
    in threadgroup memory so simdgroup_load can form a fragment (the matrix
    units cannot read per-lane registers).  x is staged once per threadgroup
    and shared by all 8 simdgroups.
    """
    wpt = 8 * K
    extract = msl._substitute_fetch(msl._EXTRACT_K2 if K == 2 else msl._EXTRACT_K3)
    stores = []
    for j in range(8):
        fi = j >> 1
        row = f"(lane & 3u) * 2u + {j & 1}u + {(fi & 1) * 8}u"
        col = f"2u * ((l0 >> 3) + {4 if j >= 4 else 0}u) + c_off"
        stores.append(f"        s_w[sg * 256u + ({row}) * 16u + ({col})] = cba_decode(s{j});")
    src = f"""
    uint tid  = thread_position_in_threadgroup.x;
    uint ocb  = thread_position_in_grid.x >> 8;
    uint grp  = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;
    if (grp * 16u >= (uint)M) return;

    threadgroup half s_x[256];
    threadgroup half s_w[2048];

    simdgroup_float8x8 C00 = simdgroup_float8x8(0.0f), C01 = simdgroup_float8x8(0.0f),
                       C10 = simdgroup_float8x8(0.0f), C11 = simdgroup_float8x8(0.0f);
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
    device float* o = mid + (ulong)(grp * 16u) * OC + ocb * 128u + sg * 16u;
    simdgroup_store(C00, o, OC);            simdgroup_store(C01, o + 8u, OC);
    simdgroup_store(C10, o + 8u * OC, OC);  simdgroup_store(C11, o + 8u * OC + 8u, OC);
"""
    return mx.fast.metal_kernel(
        name=f"escha_gemm_dense_sgmat_k{K}", input_names=["xh", "code"],
        output_names=["mid"], header=HDR, source=src)


def run_mat(kern, xh, cu, K, IC, OC):
    m = xh.shape[0]
    (mid,) = kern(inputs=[xh, cu.reshape(-1)],
                  template=[("TK", IC // 16), ("TN", OC // 16), ("IC", IC),
                            ("OC", OC), ("M", m)],
                  grid=(256 * (OC // 128), (m + 15) // 16, 1),
                  threadgroup=(256, 1, 1),
                  output_shapes=[(m, OC)], output_dtypes=[mx.float32])
    return mid


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rng = np.random.default_rng(0)
    # (name, IC, OC, K, layers in the 27B)
    shapes = [("mlp.gate K2", 5120, 17408, 2, 64),
              ("mlp.up   K3", 5120, 17408, 3, 64),
              ("mlp.down K3", 17408, 5120, 3, 64),
              ("gdn.qkv  K2", 5120, 10240, 2, 48),
              ("gdn.z    K2", 5120, 6144, 2, 48),
              ("gdn.out  K2", 6144, 5120, 2, 48),
              ("attn.q   K2", 5120, 12288, 2, 16),
              ("attn.o   K2", 6144, 5120, 2, 16)]
    print(f"m={M} rows, R=16 -- scalar row-blocked GEMM vs simdgroup-matrix\n")
    print(f"{'shape':13s} {'scalar ms':>10s} {'matrix ms':>10s} {'speedup':>8s} "
          f"{'max rel dev':>12s} {'mean rel dev':>13s}")
    tot_s = tot_m = 0.0
    for name, IC, OC, K, nl in shapes:
        code = rng.integers(-32768, 32767, size=(IC // 16, OC // 16, 16 * K), dtype=np.int16)
        cu = mx.array(msl.code_to_u32(code))
        xh = mx.array((rng.standard_normal((M, IC)) * 0.3).astype(np.float16))
        mx.eval(cu, xh)
        kern = build(K)
        a = msl.dense_gemm_rows(xh, cu, K, IC, OC, 16)
        b = run_mat(kern, xh, cu, K, IC, OC)
        mx.eval(a, b)
        scale = float(mx.abs(a).mean())
        rel = mx.abs(a - b) / max(scale, 1e-9)
        mx.eval(rel)
        ts = bench(lambda: msl.dense_gemm_rows(xh, cu, K, IC, OC, 16))
        tm = bench(lambda: run_mat(kern, xh, cu, K, IC, OC))
        tot_s += ts * nl
        tot_m += tm * nl
        print(f"{name:13s} {ts*1000:10.3f} {tm*1000:10.3f} {ts/tm:7.2f}x "
              f"{float(rel.max()):12.2e} {float(rel.mean()):13.2e}")
    print(f"\ncoded-GEMM total per {M}-row chunk: scalar {tot_s*1000:.1f} ms, "
          f"matrix {tot_m*1000:.1f} ms  ->  {tot_s/tot_m:.2f}x on the GEMM alone")
    print("(deviation is relative to mean |output|; pure f32 reassociation -- "
          "the MMA product is exact)")


if __name__ == "__main__":
    main()
