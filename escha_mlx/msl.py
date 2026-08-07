"""Metal kernels (inline MSL via ``mx.fast.metal_kernel``) for the escha codec.

Two kernel families only — everything else in this package is pure MLX ops:

  * ``decode_tiles``  — packed code stream -> fp16 bare weight [IC, OC].
    Correctness anchor + prefill/debug dequant path. One simdgroup per 16x16
    tile; each lane scatter-stores its 8 decoded values (no shuffles).
  * ``moe_gemv``      — fused per-row expert GEMV: xh [M, IC] x decoded W
    -> mid f32 [M, OC].  One 256-thread threadgroup covers 128 output
    channels of one row (8 simdgroups; simdgroup s handles output tile
    ocb*8+s).  Rows are single-token/single-expert ("row_expert" indexed),
    which keeps everything device-resident (no host sync in the MoE path).

Numeric contract (must match escha_mlx.ref bit-for-bit at the decode level):
  decode(x) = fp16_lo(r) + fp16_hi(r), r = ((x*0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60
The fp16 add must round-to-nearest-even. G0.1 (bench/p0_gates.py) verifies
this on-device; if the compiled hash path ever diverges (fast-math), set
ESCHA_MLX_LUT=1 to use the 65,536-entry fp16 table (bit-exact by
construction, built by ref.cba_lut on the host).
"""
from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import ref

_HEADER = """
static inline half cba_decode(uint x) {
    x = x * 0xCBAC1FEDu;
    uint r = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half2 h = as_type<half2>(r);
    return h.x + h.y;
}
"""

# Per-lane window extraction. `W(i)` is substituted with an address-space
# appropriate word fetch. Produces `uint s0..s7` (16-bit states in low bits).
_EXTRACT_K2 = """
    uint t_off = lane * 8u;
    uint i1 = t_off >> 4;
    uint i0 = (i1 + 15u) & 15u;
    ulong merged = (((ulong)W(i0)) << 32) | (ulong)W(i1);
    uint shift = ((~t_off) & 8u) << 1;
    uint wv = (uint)(merged >> shift);   // NOT named `w`: decode_tiles' output
    uint s7 = wv & 0xffffu;              // buffer parameter is `w`, and MSL
    uint s6 = (wv >> 2) & 0xffffu;       // forbids shadowing a parameter in
    uint s5 = (wv >> 4) & 0xffffu;       // the outermost function block
    uint s4 = (wv >> 6) & 0xffffu;
    uint s3 = (wv >> 8) & 0xffffu;
    uint s2 = (wv >> 10) & 0xffffu;
    uint s1 = (wv >> 12) & 0xffffu;
    uint s0 = (wv >> 14) & 0xffffu;
"""

_EXTRACT_K3 = """
    uint t_off = lane * 8u;
    uint b1 = (t_off + 257u) * 3u;
    uint b0 = b1 - 16u;
    uint b2 = b1 + 21u;
    uint i0 = b0 >> 5;
    uint i2 = (b2 - 1u) >> 5;
    uint sh2 = ((i2 + 1u) << 5) - b2;
    ulong merged = (((ulong)W(i0 % 24u)) << 32) | (ulong)W(i2 % 24u);
    uint w7 = (uint)(merged >> sh2);
    uint w3 = (uint)(merged >> (sh2 + 12u));
    uint s7 = w7 & 0xffffu;
    uint s6 = (w7 >> 3) & 0xffffu;
    uint s5 = (w7 >> 6) & 0xffffu;
    uint s4 = (w7 >> 9) & 0xffffu;
    uint s3 = w3 & 0xffffu;
    uint s2 = (w3 >> 3) & 0xffffu;
    uint s1 = (w3 >> 6) & 0xffffu;
    uint s0 = (w3 >> 9) & 0xffffu;
"""


# Lane-only address computation for the code tile, hoisted so the fetch can be
# issued for several kt tiles at once.  i0/i1 (K2) and i0/i2 (K3) depend on `lane`
# alone -- NOT on kt -- so every tile in a KB block reads the same two offsets and
# the loads are mutually independent.
_PF_IDX_K2 = """
    uint pt_off = lane * 8u;
    uint pi1 = pt_off >> 4;
    uint pi0 = (pi1 + 15u) & 15u;
"""

_PF_IDX_K3 = """
    uint pt_off = lane * 8u;
    uint pb1 = (pt_off + 257u) * 3u;
    uint pi0 = ((pb1 - 16u) >> 5) % 24u;
    uint pi2 = (((pb1 + 21u) - 1u) >> 5) % 24u;
"""


def _substitute_fetch_prefetch(extract: str, K: int) -> str:
    """Point W(i) at registers already loaded by the prefetch loop.

    The extract block recomputes its own i0/i1 (lane-only, pure ALU); those
    shadow the hoisted pi* and are identical, so the decode is unchanged.
    """
    if K == 2:
        return (extract.replace("W(i0)", "pfa[k2]").replace("W(i1)", "pfb[k2]"))
    return (extract.replace("W(i0 % 24u)", "pfa[k2]")
                   .replace("W(i2 % 24u)", "pfb[k2]"))


def _dec(expr: str, use_lut: bool) -> str:
    return f"lut[{expr}]" if use_lut else f"cba_decode({expr})"


def _substitute_fetch(extract: str) -> str:
    """Replace the abstract word fetch W(i) with an indexed load from `wp`."""
    return (extract
            .replace("W(i0)", "wp[i0]").replace("W(i1)", "wp[i1]")
            .replace("W(i0 % 24u)", "wp[i0 % 24u]").replace("W(i2 % 24u)", "wp[i2 % 24u]"))


def _substitute_fetch_shuffle(extract: str) -> str:
    """Replace W(i) with an intra-simdgroup broadcast of a cooperative load.

    Every lane needs 2 words of the SAME {wpt}-word tile ({wpt} <= 24 < 32), so
    the direct kernel's 2 loads/lane are 64 loads of 16 distinct words for K2 --
    4x redundant.  They coalesce into few DRAM transactions, but they still
    occupy 64 load-issue slots on a kernel that is latency-bound rather than
    bandwidth-bound at bs1 row counts.

    Instead lane L loads word L (lanes >= wpt load nothing) and the two words a
    lane needs arrive by `simd_shuffle`: 1 load + 2 ALU shuffles per lane, no
    barrier and no threadgroup memory.  Values and accumulation order are
    unchanged => bit-identical.

    Safe because the direct GEMV has no divergence in the kt loop (no early
    return, every lane runs the same TK iterations), which simd_shuffle requires.
    """
    return (extract
            .replace("W(i0)", "simd_shuffle(myw, i0)")
            .replace("W(i1)", "simd_shuffle(myw, i1)")
            .replace("W(i0 % 24u)", "simd_shuffle(myw, i0 % 24u)")
            .replace("W(i2 % 24u)", "simd_shuffle(myw, i2 % 24u)"))


def _decode_tiles_source(K: int, use_lut: bool) -> str:
    wpt = 8 * K
    extract = _substitute_fetch(_EXTRACT_K2 if K == 2 else _EXTRACT_K3)
    stores = []
    for j in range(8):
        fi = j >> 1
        row = f"(lane & 3u) * 2u + {j & 1}u + {(fi & 1) * 8}u"
        col = f"2u * ((l0 >> 3) + {4 if j >= 4 else 0}u) + c_off"
        stores.append(
            f"    w[((ulong)(kt * 16u + {row})) * OC + nt * 16u + {col}] = {_dec(f's{j}', use_lut)};")
    return f"""
    uint lane = thread_position_in_grid.x;
    uint nt = thread_position_in_grid.y;
    uint kt = thread_position_in_grid.z;
    device const uint* wp = code + ((ulong)kt * TN + nt) * {wpt}u;
{extract}
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
{chr(10).join(stores)}
"""


def _moe_gemv_direct_source(K: int, use_lut: bool, shuffle: bool = False) -> str:
    """Barrier-free per-row GEMV: no threadgroup staging of code OR of x.

    The staged variant below runs TK iterations (128 for gate_up) with TWO
    threadgroup barriers each -- 256 synchronization points on a dependency
    chain that is already serial in kt.  At bs1 row counts the kernel is
    latency-bound, not bandwidth-bound (11 GB/s of a 101 GB/s roofline), so the
    barriers cost more than the staging saves.

    Here each simdgroup reads its own {wpt}-word code tile straight from device
    memory, and each lane reads the 16-wide x slice it needs directly.  The x
    reads are redundant across lanes but hit the same cache line, and no thread
    ever waits on another -- the inner loop has zero synchronization.

    Accumulation order per (row, out-channel) is unchanged => bit-identical.
    """
    wpt = 8 * K
    raw = _EXTRACT_K2 if K == 2 else _EXTRACT_K3
    extract = (_substitute_fetch_shuffle(raw) if shuffle else _substitute_fetch(raw))
    fetch = (f"        uint myw = (lane < {wpt}u) ? wp[lane] : 0u;\n"
             if shuffle else "")
    accs = []
    for j in range(8):
        fi = j >> 1
        xo = f"xrow + {j & 1}u + {(fi & 1) * 8}u"
        acc = "acc0" if j < 4 else "acc1"
        accs.append(f"        {acc} += (float)xrowp[{xo}] * (float){_dec(f's{j}', use_lut)};")
    return f"""
    uint ocb = thread_position_in_grid.x >> 8;
    uint row = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    int e = row_expert[row];
    device const uint* base = code + (ulong)e * ((ulong)TK * TN * {wpt}u);
    device const half* xbase = xh + (ulong)row * IC;

    float acc0 = 0.0f, acc1 = 0.0f;
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xrow = (lane & 3u) * 2u;

    for (uint kt = 0; kt < TK; kt++) {{
        device const uint* wp = base + ((ulong)kt * TN + ocb * 8u + sg) * {wpt}u;
        device const half* xrowp = xbase + kt * 16u;
{fetch}{extract}
{chr(10).join(accs)}
    }}

    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if ((lane & 3u) == 0u) {{
        uint col = 2u * (l0 >> 3) + c_off;
        ulong ob = (ulong)row * OC + ocb * 128u + sg * 16u;
        mid[ob + col] = acc0;
        mid[ob + col + 8u] = acc1;
    }}
"""


def _moe_gemv_splitk_source(K: int, use_lut: bool, shuffle: bool, S: int) -> str:
    """Split-K GEMV: partition the kt chain across S threadgroups.

    The per-row kernel above launches (OC/128) * M threadgroups -- only 64 for
    gate_up at bs1 (M=8, OC=1024) -- each grinding a TK=128 iteration chain whose
    only cross-iteration dependency is the acc0/acc1 float-add chain.  That is
    why it sits at 27% of roofline while lm_head sits at 97%: not enough
    independent work in flight to hide load+FMA latency.  Worse, the shortfall
    grows with GPU width (64 threadgroups is 6.4/core on a 10-core M4 but 0.8/core
    on an 80-core M3 Ultra), so this is the most portable of the kernel levers.

    Split S ways along kt: S x the threadgroups, chains S x shorter.  Each split
    writes its own partial and the caller sums them.  Partials are small -- S x M
    x OC f32, 128 KB at bs1 gate_up -- and this path only runs at GEMV row counts
    (prefill uses moe_gemm_rows), so the buffer can never blow up.

    DETERMINISM.  Reassociating a float sum is not bit-identical to the
    sequential kernel (f32 addition is not associative), but it IS fully
    deterministic: the partition is fixed by S, each split accumulates in its own
    fixed kt order, and the cross-split reduction is a fixed-order mx.sum, not an
    atomic.  So repeated runs are bit-reproducible -- gated by
    tests/test_splitk.py -- while agreement with the sequential kernel is a
    rounding-level check.  This is the one place in the runtime where we accept
    "deterministic but not bit-identical to the reference"; it is deliberate and
    the goldens are still checked within tolerance.
    """
    wpt = 8 * K
    raw = _EXTRACT_K2 if K == 2 else _EXTRACT_K3
    extract = (_substitute_fetch_shuffle(raw) if shuffle else _substitute_fetch(raw))
    fetch = (f"        uint myw = (lane < {wpt}u) ? wp[lane] : 0u;\n"
             if shuffle else "")
    accs = []
    for j in range(8):
        fi = j >> 1
        xo = f"xrow + {j & 1}u + {(fi & 1) * 8}u"
        acc = "acc0" if j < 4 else "acc1"
        accs.append(f"        {acc} += (float)xrowp[{xo}] * (float){_dec(f's{j}', use_lut)};")
    return f"""
    uint ocb = thread_position_in_grid.x >> 8;
    uint row = thread_position_in_grid.y;
    uint split = thread_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    int e = row_expert[row];
    device const uint* base = code + (ulong)e * ((ulong)TK * TN * {wpt}u);
    device const half* xbase = xh + (ulong)row * IC;

    // TK is a multiple of S by construction (the caller only picks S that
    // divides it), so every split covers exactly kps iterations.
    uint kps = TK / {S}u;
    uint kt0 = split * kps;
    uint kt1 = kt0 + kps;

    float acc0 = 0.0f, acc1 = 0.0f;
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xrow = (lane & 3u) * 2u;

    for (uint kt = kt0; kt < kt1; kt++) {{
        device const uint* wp = base + ((ulong)kt * TN + ocb * 8u + sg) * {wpt}u;
        device const half* xrowp = xbase + kt * 16u;
{fetch}{extract}
{chr(10).join(accs)}
    }}

    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if ((lane & 3u) == 0u) {{
        uint col = 2u * (l0 >> 3) + c_off;
        ulong ob = ((ulong)split * M + row) * OC + ocb * 128u + sg * 16u;
        part[ob + col] = acc0;
        part[ob + col + 8u] = acc1;
    }}
"""


def _moe_gemv_source(K: int, use_lut: bool) -> str:
    wpt = 8 * K
    extract = _substitute_fetch(_EXTRACT_K2 if K == 2 else _EXTRACT_K3)
    accs = []
    for j in range(8):
        fi = j >> 1
        row = f"xrow + {j & 1}u + {(fi & 1) * 8}u"
        acc = "acc0" if j < 4 else "acc1"
        accs.append(f"        {acc} += (float)s_x[{row}] * (float){_dec(f's{j}', use_lut)};")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint ocb = thread_position_in_grid.x >> 8;
    uint row = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;       // == tid & 31 on M-series;
    uint sg = simdgroup_index_in_threadgroup;    // builtins are authoritative

    threadgroup uint s_code[{8 * wpt}];
    threadgroup half s_x[16];

    int e = row_expert[row];
    device const uint* base = code + (ulong)e * ((ulong)TK * TN * {wpt}u);

    float acc0 = 0.0f, acc1 = 0.0f;
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xrow = (lane & 3u) * 2u;

    for (uint kt = 0; kt < TK; kt++) {{
        device const uint* g = base + ((ulong)kt * TN + ocb * 8u) * {wpt}u;
        for (uint i = tid; i < {8 * wpt}u; i += 256u) s_code[i] = g[i];
        if (tid < 16u) s_x[tid] = xh[(ulong)row * IC + kt * 16u + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const uint* wp = s_code + sg * {wpt}u;
{extract}
{chr(10).join(accs)}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    acc0 += simd_shuffle_xor(acc0, 1u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    if ((lane & 3u) == 0u) {{
        uint col = 2u * (l0 >> 3) + c_off;
        ulong ob = (ulong)row * OC + ocb * 128u + sg * 16u;
        mid[ob + col] = acc0;
        mid[ob + col + 8u] = acc1;
    }}
"""


def _moe_gemm_rows_source(K: int, use_lut: bool, R: int, KB: int = 1,
                          sortx: bool = False, prefetch: bool = False) -> str:
    """Row-blocked variant: ONE expert code stream shared by R rows.

    The decode-path kernel above reads a full expert stream per ROW, so its cost
    is exactly linear in rows with zero batch amortization -- which is what makes
    prefill quadratic-ish (a 2048-token chunk issues 16,384 rows).  Here a
    threadgroup owns one (expert, output-block) tile and accumulates it against R
    rows that share that expert, cutting stream reads by up to R.

    Bit-exactness: for a fixed (row, out-channel) the kt/j accumulation order is
    IDENTICAL to the R=1 kernel -- only the loop nest around it changed -- so the
    f32 result is bit-for-bit equal, not merely close.

    Indexing contract:
      rows_idx[grp*R + r]  = index into xh/mid of the r-th row of group grp,
                             or M (a zero row appended by the caller) for padding.
      group_expert[grp]    = expert id for the group, or -1 past the end.
    """
    wpt = 8 * K
    raw_extract = _EXTRACT_K2 if K == 2 else _EXTRACT_K3
    if prefetch:
        extract = _substitute_fetch_prefetch(raw_extract, K)
        pf_idx = _PF_IDX_K2 if K == 2 else _PF_IDX_K3
        pf_b = "pi1" if K == 2 else "pi2"
        # Issued BEFORE the barrier so these KB independent fetches overlap the
        # x-staging stores and the barrier wait.  This is the only remaining
        # dependent load on the kt critical path: today each tile is fetched and
        # immediately consumed, giving one outstanding load per lane.
        pf_block = f"""
        uint pfa[{KB}], pfb[{KB}];
#pragma clang loop unroll(full)
        for (uint k = 0; k < {KB}u; ++k) {{
            device const uint* wpk = base
                + ((ulong)(kb + k) * TN + ocb * 8u + sg) * {wpt}u;
            pfa[k] = wpk[pi0];
            pfb[k] = wpk[{pf_b}];
        }}"""
        wp_decl = ""
    else:
        extract = _substitute_fetch(raw_extract)
        pf_idx, pf_block = "", ""
        # `kt` lives here rather than in the shared loop header: the prefetch
        # variant does not need it, and leaving a dangling reference in the
        # fallback is exactly how this broke the first time.
        wp_decl = ("            uint kt = kb + k2;\n"
                   "            device const uint* wp = base + ((ulong)kt * TN "
                   "+ ocb * 8u + sg) * %du;" % wpt)
    accs = []
    for j in range(8):
        fi = j >> 1
        xo = f"xrow + {j & 1}u + {(fi & 1) * 8}u"
        acc = "acc0" if j < 4 else "acc1"
        accs.append(f"""
        {{
            float d = (float){_dec(f's{j}', use_lut)};
            uint xo = {xo};
#pragma clang loop unroll(full)
            for (uint r = 0; r < {R}u; ++r) {acc}[r] += (float)s_x[xb + r * 16u + xo] * d;
        }}""")
    # Sorted-x staging: rows of a group are CONSECUTIVE in xs, so the address is
    # computed (src_row0 + rr) rather than loaded from rows_idx and chased.  The
    # values staged are identical -- padding contributes 0 either way -- so the
    # output is bit-identical.  The OUTPUT write keeps the rows_idx indirection:
    # it happens once per group while staging happens TK times, so scattering on
    # the write side is far cheaper than un-permuting the whole mid tensor.
    if sortx:
        stage_body = ("            uint sr = src_row0[grp] + rr;\n"
                      "            s_x[i] = (rr < n_valid[grp])\n"
                      "                ? xh[(ulong)sr * IC + (kb + kk) * 16u + cc]\n"
                      "                : (half)0.0h;")
    else:
        stage_body = ("            s_x[i] = xh[(ulong)rows_idx[grp * %du + rr] * IC\n"
                      "                        + (kb + kk) * 16u + cc];" % R)
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint ocb = thread_position_in_grid.x >> 8;
    uint grp = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    int e = group_expert[grp];
    if (e < 0) return;

    threadgroup half s_x[{16 * R * KB}];

    device const uint* base = code + (ulong)e * ((ulong)TK * TN * {wpt}u);

    float acc0[{R}], acc1[{R}];
#pragma clang loop unroll(full)
    for (uint r = 0; r < {R}u; ++r) {{ acc0[r] = 0.0f; acc1[r] = 0.0f; }}

    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xrow = (lane & 3u) * 2u;

    // KB kt-tiles are staged per barrier pair.  The inner loop is otherwise
    // identical, so the kt/j accumulation order -- and therefore every output
    // bit -- is unchanged; only the barrier COUNT drops by KB.
{pf_idx}
    for (uint kb = 0; kb < TK; kb += {KB}u) {{
        for (uint i = tid; i < {16 * R * KB}u; i += 256u) {{
            uint kk = i / {16 * R}u;
            uint rem = i - kk * {16 * R}u;
            uint rr = rem >> 4, cc = rem & 15u;
{stage_body}
        }}
{pf_block}
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint k2 = 0; k2 < {KB}u; ++k2) {{
            uint xb = k2 * {16 * R}u;
{wp_decl}
{extract}
{"".join(accs)}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

#pragma clang loop unroll(full)
    for (uint r = 0; r < {R}u; ++r) {{
        float a0 = acc0[r], a1 = acc1[r];
        a0 += simd_shuffle_xor(a0, 1u);
        a0 += simd_shuffle_xor(a0, 2u);
        a1 += simd_shuffle_xor(a1, 1u);
        a1 += simd_shuffle_xor(a1, 2u);
        if ((lane & 3u) == 0u) {{
            uint col = 2u * (l0 >> 3) + c_off;
            ulong ob = (ulong)rows_idx[grp * {R}u + r] * OC + ocb * 128u + sg * 16u;
            mid[ob + col] = a0;
            mid[ob + col + 8u] = a1;
        }}
    }}
"""


def _scaled_had_source(rs: float) -> str:
    """Fused  f16( H128( f32(rows) * rin[e] ) * RS )  in one kernel.

    The unfused chain is arithmetically trivial and memory-brutal: at m=2048,
    IC=2048 it materialises `[m, IC]` f32 for the cast, again for the rin gather,
    again for the product, again for the native transform output, again for the RS scale --
    ~150 MB of traffic per layer per leg to do 128 adds per output.  Measured
    cost: 15.8% of prefill and 11.3% of decode for the rin stage alone, plus
    3.3%/8.9% for the Hadamard (doc §16.2).

    Here one threadgroup owns one (row, 128-block): it loads 128 values, scales
    them, runs the transform in threadgroup memory, and writes f16 once.  Only
    the input and the output ever reach DRAM.

    The transform is the in-place radix-2 butterfly, 7 stages, which computes
    y[j] = sum_i (-1)^popcount(i&j) x[i] -- the same Sylvester-ordered,
    unnormalised WHT and reduction order as mx.hadamard_transform. The final f16
    output is gated bit-for-bit against that native op chain; a separate
    tolerance test ties both implementations to ref.h128.
    """
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint blk = thread_position_in_grid.x >> 7;
    uint row = thread_position_in_grid.y;

    int e = row_expert[row];
    ulong off = (ulong)row * IC + (ulong)blk * 128u + tid;
    ulong roff = (ulong)e * IC + (ulong)blk * 128u + tid;

    threadgroup float v[128];
    v[tid] = (float)rows[off] * rin[roff];
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
    for (uint s = 0; s < 7u; ++s) {{
        uint msk = 1u << s;
        float mine = v[tid];
        float other = v[tid ^ msk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[tid] = (tid & msk) ? (other - mine) : (mine + other);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    out[off] = (half)(v[tid] * {rs!r}f);
"""


def _scaled_had_out_source(rs: float) -> str:
    """Fused  f16( H128(mid) * RS * rout[e] )  in one kernel.

    Keep the two post-transform multiplies as separate f32 statements in the
    same left-to-right order as the native MLX chain.  That preserves both f32
    rounding points before the single final f16 cast.
    """
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint blk = thread_position_in_grid.x >> 7;
    uint row = thread_position_in_grid.y;

    int e = row_expert[row];
    ulong off = (ulong)row * OC + (ulong)blk * 128u + tid;
    ulong roff = (ulong)e * OC + (ulong)blk * 128u + tid;

    threadgroup float v[128];
    v[tid] = mid[off];
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
    for (uint s = 0; s < 7u; ++s) {{
        uint msk = 1u << s;
        float mine = v[tid];
        float other = v[tid ^ msk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[tid] = (tid & msk) ? (other - mine) : (mine + other);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float scaled = v[tid] * {rs!r}f;
    float weighted = scaled * rout[roff];
    out[off] = (half)weighted;
"""


@lru_cache(maxsize=None)
def _scaled_had_kernel(rs: float):
    return mx.fast.metal_kernel(
        name="escha_scaled_had",
        input_names=["rows", "rin", "row_expert"],
        output_names=["out"],
        source=_scaled_had_source(rs),
    )


@lru_cache(maxsize=None)
def _scaled_had_out_kernel(rs: float):
    return mx.fast.metal_kernel(
        name="escha_scaled_had_out",
        input_names=["mid", "rout", "row_expert"],
        output_names=["out"],
        source=_scaled_had_out_source(rs),
    )


def use_fused_had() -> bool:
    """Fused expert Hadamard transforms (ESCHA_MLX_FUSED_HAD=0 disables)."""
    return os.environ.get("ESCHA_MLX_FUSED_HAD", "1") != "0"


def scaled_had(rows: mx.array, rin: mx.array, row_expert: mx.array,
               rs: float) -> mx.array:
    """rows [m, IC] f16, rin [E, IC] f32, row_expert [m] i32 -> [m, IC] f16."""
    m, ic = rows.shape
    kern = _scaled_had_kernel(float(rs))
    (out,) = kern(
        inputs=[rows, rin, row_expert],
        template=[("IC", ic)],
        grid=(128 * (ic // 128), m, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, ic)],
        output_dtypes=[mx.float16],
    )
    return out


def scaled_had_out(mid: mx.array, rout: mx.array, row_expert: mx.array,
                   rs: float) -> mx.array:
    """mid [m, OC] f32, rout [E, OC] f32 -> transformed [m, OC] f16."""
    m, oc = mid.shape
    kern = _scaled_had_out_kernel(float(rs))
    (out,) = kern(
        inputs=[mid, rout, row_expert],
        template=[("OC", oc)],
        grid=(128 * (oc // 128), m, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, oc)],
        output_dtypes=[mx.float16],
    )
    return out


def use_lut() -> bool:
    return os.environ.get("ESCHA_MLX_LUT", "0") == "1"


@lru_cache(maxsize=None)
def _lut_array() -> mx.array:
    return mx.array(ref.cba_lut())


@lru_cache(maxsize=None)
def _decode_tiles_kernel(K: int, lut: bool):
    inputs = ["code"] + (["lut"] if lut else [])
    return mx.fast.metal_kernel(
        name=f"escha_decode_tiles_k{K}{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["w"],
        header=_HEADER,
        source=_decode_tiles_source(K, lut),
    )


@lru_cache(maxsize=None)
def _moe_gemv_kernel(K: int, lut: bool, direct: bool = False, shuffle: bool = False):
    inputs = ["xh", "code", "row_expert"] + (["lut"] if lut else [])
    src = (_moe_gemv_direct_source(K, lut, shuffle) if direct
           else _moe_gemv_source(K, lut))
    tag = ("_direct" if direct else "") + ("_shf" if direct and shuffle else "")
    return mx.fast.metal_kernel(
        name=f"escha_moe_gemv_k{K}{tag}{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["mid"],
        header=_HEADER,
        source=src,
    )


def use_direct() -> bool:
    """Barrier-free per-row GEMV. Default ON; ESCHA_MLX_GEMV=staged reverts."""
    return os.environ.get("ESCHA_MLX_GEMV", "direct") == "direct"


def use_shuffle() -> bool:
    """Shuffle-broadcast the code tile. DEFAULT OFF -- measured a regression.

    The idea: every lane needs 2 words of the same <=24-word tile, so the direct
    kernel issues 64 loads of 16 distinct words (K2).  Replacing them with one
    cooperative load + 2 simd_shuffles should cut load-issue pressure.

    In-model measurement says otherwise (bench/sweep_kernel_variants.py --table
    splitk, medians of 5, drift control -0.4%):

        B=1  load 26.76 vs shuffle 25.94 tok/s   load +3.2%
        B=4  load 53.29 vs shuffle 49.99 tok/s   load +6.6%

    Consistent in sign at every batch size and split factor.  The premise was
    wrong: those redundant loads all hit one cache line and are nearly free,
    while simd_shuffle puts ALU latency directly on the kt dependency chain and
    idles lanes >= wpt during the cooperative load.

    Kept (and gated bit-identical) rather than deleted because the trade is
    hardware-dependent -- issue pressure matters more where there is less
    latency-hiding.  ESCHA_MLX_FETCH=shuffle re-enables it for a retest.
    """
    return os.environ.get("ESCHA_MLX_FETCH", "load") == "shuffle"


@lru_cache(maxsize=None)
def _moe_gemv_splitk_kernel(K: int, lut: bool, shuffle: bool, S: int):
    inputs = ["xh", "code", "row_expert"] + (["lut"] if lut else [])
    return mx.fast.metal_kernel(
        name=f"escha_moe_gemv_k{K}_sk{S}{'_shf' if shuffle else ''}"
             f"{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["part"],
        header=_HEADER,
        source=_moe_gemv_splitk_source(K, lut, shuffle, S),
    )


# Threadgroup count below which the GEMV is latency-bound and worth splitting.
# (OC/128)*M threadgroups: 64 for gate_up at bs1, 512 at M=64.
_SPLIT_TG_TARGET = 512


def split_k_for(m: int, oc: int, tk: int) -> int:
    """How many ways to split the kt chain. DEFAULT 1 -- measured a regression.

    ESCHA_MLX_SPLITK: an integer pins it; "auto" enables the size policy below.
    Default "1" = always the sequential kernel.

    The premise -- bs1 launches only (OC/128)*M = 64 threadgroups for gate_up,
    each on a 128-iteration serial kt chain, so the GPU starves -- did not
    survive measurement on a 10-core M4 (medians of 5, drift control -0.4%):

        B=1  S=1 26.76  S=4 26.40 (-1.3%)  S=policy(8) 26.28 (-1.8%)
        B=4  S=1 53.29  S=4 52.82 (-0.9%)  S=policy    52.53 (-1.4%)

    64 threadgroups is 512 simdgroups, which already saturates 10 cores; the
    extra kernel launch plus the S x M x OC partial write+read costs more than
    the shortened chains save.  Note also how narrow the useful window was:
    split_k_for returns 1 for gate_up from m>=64 (B>=8), and from m>=128 (B>=16)
    the MoE takes the row-blocked GEMM path so this function is not consulted at
    all.  So even a win here would only have applied to B<=4.

    Kept because the starvation argument is real on WIDER GPUs -- 64
    threadgroups is 6.4/core on this M4 but 1.6/core on a 40-core M4 Max and
    0.8/core on an 80-core M3 Ultra.  This is the first thing to re-measure
    there; `ESCHA_MLX_SPLITK=auto` turns the policy back on.
    """
    env = os.environ.get("ESCHA_MLX_SPLITK", "1")
    if env != "auto":
        s = int(env)
        return s if s > 1 and tk % s == 0 else 1
    tg = (oc // 128) * m
    if tg >= _SPLIT_TG_TARGET:
        return 1
    for s in (8, 4, 2):
        if tk % s == 0 and tg * s <= _SPLIT_TG_TARGET:
            return s
    return 1


def kt_block() -> int:
    """kt-tiles staged per barrier pair in the row-blocked GEMM.

    The GEMM ran two threadgroup barriers per kt iteration -- 256 sync points
    per threadgroup for gate_up (TK=128) -- which is the same pattern that cost
    the per-row GEMV 1.15-1.78x before it went barrier-free.  The GEMM cannot
    simply drop staging (its s_x tile is genuinely shared by all 8 simdgroups,
    and the rows_idx indirection makes a direct read a gather), but it can
    amortize: stage KB tiles, barrier once per KB.

    Must divide TK; TK is 128 (gate_up) and 32 (down), so any power of two up
    to 32 is safe.  ESCHA_MLX_KT_BLOCK overrides.
    """
    return int(os.environ.get("ESCHA_MLX_KT_BLOCK", "4"))


def use_prefetch() -> bool:
    """Prefetch KB code tiles into registers before consuming any of them.

    powermetrics says the GPU is 100% resident at max clock during this kernel
    while delivering 48% of the bandwidth roofline, and draws 17% LESS power than
    mx.quantized_matmul at 97% (doc §15.6) -- resident, clocked, not switching:
    stalled, not computing.  Everything else is already ruled out (not MAC-,
    barrier-, bandwidth- or dispatch-bound), which leaves memory-level
    parallelism.

    KT_BLOCK batched the x staging loads; the CODE stream was still fetched one
    tile per kt and consumed immediately -- one outstanding load per lane on a
    128-iteration chain.  The two word offsets depend on `lane` only, never on
    kt, so a whole KB block can be fetched up front as independent loads.

    Costs 2*KB registers per lane on top of acc0[R]/acc1[R].
    """
    return os.environ.get("ESCHA_MLX_PREFETCH", "0") != "0"


def use_sortx() -> bool:
    """Pre-sort x so a group's rows are consecutive (ESCHA_MLX_SORTX=0 reverts).

    Borrowed from mlx-lm's SwitchGLU, which physically permutes x via
    `_gather_sort` before calling the fused `gather_qmm` -- the kernel that
    sustains ~80% of roofline where ours manages 39-53% (doc §15).  Our kernel
    instead chased `rows_idx[grp*R+rr]` for every staged element, TK times per
    group; with x pre-sorted the row address is just `src_row0 + rr`.

    Costs one permute of m rows per leg; the OUTPUT write keeps the indirection,
    so no un-permute is needed.

    DEFAULT OFF -- measured a wash (doc §15.4): prefill -0.8%/-0.0%, decode
    -1.7% at B=16 and +0.6% at B=32, all inside an A/B/A drift band of the same
    size.  The premise (that chasing an arbitrary row per staged element is
    expensive) is wrong here, exactly as the previous five kernel hypotheses
    were.  Kept behind the flag, gated bit-identical, because the memory-
    divergence argument may hold on wider GPUs.
    """
    return os.environ.get("ESCHA_MLX_SORTX", "0") != "0"


@lru_cache(maxsize=None)
def _moe_gemm_rows_kernel(K: int, lut: bool, R: int, KB: int = 1,
                          sortx: bool = False, prefetch: bool = False):
    idx_inputs = (["src_row0", "n_valid", "rows_idx", "group_expert"] if sortx
                  else ["rows_idx", "group_expert"])
    inputs = ["xh", "code"] + idx_inputs + (["lut"] if lut else [])
    return mx.fast.metal_kernel(
        name=f"escha_moe_gemm_k{K}_r{R}_kb{KB}"
             f"{'_sx' if sortx else ''}{'_pf' if prefetch else ''}"
             f"{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["mid"],
        header=_HEADER,
        source=_moe_gemm_rows_source(K, lut, R, KB, sortx, prefetch),
    )


def moe_gemm_rows(xh: mx.array, code_u32: mx.array, rows_idx: mx.array,
                  group_expert: mx.array, K: int, IC: int, OC: int,
                  R: int, n_rows: int, sort_idx=None) -> mx.array:
    """Row-blocked expert GEMM.  See ``_moe_gemm_rows_source``.

    xh:           [n_rows + 1, IC] f16 -- row n_rows MUST be zeros (padding sink).
    rows_idx:     [n_groups * R] int32, padding entries == n_rows.
    group_expert: [n_groups] int32, -1 past the end.
    Returns mid [n_rows + 1, OC] f32; caller slices off the padding row.
    """
    n_groups = group_expert.shape[0]
    tk, tn = IC // 16, OC // 16
    lut = use_lut()
    kb = kt_block()
    if tk % kb:                      # a partial block would drop kt iterations
        kb = 1
    sortx = sort_idx is not None
    kern = _moe_gemm_rows_kernel(K, lut, R, kb, sortx, use_prefetch())
    idx = ([sort_idx[0], sort_idx[1], rows_idx, group_expert] if sortx
           else [rows_idx, group_expert])
    inputs = [xh, code_u32.reshape(-1)] + idx + ([_lut_array()] if lut else [])
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC)],
        grid=(256 * (OC // 128), n_groups, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_rows + 1, OC)],
        output_dtypes=[mx.float32],
    )
    return mid


def code_to_u32(code_i16: np.ndarray) -> np.ndarray:
    """View an int16 code tensor [..., 16K] as little-endian uint32 [..., 8K]."""
    a = np.ascontiguousarray(code_i16).view(np.uint16)
    return a.view(np.uint32)


def decode_tiles(code_u32: mx.array, K: int, IC: int, OC: int) -> mx.array:
    """Decode one projection's packed stream -> fp16 bare weight [IC, OC].

    code_u32: [TK, TN, 8K] uint32 (one expert's stream).
    """
    tk, tn = IC // 16, OC // 16
    lut = use_lut()
    kern = _decode_tiles_kernel(K, lut)
    inputs = [code_u32.reshape(-1)] + ([_lut_array()] if lut else [])
    (w,) = kern(
        inputs=inputs,
        template=[("TN", tn), ("OC", OC)],
        grid=(32, tn, tk),
        threadgroup=(32, 1, 1),
        output_shapes=[(IC, OC)],
        output_dtypes=[mx.float16],
    )
    return w


def moe_gemv(xh: mx.array, code_u32: mx.array, row_expert: mx.array,
             K: int, IC: int, OC: int, splits: int | None = None) -> mx.array:
    """Fused expert GEMV: mid[r, :] = xh[r, :] @ decode(code[row_expert[r]]).

    xh: [M, IC] f16 (already input-transformed per row).
    code_u32: [E, TK, TN, 8K] uint32 (E-stacked, as loaded).
    row_expert: [M] int32.
    splits: kt-split factor; None = the size policy, 1 = force the sequential
        kernel.  Pass 1 when you need the result to be bit-identical to the
        row-blocked GEMM (split-K reassociates the sum -- see
        _moe_gemv_splitk_source).
    Returns mid [M, OC] f32 (pre output-transform).
    """
    m = xh.shape[0]
    tk, tn = IC // 16, OC // 16
    lut = use_lut()
    inputs = [xh, code_u32.reshape(-1), row_expert] + ([_lut_array()] if lut else [])

    if splits is not None:
        s = splits if splits > 1 and tk % splits == 0 else 1
    else:
        s = split_k_for(m, OC, tk) if use_direct() else 1
    if s > 1:
        kern = _moe_gemv_splitk_kernel(K, lut, use_shuffle(), s)
        (part,) = kern(
            inputs=inputs,
            template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC), ("M", m)],
            grid=(256 * (OC // 128), m, s),
            threadgroup=(256, 1, 1),
            output_shapes=[(s, m, OC)],
            output_dtypes=[mx.float32],
        )
        # Fixed-order reduction (NOT an atomic): deterministic across runs.
        return part.sum(axis=0)

    kern = _moe_gemv_kernel(K, lut, use_direct(), use_shuffle())
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC)],
        grid=(256 * (OC // 128), m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, OC)],
        output_dtypes=[mx.float32],
    )
    return mid
