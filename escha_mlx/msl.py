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
    // 32-bit funnel shift replacing the emulated 64-bit shift.  The old form
    // merged W(i0)<<32 | W(i1) then shifted right by {0,16}: a ulong shift
    // lowers to a multi-op sequence on Apple GPUs (no 64-bit shifter), on the
    // critical path of every kt iteration.  shift==16 <=> t_off bit 3 == 0, in
    // which case wv = (hi<<16)|(lo>>16); otherwise wv = lo.  Bit-identical.
    uint lo = W(i1);
    uint hi = W(i0);
    uint wv = (t_off & 8u) ? lo : ((hi << 16u) | (lo >> 16u));
    uint s7 = wv & 0xffffu;              // NOT named `w`: decode_tiles' output
    uint s6 = (wv >> 2) & 0xffffu;       // buffer parameter is `w`, and MSL
    uint s5 = (wv >> 4) & 0xffffu;       // forbids shadowing a parameter in
    uint s4 = (wv >> 6) & 0xffffu;       // the outermost function block
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
    // 32-bit funnel shift replacing the emulated 64-bit shifts for w7/w3.
    // w7 = (merged >> sh2) low 32 (sh2 in [0,31]): sh2==0 ? lo : (lo>>sh2) |
    // (hi<<(32-sh2)).  w3 = (merged >> (sh2+12)) low 32: r2 = sh2+12 in
    // [12,43], so it needs the r2<32 / ==32 / >32 branches.  Bit-identical.
    uint hi = W(i0 % 24u);
    uint lo = W(i2 % 24u);
    uint w7 = (sh2 == 0u) ? lo : ((lo >> sh2) | (hi << (32u - sh2)));
    uint r2 = sh2 + 12u;
    uint w3 = (r2 < 32u) ? ((lo >> r2) | (hi << (32u - r2)))
            : (r2 == 32u) ? hi
            : (hi >> (r2 - 32u));
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
    x OC f32, 256 KB at bs1 gate_up under the S=8 policy -- and this path only runs at GEMV row counts
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



def _moe_gemv_had_source(K: int, use_lut: bool, rs: float) -> str:
    """Fused per-row GEMV with the input Hadamard transform built in.

    Currently the moe path runs `scaled_had` (writes xh [m, IC] f16) then
    `moe_gemv` (reads it) as two kernels with a full device round-trip.  This
    fuses them: each threadgroup transforms its row *inside* the kernel into
    threadgroup memory, then runs the direct GEMV reading from there -- one
    launch per leg instead of two and no [m, IC] intermediate.

    Threadgroup `sg` (0..7) transforms blocks {sg*2, sg*2+1} of the row with
    the slotless radix-2 butterfly (stages 0-1 in registers, 2-6 via
    simd_shuffle_xor) -- the same stage order and pairings as `_had_source`, so
    the transformed f16 values are bit-for-bit the standalone scaled_had's.
    Subsequent threadgroups read the shared s_x, so the row is transformed once
    per threadgroup (redundant across the ocb-slices of a row, but that ALU is
    ~5% of the GEMV's MACs).  Accumulation order per (row, out-channel) is
    unchanged => bit-identical to [scaled_had; moe_gemv].

    Blocks past IC/128 are guarded off (the down leg is IC=512 = 4 blocks, so
    simdgroups 2..7 transform nothing).  IC must be a multiple of 256: a
    simdgroup covers 2 blocks of 128 elements.
    """
    wpt = 8 * K
    raw = _EXTRACT_K2 if K == 2 else _EXTRACT_K3
    extract = _substitute_fetch(raw)
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

    threadgroup half s_x[IC];

    // ---- input Hadamard transform: sg transforms blocks (2*sg, 2*sg+1) ----
#pragma clang loop unroll(full)
    for (uint bi = 0u; bi < 2u; ++bi) {{
        uint blk2 = sg * 2u + bi;
        if (blk2 >= IC / 128u) continue;
        ulong inb = (ulong)row_token[row] * IC + (ulong)blk2 * 128u + (ulong)lane * 4u;
        ulong rinb = (ulong)e * IC + (ulong)blk2 * 128u + (ulong)lane * 4u;
        float v[4];
        v[0] = (float)x[inb + 0u] * rin[rinb + 0u];
        v[1] = (float)x[inb + 1u] * rin[rinb + 1u];
        v[2] = (float)x[inb + 2u] * rin[rinb + 2u];
        v[3] = (float)x[inb + 3u] * rin[rinb + 3u];
        {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }}
        {{ float a = v[2], b = v[3]; v[2] = a + b; v[3] = a - b; }}
        {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; }}
        {{ float a = v[1], b = v[3]; v[1] = a + b; v[3] = a - b; }}
#pragma clang loop unroll(full)
        for (uint s = 2u; s < 7u; ++s) {{
            uint shl = 1u << (s - 2u);
#pragma clang loop unroll(full)
            for (uint k = 0u; k < 4u; ++k) {{
                float mine = v[k];
                float other = simd_shuffle_xor(mine, shl);
                v[k] = (lane & shl) ? (other - mine) : (mine + other);
            }}
        }}
        ulong so = (ulong)blk2 * 128u + (ulong)lane * 4u;
        float RS = {rs!r}f;
        s_x[so + 0u] = (half)(v[0] * RS);
        s_x[so + 1u] = (half)(v[1] * RS);
        s_x[so + 2u] = (half)(v[2] * RS);
        s_x[so + 3u] = (half)(v[3] * RS);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- direct GEMV reading the transformed row from s_x ----
    float acc0 = 0.0f, acc1 = 0.0f;
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xrow = (lane & 3u) * 2u;

    for (uint kt = 0u; kt < TK; kt++) {{
        device const uint* wp = base + ((ulong)kt * TN + ocb * 8u + sg) * {wpt}u;
        const threadgroup half* xrowp = s_x + kt * 16u;
{extract}
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


def _moe_gemv_had_kb_source(K: int, use_lut: bool, rs: float, KB: int) -> str:
    """moe_gemv_had with KB-deep code-tile prefetch in the direct loop.

    The shipped had kernel keeps the 128-iteration serial kt chain (gate_up)
    with one outstanding code-word load per lane per iteration.  At bs1 row
    counts the code stream stalls the chain: folding cba_decode to a constant
    drops the isolated gu leg 49us -> 22us, i.e. the on-the-fly code loads are
    ~55% of the leg.  The word offsets depend on `lane` only (never kt), so a
    whole KB block can be fetched as independent loads before any is consumed
    (same idea as the direct-GEMV KB staging), turning the chain KB-deep.

    Bit-identical: same kt order, same per-kt accumulate.  The had transform
    preamble (s_x + barrier) is unchanged.
    """
    wpt = 8 * K
    raw = _EXTRACT_K2 if K == 2 else _EXTRACT_K3
    pf_b = "pi1" if K == 2 else "pi2"
    pf_idx = _PF_IDX_K2 if K == 2 else _PF_IDX_K3
    extract = _substitute_fetch_prefetch(raw, K)
    # Software-pipelined double-buffer prefetch.  The shipped kernel loads the
    # KB code tiles for block `kb` at the top of iteration `kb` and consumes
    # them immediately, so the next block's load cannot start until the current
    # block is fully consumed -- one code-load latency exposed per iteration.
    # Here block 0 is fetched BEFORE the had transform (overlapping the x/rin
    # loads and the butterfly+barrier), and each iteration issues the load for
    # block kb+KB into the `next` registers before consuming `cur`, so the
    # load latency is hidden behind the current block's decode+FMA chain
    # (same memory-level-parallelism argument as the KB-depth itself).
    # Values and kt order are unchanged => bit-identical.
    pf_block = f"""
        uint pfa[{KB}], pfb[{KB}];
        uint nfa[{KB}], nfb[{KB}];
#pragma clang loop unroll(full)
        for (uint k = 0; k < {KB}u; ++k) {{
            uint nk = (uint)k;
            device const uint* wpk = base
                + (ulong)(nk * TN + ocb * 8u + sg) * {wpt}u;
            pfa[k] = wpk[pi0]; pfb[k] = wpk[{pf_b}];
            nfa[k] = 0u; nfb[k] = 0u;
        }}"""
    pf_look = f"""
        uint kbk = kb + {KB}u;
        if (kbk < TK) {{
#pragma clang loop unroll(full)
            for (uint k = 0; k < {KB}u; ++k) {{
                device const uint* wpk = base
                    + ((ulong)(kbk + k) * TN + ocb * 8u + sg) * {wpt}u;
                nfa[k] = wpk[pi0]; nfb[k] = wpk[{pf_b}];
            }}
        }}"""
    pf_adv = f"""
#pragma clang loop unroll(full)
        for (uint k = 0; k < {KB}u; ++k) {{ pfa[k] = nfa[k]; pfb[k] = nfb[k]; }}"""
    accs = []
    for j in range(8):
        fi = j >> 1
        xo = f"xrow + {j & 1}u + {(fi & 1) * 8}u"
        acc = "acc0" if j < 4 else "acc1"
        accs.append(f"            {acc} += (float)xrowp[{xo}] * (float){_dec(f's{j}', use_lut)};")
    return f"""
    uint ocb = thread_position_in_grid.x >> 8;
    uint row = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    int e = row_expert[row];
    device const uint* base = code + (ulong)e * ((ulong)TK * TN * {wpt}u);

    threadgroup half s_x[IC];
{pf_idx}
{pf_block}

#pragma clang loop unroll(full)
    for (uint bi = 0u; bi < 2u; ++bi) {{
        uint blk2 = sg * 2u + bi;
        if (blk2 >= IC / 128u) continue;
        ulong inb = (ulong)row_token[row] * IC + (ulong)blk2 * 128u + (ulong)lane * 4u;
        ulong rinb = (ulong)e * IC + (ulong)blk2 * 128u + (ulong)lane * 4u;
        float v[4];
        v[0] = (float)x[inb + 0u] * rin[rinb + 0u];
        v[1] = (float)x[inb + 1u] * rin[rinb + 1u];
        v[2] = (float)x[inb + 2u] * rin[rinb + 2u];
        v[3] = (float)x[inb + 3u] * rin[rinb + 3u];
        {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }}
        {{ float a = v[2], b = v[3]; v[2] = a + b; v[3] = a - b; }}
        {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; }}
        {{ float a = v[1], b = v[3]; v[1] = a + b; v[3] = a - b; }}
#pragma clang loop unroll(full)
        for (uint s = 2u; s < 7u; ++s) {{
            uint shl = 1u << (s - 2u);
#pragma clang loop unroll(full)
            for (uint k = 0u; k < 4u; ++k) {{
                float mine = v[k];
                float other = simd_shuffle_xor(mine, shl);
                v[k] = (lane & shl) ? (other - mine) : (mine + other);
            }}
        }}
        ulong so = (ulong)blk2 * 128u + (ulong)lane * 4u;
        float RS = {rs!r}f;
        s_x[so + 0u] = (half)(v[0] * RS);
        s_x[so + 1u] = (half)(v[1] * RS);
        s_x[so + 2u] = (half)(v[2] * RS);
        s_x[so + 3u] = (half)(v[3] * RS);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc0 = 0.0f, acc1 = 0.0f;
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xrow = (lane & 3u) * 2u;
    for (uint kb = 0; kb < TK; kb += {KB}u) {{
{pf_look}
#pragma clang loop unroll(full)
        for (uint k2 = 0; k2 < {KB}u; ++k2) {{
            const threadgroup half* xrowp = s_x + (kb + k2) * 16u;
{extract}
{"\n".join(accs)}
        }}
{pf_adv}
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


def _had_source(rs: float, kind: str) -> str:
    """Fused f16( H128( x ) * RS [* rout[e]] ) in one kernel.

    kind="in" : x = f32(rows) * rin[e];  out = f16(H128(x) * RS)
    kind="out": x = mid (f32);           out = f16(H128(x) * RS * rout[e])

    The unfused chain is arithmetically trivial and memory-brutal: at m=2048,
    IC=2048 it materialises `[m, IC]` f32 for the cast, again for the rin gather,
    again for the product, again for the native transform output, again for the
    RS scale -- ~150 MB of traffic per layer per leg to do 128 adds per output.
    Measured cost: 15.8% of prefill and 11.3% of decode for the rin stage
    alone, plus 3.3%/8.9% for the Hadamard (doc §16.2).

    The transform is the in-place radix-2 butterfly, 7 stages, computing
    y[j] = sum_i (-1)^popcount(i&j) x[i] -- the same Sylvester-ordered,
    unnormalised WHT and reduction order as mx.hadamard_transform.  The earlier
    kernel ran it in threadgroup memory with two barriers per stage (14 barriers
    per 128-block).  This one uses one simdgroup (32 lanes x 4 elements = 128)
    per block: stages 0-1 butterfly within a lane's 4 registers, stages 2-6
    across lanes via `simd_shuffle_xor` -- zero barriers, zero threadgroup
    memory.  Element index i = lane*4 + k, so bit s of i is register/bit (s-2)
    of the lane for s >= 2 and every pair lives inside the simdgroup.

    Per element the stage sequence, the (i, i^2^s) pairings, the +/- convention
    and the f32 rounding positions are unchanged, so the f16 output is
    bit-for-bit the barrier version's.

    Grid: one simdgroup per 128-block (threadgroups pack 8 blocks each when
    HAD_TG=256), M rows in y.  Launch overhead is real here -- prefill issues
    32768 32-thread threadgroups per transform ([m,IC]=[2048,2048]); packing
    8 blocks into a 256-thread threadgroup cuts that 8x with the exact same
    per-block butterfly, so output bits are unchanged.  A partial last group
    (IC/128 not a multiple of 8) is guarded out below.
    """
    if kind == "in":
        loads = [f"v[{k}] = (float)rows[base + {k}u] * rin[rbase + {k}u];"
                 for k in range(4)]
        stores = [f"out[base + {k}u] = (half)(v[{k}] * "
                  + f"{rs!r}f" + f");" for k in range(4)]
    else:
        loads = [f"v[{k}] = mid[base + {k}u];" for k in range(4)]
        stores = [f"out[base + {k}u] = (half)(v[{k}] * "
                  + f"{rs!r}f" + f" * rout[rbase + {k}u]);" for k in range(4)]
    loads_src = "\n".join(loads)
    stores_src = "\n".join(stores)
    return f"""
    uint lane = thread_index_in_simdgroup;   // 0..31
    uint blk = thread_position_in_grid.x >> 5;
    uint row = thread_position_in_grid.y;
    uint numBlocks = IC / 128u;
    if (blk >= numBlocks) return;   // partial last group

    int e = row_expert[row];
    ulong base = (ulong)row * IC + (ulong)blk * 128u + (ulong)lane * 4u;
    ulong rbase = (ulong)e * IC + (ulong)blk * 128u + (ulong)lane * 4u;

    float v[4];
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {{
{loads_src}
    }}

    // stages 0-1: element bits within the lane's 4 registers
    {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }}
    {{ float a = v[2], b = v[3]; v[2] = a + b; v[3] = a - b; }}
    {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; }}
    {{ float a = v[1], b = v[3]; v[1] = a + b; v[3] = a - b; }}

    // stages 2-6: lane bits; every pair is inside this simdgroup
#pragma clang loop unroll(full)
    for (uint s = 2u; s < 7u; ++s) {{
        uint shl = 1u << (s - 2u);
#pragma clang loop unroll(full)
        for (uint k = 0u; k < 4u; ++k) {{
            float mine = v[k];
            float other = simd_shuffle_xor(mine, shl);
            v[k] = (lane & shl) ? (other - mine) : (mine + other);
        }}
    }}

#pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {{
{stores_src}
    }}
"""


def _scaled_had_source(rs: float) -> str:
    return _had_source(rs, "in")


def _scaled_had_out_source(rs: float) -> str:
    return _had_source(rs, "out")


def _scaled_had_out_silu_source(rs: float, hb: int) -> str:
    """gu-leg epilogue fused into one kernel: output-Hadamard + silu gate.

    Computes d_lo = f16(H128(mid[blk])*RS*rout) and d_hi = f16(H128(mid[blk+
    hb])*RS*rout) in one simdgroup (the same slotless radix-2 butterfly and
    pair order as `_had_source("out")`), then s16[i] = f16(f32(d_lo[i]) *
    sigmoid(f32(d_lo[i]))) and h[i] = s16[i] * d_hi[i].  Numerically identical
    to scaled_had_out -> the gu-leg silu chain EXCEPT the sigmoid, which MLX
    compiles with fast-math on this build (see use_silu_tail) -- that last-ulp
    difference is why this kernel is opt-in, not default."""
    return f"""
    uint lane = thread_index_in_simdgroup;   // 0..31
    uint blk = thread_position_in_grid.x >> 5;  // 0..{hb}-1
    uint row = thread_position_in_grid.y;

    int e = row_expert[row];
    ulong lbase = (ulong)row * OC + (ulong)blk * 128u + (ulong)lane * 4u;
    ulong lrbase = (ulong)e * OC + (ulong)blk * 128u + (ulong)lane * 4u;
    ulong hbase = lbase + (ulong)({hb}u * 128u);
    ulong hrbase = lrbase + (ulong)({hb}u * 128u);

    float vL[4]; float vH[4];
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {{ vL[k] = mid[lbase + k]; vH[k] = mid[hbase + k]; }}
    {{ float a = vL[0], b = vL[1]; vL[0] = a + b; vL[1] = a - b; }}
    {{ float a = vL[2], b = vL[3]; vL[2] = a + b; vL[3] = a - b; }}
    {{ float a = vL[0], b = vL[2]; vL[0] = a + b; vL[2] = a - b; }}
    {{ float a = vL[1], b = vL[3]; vL[1] = a + b; vL[3] = a - b; }}
    {{ float a = vH[0], b = vH[1]; vH[0] = a + b; vH[1] = a - b; }}
    {{ float a = vH[2], b = vH[3]; vH[2] = a + b; vH[3] = a - b; }}
    {{ float a = vH[0], b = vH[2]; vH[0] = a + b; vH[2] = a - b; }}
    {{ float a = vH[1], b = vH[3]; vH[1] = a + b; vH[3] = a - b; }}
#pragma clang loop unroll(full)
    for (uint s = 2u; s < 7u; ++s) {{
        uint shl = 1u << (s - 2u);
#pragma clang loop unroll(full)
        for (uint k = 0u; k < 4u; ++k) {{
            float mL = vL[k]; float oL = simd_shuffle_xor(mL, shl);
            vL[k] = (lane & shl) ? (oL - mL) : (mL + oL);
            float mH = vH[k]; float oH = simd_shuffle_xor(mH, shl);
            vH[k] = (lane & shl) ? (oH - mH) : (mH + oH);
        }}
    }}

    float RS = {rs!r}f;
    ulong so = (ulong)row * (OC/2u) + (ulong)blk * 128u + (ulong)lane * 4u;
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {{
        half lo = (half)(vL[k] * RS * rout[lrbase + k]);
        float g = (float)lo;
        float ax = metal::abs(g);
        float y = 1.0f / (1.0f + metal::exp(ax));
        float sg = (g < 0.0f) ? y : 1.0f - y;
        half s = (half)(g * sg);
        half hv = (half)(vH[k] * RS * rout[hrbase + k]);
        s16[so + k] = s;
        h[so + k] = s * hv;
    }}
"""





def had_tg() -> int:
    """Threads per threadgroup for the scaled-Hadamard transforms.  Default
    256 packs 8 (128-block) simdgroups into each threadgroup, cutting launch
    count 8x with identical per-block butterflies (bit-identical output).
    ESCHA_MLX_HAD_TG=32 restores one simdgroup per threadgroup."""
    v = os.environ.get("ESCHA_MLX_HAD_TG", "256")
    return int(v) if v.lstrip("-").isdigit() and int(v) >= 32 else 256


def _had_grid(ic: int, m: int, tg: int):
    """Grid (threads x, threads y) for a transform of [m, ic], with `tg`-thread
    threadgroups; a partial last group is guarded inside the kernel."""
    blocks = ic // 128
    bpg = tg // 32
    return (tg * ((blocks + bpg - 1) // bpg), m, 1)



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


@lru_cache(maxsize=None)
def _scaled_had_out_pack_kernel(rs: float):
    return mx.fast.metal_kernel(
        name="escha_scaled_had_out_pk",
        input_names=["mid", "rout", "row_expert"],
        output_names=["out"],
        source=_scaled_had_out_pack_source(rs),
    )


@lru_cache(maxsize=None)
def _scaled_had_out_sum_pack_kernel(rs: float, top_k: int):
    return mx.fast.metal_kernel(
        name=f"escha_scaled_had_out_sum_pk_tk{top_k}",
        input_names=["mid", "rout", "row_expert", "w16"],
        output_names=["y"],
        source=_scaled_had_out_sum_pack_source(rs, top_k),
    )


def use_had_pack() -> bool:
    """Pack 8 128-blocks of the decode output-Hadamard kernels into one
    256-thread threadgroup (ESCHA_MLX_HAD_PACK).  The decode transforms launch
    tiny 32-thread threadgroups (64+16 per layer at bs1); packing cuts that to
    8+2 and removes ~2500 tiny launches per step.  Bit-identical (each block's
    butterfly is simdgroup-local).  Default ON for the decode path."""
    return os.environ.get("ESCHA_MLX_HAD_PACK", "1") != "0"


def use_fused_had() -> bool:
    """Fused expert Hadamard transforms (ESCHA_MLX_FUSED_HAD=0 disables)."""
    return os.environ.get("ESCHA_MLX_FUSED_HAD", "1") != "0"


def use_silu_tail() -> bool:
    """Fuse the gu-leg output Hadamard + silu gate into one kernel
    (ESCHA_MLX_SILU_TAIL=1 enables).

    DEFAULT OFF.  The fusion is bit-identical on the butterfly arithmetic but
    NOT on the sigmoid: MLX compiles its elementwise Sigmoid with fast-math
    (measured: on this 0.32.0 build its GPU sigmoid differs from the stable
    1/(1+exp(|x|)) form on ~39% of f32 inputs), and a mx.fast.metal_kernel
    cannot reproduce it, so the fused s16/h would break the bit-identical
    decode contract.  `scaled_had_out_silu` stays for a build where MLX's
    sigmoid is bit-exact; flipping the flag would buy one launch and the
    [m, OC] f16 round-trip per MoE layer."""
    return os.environ.get("ESCHA_MLX_SILU_TAIL", "0") == "1"


@lru_cache(maxsize=None)
def _scaled_had_out_silu_kernel(rs: float, hb: int):
    return mx.fast.metal_kernel(
        name=f"escha_scaled_had_out_silu_hb{hb}",
        input_names=["mid", "rout", "row_expert"],
        output_names=["s16", "h"],
        source=_scaled_had_out_silu_source(rs, hb),
    )


def scaled_had_out_silu(mid: mx.array, rout: mx.array, row_expert: mx.array,
                        oc: int, rs: float):
    """mid [m, OC] f32 -> (s16 [m, OC/2] f16, h [m, OC/2] f16), computed in
    one kernel.  Opt-in (ESCHA_MLX_SILU_TAIL=1): bit-identical to
    scaled_had_out then the gu-leg silu chain only where MLX's GPU sigmoid
    matches 1/(1+exp(|x|)) -- see use_silu_tail."""
    m = mid.shape[0]
    half = oc // 2
    hb = half // 128
    kern = _scaled_had_out_silu_kernel(float(rs), int(hb))
    s16, h = kern(
        inputs=[mid, rout, row_expert],
        template=[("OC", oc)],
        grid=(32 * hb, m, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, half), (m, half)],
        output_dtypes=[mx.float16, mx.float16],
    )
    return s16, h


def scaled_had(rows: mx.array, rin: mx.array, row_expert: mx.array,
               rs: float) -> mx.array:
    """rows [m, IC] f16, rin [E, IC] f32, row_expert [m] i32 -> [m, IC] f16."""
    m, ic = rows.shape
    kern = _scaled_had_kernel(float(rs))
    tg = had_tg()
    (out,) = kern(
        inputs=[rows, rin, row_expert],
        template=[("IC", ic)],
        grid=_had_grid(ic, m, tg),
        threadgroup=(tg, 1, 1),
        output_shapes=[(m, ic)],
        output_dtypes=[mx.float16],
    )
    return out


def scaled_had_out(mid: mx.array, rout: mx.array, row_expert: mx.array,
                   rs: float) -> mx.array:
    """mid [m, OC] f32, rout [E, OC] f32 -> transformed [m, OC] f16."""
    m, oc = mid.shape
    if use_had_pack():
        kern = _scaled_had_out_pack_kernel(float(rs))
        nblk = oc // 128
        tgx = (nblk + 7) // 8
        (out,) = kern(
            inputs=[mid, rout, row_expert],
            template=[("IC", oc)],
            grid=(256 * tgx, m, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(m, oc)],
            output_dtypes=[mx.float16],
        )
        return out
    kern = _scaled_had_out_kernel(float(rs))
    tg = had_tg()
    (out,) = kern(
        inputs=[mid, rout, row_expert],
        template=[("IC", oc)],
        grid=_had_grid(oc, m, tg),
        threadgroup=(tg, 1, 1),
        output_shapes=[(m, oc)],
        output_dtypes=[mx.float16],
    )
    return out


def _scaled_had_out_pack_source(rs: float) -> str:
    """scaled_had_out with 8 128-blocks packed per 256-thread threadgroup.

    The decode output-transforms launch one tiny 32-thread threadgroup per
    (row, 128-block).  At bs1 a gu leg is (1024/128)*8 = 64 such threadgroups
    and the dn leg (2048/128)*1 = 16 -- legacy tiny-TG launches whose fixed
    dispatch cost dominates at single-token row counts.  Each block's radix-2
    butterfly is entirely inside one simdgroup (barrier-free, stages 2-6 via
    simd_shuffle_xor), so 8 blocks pack into one 256-thread threadgroup with
    no inter-simdgroup communication.  Same per-block butterfly, same rounding
    => bit-identical to _scaled_had_out_source."""
    stores = []
    for k in range(4):
        stores.append(f"    out[base + {k}u] = (half)(v[{k}] * "
                      + f"{rs!r}f" + f" * rout[rbase + {k}u]);")
    return f"""
    uint lane = thread_index_in_simdgroup;   // 0..31
    uint sg = simdgroup_index_in_threadgroup; // 0..7 (8 blocks per TG)
    uint row = thread_position_in_grid.y;
    uint nblk = IC >> 7;
    uint blk = (thread_position_in_grid.x >> 8) * 8u + sg;
    if (blk >= nblk) return;

    int e = row_expert[row];
    ulong base = (ulong)row * IC + (ulong)blk * 128u + (ulong)lane * 4u;
    ulong rbase = (ulong)e * IC + (ulong)blk * 128u + (ulong)lane * 4u;

    float v[4];
    v[0] = mid[base + 0u]; v[1] = mid[base + 1u]; v[2] = mid[base + 2u]; v[3] = mid[base + 3u];
    {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }}
    {{ float a = v[2], b = v[3]; v[2] = a + b; v[3] = a - b; }}
    {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; }}
    {{ float a = v[1], b = v[3]; v[1] = a + b; v[3] = a - b; }}
#pragma clang loop unroll(full)
    for (uint s = 2u; s < 7u; ++s) {{
        uint shl = 1u << (s - 2u);
#pragma clang loop unroll(full)
        for (uint k = 0u; k < 4u; ++k) {{
            float mine = v[k];
            float other = simd_shuffle_xor(mine, shl);
            v[k] = (lane & shl) ? (other - mine) : (mine + other);
        }}
    }}
{chr(10).join(stores)}
"""

def _scaled_had_out_sum_source(rs: float, top_k: int) -> str:
    """Fused down-leg output transform + score-weighted token reduction.

    Replaces [scaled_had_out(dn); d16.astype(f32)*w16; reshape; sum; f16 round]
    with ONE kernel: y[token, :] = f16( sum_k f32(d16_{token*top_k+k}) *
    f32(w16[token*top_k+k]) ) over the token's top_k rows.

    Rows are token-major (row = token*top_k + k, the _rows invariant).  Each
    32-lane simdgroup owns one (token, 128-OC block); it loops the top_k rows
    of that token, running the SAME barrier-free radix-2 butterfly as
    `_had_source(kind="out")` on each row's mid block, scaling by RS*rout,
    casting f16, and accumulating f32(d16)*f32(w16) in k order into f32
    registers.  The f32 accumulation order across k is identical to the
    segmented-sum chain it replaces, and the final f16 cast happens once, so
    the output bits are identical to scaled_had_out -> score-mul -> sum.

    Run order of stages/pairings and the f32 rounding positions match
    `_had_source` exactly, so each d16 element is bit-for-bit the standalone
    scaled_had_out's.  One 32-thread threadgroup per (token, 128-block).
    """
    stores = []
    for k in range(4):
        stores.append(f"        y[ob + {k}u] = (half)acc[{k}];")
    return f"""
    uint lane = thread_index_in_simdgroup;   // 0..31
    uint blk = thread_position_in_grid.x >> 5;  // 0 .. OC/128-1
    uint token = thread_position_in_grid.y;
    uint numBlocks = OC / 128u;
    if (blk >= numBlocks) return;   // partial last group

    uint ob = (ulong)token * OC + (ulong)blk * 128u + (ulong)lane * 4u;

    float acc[4];
    acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;

#pragma clang loop unroll(full)
    for (uint k = 0u; k < {top_k}u; ++k) {{
        uint r = token * {top_k}u + k;
        int e = row_expert[r];
        ulong mb = (ulong)r * OC + (ulong)blk * 128u + (ulong)lane * 4u;
        ulong rbase = (ulong)e * OC + (ulong)blk * 128u + (ulong)lane * 4u;

        float v[4];
        v[0] = mid[mb + 0u]; v[1] = mid[mb + 1u]; v[2] = mid[mb + 2u]; v[3] = mid[mb + 3u];
        {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }}
        {{ float a = v[2], b = v[3]; v[2] = a + b; v[3] = a - b; }}
        {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; }}
        {{ float a = v[1], b = v[3]; v[1] = a + b; v[3] = a - b; }}
#pragma clang loop unroll(full)
        for (uint s = 2u; s < 7u; ++s) {{
            uint shl = 1u << (s - 2u);
#pragma clang loop unroll(full)
            for (uint jj = 0u; jj < 4u; ++jj) {{
                float mine = v[jj];
                float other = simd_shuffle_xor(mine, shl);
                v[jj] = (lane & shl) ? (other - mine) : (mine + other);
            }}
        }}
        float RS = {rs!r}f;
        float w = (float)((half)w16[r]);
        float d0 = (float)((half)(v[0] * RS * rout[rbase + 0u]));
        float d1 = (float)((half)(v[1] * RS * rout[rbase + 1u]));
        float d2 = (float)((half)(v[2] * RS * rout[rbase + 2u]));
        float d3 = (float)((half)(v[3] * RS * rout[rbase + 3u]));
        acc[0] += d0 * w; acc[1] += d1 * w; acc[2] += d2 * w; acc[3] += d3 * w;
    }}
{chr(10).join(stores)}
"""


def _scaled_had_out_sum_pack_source(rs: float, top_k: int) -> str:
    """scaled_had_out_sum with 8 128-blocks packed per 256-thread TG (see
    _scaled_had_out_pack_source).  Each simdgroup owns one (token, block) and
    loops the token's top_k rows with the same barrier-free butterfly +
    score-weighted f32 accumulation; blocks are independent => bit-identical
    to _scaled_had_out_sum_source."""
    stores = []
    for k in range(4):
        stores.append(f"        y[ob + {k}u] = (half)acc[{k}];")
    return f"""
    uint lane = thread_index_in_simdgroup;   // 0..31
    uint sg = simdgroup_index_in_threadgroup; // 0..7
    uint token = thread_position_in_grid.y;
    uint nblk = OC >> 7;
    uint blk = (thread_position_in_grid.x >> 8) * 8u + sg;
    if (blk >= nblk) return;

    uint ob = (ulong)token * OC + (ulong)blk * 128u + (ulong)lane * 4u;

    float acc[4];
    acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;

#pragma clang loop unroll(full)
    for (uint k = 0u; k < {top_k}u; ++k) {{
        uint r = token * {top_k}u + k;
        int e = row_expert[r];
        ulong mb = (ulong)r * OC + (ulong)blk * 128u + (ulong)lane * 4u;
        ulong rbase = (ulong)e * OC + (ulong)blk * 128u + (ulong)lane * 4u;

        float v[4];
        v[0] = mid[mb + 0u]; v[1] = mid[mb + 1u]; v[2] = mid[mb + 2u]; v[3] = mid[mb + 3u];
        {{ float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; }}
        {{ float a = v[2], b = v[3]; v[2] = a + b; v[3] = a - b; }}
        {{ float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; }}
        {{ float a = v[1], b = v[3]; v[1] = a + b; v[3] = a - b; }}
#pragma clang loop unroll(full)
        for (uint s = 2u; s < 7u; ++s) {{
            uint shl = 1u << (s - 2u);
#pragma clang loop unroll(full)
            for (uint jj = 0u; jj < 4u; ++jj) {{
                float mine = v[jj];
                float other = simd_shuffle_xor(mine, shl);
                v[jj] = (lane & shl) ? (other - mine) : (mine + other);
            }}
        }}
        float RS = {rs!r}f;
        float w = (float)((half)w16[r]);
        float d0 = (float)((half)(v[0] * RS * rout[rbase + 0u]));
        float d1 = (float)((half)(v[1] * RS * rout[rbase + 1u]));
        float d2 = (float)((half)(v[2] * RS * rout[rbase + 2u]));
        float d3 = (float)((half)(v[3] * RS * rout[rbase + 3u]));
        acc[0] += d0 * w; acc[1] += d1 * w; acc[2] += d2 * w; acc[3] += d3 * w;
    }}
{chr(10).join(stores)}
"""


@lru_cache(maxsize=None)
def _scaled_had_out_sum_kernel(rs: float, top_k: int):
    return mx.fast.metal_kernel(
        name=f"escha_scaled_had_out_sum_tk{top_k}",
        input_names=["mid", "rout", "row_expert", "w16"],
        output_names=["y"],
        source=_scaled_had_out_sum_source(rs, top_k),
    )


def use_dn_sum() -> bool:
    """Fuse the down-leg output-Hadamard + score-weighted token sum into one
    kernel (ESCHA_MLX_DN_SUM=0 disables).  Replaces scaled_had_out(dn),
    d16.astype(f32)*w16, reshape and sum per layer with one launch.  Only
    valid on the token-major GEMV path (groups is None).  Default ON on the
    GEMV path: strictly fewer launches (one kern/launch does the butterfly,
    score product and fixed-order sum); measured a small decode step win and
    bit-identical (logit hash unchanged).  Never affects prefill (that path
    uses row-blocked groups)."""
    return os.environ.get("ESCHA_MLX_DN_SUM", "1") == "1"


def scaled_had_out_sum(mid: mx.array, rout: mx.array, row_expert: mx.array,
                       w16: mx.array, top_k: int, rs: float) -> mx.array:
    """mid [m, OC] f32 -> y [t= m/top_k, OC] f16, score-weighted per token."""
    m, oc = mid.shape
    t = m // top_k
    if use_had_pack():
        kern = _scaled_had_out_sum_pack_kernel(float(rs), int(top_k))
        nblk = oc // 128
        tgx = (nblk + 7) // 8
        (y,) = kern(
            inputs=[mid, rout, row_expert, w16],
            template=[("OC", oc)],
            grid=(256 * tgx, t, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(t, oc)],
            output_dtypes=[mx.float16],
        )
        return y
    kern = _scaled_had_out_sum_kernel(float(rs), int(top_k))
    tg = had_tg()
    (y,) = kern(
        inputs=[mid, rout, row_expert, w16],
        template=[("OC", oc)],
        grid=_had_grid(oc, t, tg),
        threadgroup=(tg, 1, 1),
        output_shapes=[(t, oc)],
        output_dtypes=[mx.float16],
    )
    return y


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


@lru_cache(maxsize=None)
def _moe_gemv_had_kernel(K: int, lut: bool, rs: float):
    # x is indexed through row_token[row]: the caller passes the raw per-token
    # activations and this kernel folds the xf[row_token] gather into the
    # transform, deleting one [m, IC] copy and one launch per leg.
    inputs = ["x", "rin", "code", "row_expert", "row_token"] + (["lut"] if lut else [])
    return mx.fast.metal_kernel(
        name=f"escha_moe_gemv_had_k{K}{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["mid"],
        header=_HEADER,
        source=_moe_gemv_had_source(K, lut, rs),
    )


@lru_cache(maxsize=None)
def _moe_gemv_had_kb_kernel(K: int, lut: bool, rs: float, KB: int):
    inputs = ["x", "rin", "code", "row_expert", "row_token"] + (["lut"] if lut else [])
    return mx.fast.metal_kernel(
        name=f"escha_moe_gemv_had_k{K}_kb{KB}{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["mid"],
        header=_HEADER,
        source=_moe_gemv_had_kb_source(K, lut, rs, KB),
    )


def had_kb() -> int:
    """kt-tiles prefetched per block in the fused moe_gemv_had direct loop
    (ESCHA_MLX_HAD_KB; default 8 on the M4 Max, 0 disables = per-kt fetch).

    The code-word offsets depend on lane only, so a KB block can be issued as
    independent loads before any is consumed -- KB-deep memory-level
    parallelism on the serial kt chain.  Isolated async gu leg 49us with the
    loads removed by constant-folding, so the code stream is ~55% of the leg;
    prefetch is the cheap way to recover it without changing bit identity.
    Must divide TK (gate_up 128, down 32)."""
    return int(os.environ.get("ESCHA_MLX_HAD_KB", "8") or 0)


@lru_cache(maxsize=None)

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



def use_gemv_had() -> bool:
    """Fuse the input Hadamard transform into the decode GEMV.

    ESCHA_MLX_GEMV_HAD=0 disables (the separate scaled_had + moe_gemv
    path).  One launch per leg instead of two, no [m,IC] intermediate.
    Only applies to the direct (non row-blocked) GEMV path — i.e. the
    bs1 decode row counts that dominate this metric.

    Default ON on the 40-core M4 Max: with BOTH legs fused (gate_up and
    down, see EschaSparseMoeBlock._expert_path) this is a measured decode
    win (+4%) and bit-identical, unlike the earlier gu-only fusion which
    was a wash at B<=2."""
    return os.environ.get("ESCHA_MLX_GEMV_HAD", "1") != "0"


def moe_gemv_had(x: mx.array, rin: mx.array, code_u32: mx.array,
                 row_expert: mx.array, row_token: mx.array,
                 K: int, IC: int, OC: int, rs: float) -> mx.array:
    """Fused input-transform + expert GEMV: mid = xh @ decode(W) with
    xh = f16(H128(f32(x[row_token[row]])*rin[e])*RS), computed in one kernel
    per leg.  The xf[row_token] gather is folded into the transform (x is the
    raw per-token activations, row_token maps each trellis row to its token).

    xf [T, IC] f16 (raw tokens), rin [E, IC] f32, code [E, TK, TN, 8K] u32,
    row_expert [m] i32, row_token [m] i32 -> mid [m, OC] f32.
    Bit-identical to gather-buffer=[xh] then
    scaled_had(xh, rin, re, RS) then moe_gemv(xh, code, re, K, IC, OC)."""
    m = row_token.shape[0]
    tk, tn = IC // 16, OC // 16
    assert IC % 256 == 0, f"IC must be a multiple of 256, got {IC}"
    kb = had_kb()
    if kb > 1 and tk % kb == 0:
        kern = _moe_gemv_had_kb_kernel(K, use_lut(), float(rs), kb)
    else:
        kern = _moe_gemv_had_kernel(K, use_lut(), float(rs))
    inputs = [x, rin, code_u32.reshape(-1), row_expert, row_token]
    if use_lut():
        inputs.append(_lut_array())
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC)],
        grid=(256 * (OC // 128), m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, OC)],
        output_dtypes=[mx.float32],
    )
    return mid


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

    DEFAULT ON (ESCHA_MLX_PREFETCH=0 disables).  Measured a wash on the 10-core
    M4 (doc §15.7) but a consistent +6% prefill on the 40-core M4 Max, where the
    wider GPU exposes the one-outstanding-load-per-lane stall the commit
    originally targeted: interleaved fresh-process prefill at ISL=512 (eval
    metric) 1037 -> 1102 tok/s mean, prefetch ahead of per-kt fetch in all three
    pairs (bench/results/m4-max-64gb, 2026-08-12).  Bit-identical -- the same
    values are computed, just issued as independent loads before the barrier.
    """
    return os.environ.get("ESCHA_MLX_PREFETCH", "1") != "0"


def use_sortx() -> bool:
    """Pre-sort x so a group's rows are consecutive (default off; ESCHA_MLX_SORTX=1 enables).

    Borrowed from mlx-lm's SwitchGLU, which physically permutes x via
    `_gather_sort` before calling the fused `gather_qmm` -- the kernel §15
    credited with ~80% of roofline against our 39-53%, a comparison §16
    retracted as whole-step misattribution (the transform pipeline was being
    charged to the kernel; isolated at matched shapes, ours is faster than
    `gather_qmm` at every decode row count, §16.1).  Our kernel
    instead chased `rows_idx[grp*R+rr]` for every staged element, TK times per
    group; with x pre-sorted the row address is just `src_row0 + rr`.

    Costs one permute of m rows per leg; the OUTPUT write keeps the indirection,
    so no un-permute is needed.

    DEFAULT OFF -- measured a wash (doc §15.4): prefill -0.8%/-0.0%, decode
    -1.7% at B=16 and -0.5% at B=32, all inside an A/B/A drift band of the same
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
