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


def _code_base(wpt: int, dense: bool) -> str:
    """Prologue resolving the code-stream base pointer for one row.

    This is the ONLY structural difference between the MoE and the dense GEMV
    kernels: a MoE row selects one of E stacked streams through ``row_expert``,
    a dense row always uses the single stream this linear owns.  Sharing the
    rest of the source verbatim keeps one codec engine — the decode, the
    accumulation order and the store are the same instructions either way, so
    the dense kernel inherits every gate the MoE kernel already passes.

    Dropping the indirection also drops a dependent device load from the
    per-row prologue and frees the kernel from carrying a row_expert buffer.
    """
    if dense:
        return "    device const uint* base = code;\n"
    return (f"    int e = row_expert[row];\n"
            f"    device const uint* base = code + (ulong)e * ((ulong)TK * TN * {wpt}u);\n")


def _scale_offsets(dim: str, dense: bool) -> str:
    """Prologue resolving the row and per-channel scale offsets for one row.

    Same split as ``_code_base``: the MoE transforms gather rin/rout from an
    E-stacked [E, dim] vector, the dense ones read the single [dim] vector this
    linear owns.  Emitted as one block (rather than only the scale offset) so
    the MoE text stays byte-identical to the pre-dense kernel.
    """
    row_off = f"    ulong off = (ulong)row * {dim} + (ulong)blk * 128u + tid;\n"
    if dense:
        return row_off + "    ulong roff = (ulong)blk * 128u + tid;\n"
    return ("    int e = row_expert[row];\n" + row_off
            + f"    ulong roff = (ulong)e * {dim} + (ulong)blk * 128u + tid;\n")


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


def _moe_gemv_direct_source(K: int, use_lut: bool, shuffle: bool = False,
                            dense: bool = False, pf: int = 1) -> str:
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

    ``pf`` > 1 unrolls the kt loop and hoists the code-word fetches of pf
    consecutive tiles ahead of their decode+accumulate bodies, so each lane
    keeps 2*pf loads in flight instead of 2.  The kernel is latency-bound at
    bs1, and the next tile's address is affine in kt -- independent of the
    current decode -- so the only thing stopping deeper pipelining is that the
    single-tile loop consumes each word the instruction after it loads.  The
    bodies still run in ascending kt order with the same j order, so the f32
    accumulation order -- and every output bit -- is unchanged (the same
    argument, and the same fetch-substitution helper, as the row-blocked GEMM's
    prefetch variant).  Requires TK % pf == 0; the wrapper falls back to pf=1
    otherwise.

    Accumulation order per (row, out-channel) is unchanged => bit-identical.
    """
    wpt = 8 * K
    raw = _EXTRACT_K2 if K == 2 else _EXTRACT_K3
    # The 8 x reads per lane-tile cover only 4 distinct halves (xrow, xrow+1,
    # xrow+8, xrow+9), each used by one acc0 and one acc1 term.  Load them as
    # two half2 vectors -- xrow is even, so both addresses are 4B-aligned --
    # and feed the SAME values to the FMAs in the SAME j order: a pure
    # load-merge, bit-identical.  Measured ~8% of the m=1 kernel: redundant
    # scalar loads all hit one cache line but still occupy issue slots.
    xv = ("        half2 xv0 = *(device const half2*)(xrowp + xrow);\n"
          "        half2 xv1 = *(device const half2*)(xrowp + xrow + 8u);")
    _XCOMP = {0: "xv0.x", 1: "xv0.y", 8: "xv1.x", 9: "xv1.y"}
    accs = [xv]
    for j in range(8):
        fi = j >> 1
        xo = (j & 1) + (fi & 1) * 8
        acc = "acc0" if j < 4 else "acc1"
        accs.append(f"        {acc} += (float){_XCOMP[xo]} * (float){_dec(f's{j}', use_lut)};")
    if pf > 1:
        extract = _substitute_fetch_prefetch(raw, K)
        pf_idx = _PF_IDX_K2 if K == 2 else _PF_IDX_K3
        pf_b = "pi1" if K == 2 else "pi2"
        loop = f"""
{pf_idx}
    for (uint kb = 0; kb < TK; kb += {pf}u) {{
        uint pfa[{pf}], pfb[{pf}];
#pragma clang loop unroll(full)
        for (uint u = 0; u < {pf}u; ++u) {{
            device const uint* wpk = base + ((ulong)(kb + u) * TN + ocb * 8u + sg) * {wpt}u;
            pfa[u] = wpk[pi0];
            pfb[u] = wpk[{pf_b}];
        }}
#pragma clang loop unroll(full)
        for (uint k2 = 0; k2 < {pf}u; ++k2) {{
            device const half* xrowp = xbase + (kb + k2) * 16u;
{extract}
{chr(10).join('    ' + a for a in accs)}
        }}
    }}"""
    else:
        extract = (_substitute_fetch_shuffle(raw) if shuffle
                   else _substitute_fetch(raw))
        fetch = (f"        uint myw = (lane < {wpt}u) ? wp[lane] : 0u;\n"
                 if shuffle else "")
        loop = f"""
    for (uint kt = 0; kt < TK; kt++) {{
        device const uint* wp = base + ((ulong)kt * TN + ocb * 8u + sg) * {wpt}u;
        device const half* xrowp = xbase + kt * 16u;
{fetch}{extract}
{chr(10).join(accs)}
    }}"""
    return f"""
    uint ocb = thread_position_in_grid.x >> 8;
    uint row = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

{_code_base(wpt, dense)}    device const half* xbase = xh + (ulong)row * IC;

    float acc0 = 0.0f, acc1 = 0.0f;
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xrow = (lane & 3u) * 2u;
{loop}

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


def _moe_gemv_splitk_source(K: int, use_lut: bool, shuffle: bool, S: int,
                            dense: bool = False) -> str:
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

{_code_base(wpt, dense)}    device const half* xbase = xh + (ulong)row * IC;

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


def _moe_gemv_source(K: int, use_lut: bool, dense: bool = False) -> str:
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

{_code_base(wpt, dense)}
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


def _gemm_group_prologue(R: int, dense: bool) -> str:
    """Group-validity prologue for the row-blocked GEMM.

    A MoE group past the end is marked by expert -1; a dense group past the end
    is simply one whose first row is beyond M.
    """
    if dense:
        # (uint)M for the same reason as the row clamp: the template parameter
        # is emitted as an int, and a signed/unsigned compare is a warning at
        # best and a surprise at worst.
        return f"    if (grp * {R}u >= (uint)M) return;\n"
    return "    int e = group_expert[grp];\n    if (e < 0) return;\n"


def _moe_gemm_rows_source(K: int, use_lut: bool, R: int, KB: int = 1,
                          sortx: bool = False, prefetch: bool = False,
                          dense: bool = False) -> str:
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

    ``dense=True`` drops both buffers. A dense linear has one stream, so every
    row belongs to it and group grp simply owns rows [grp*R, grp*R+R) -- the
    membership question the MoE grouping machinery exists to answer does not
    arise, and neither does its padding: only the LAST group is partly padding,
    against up to R-1 wasted slots per expert in the MoE case. Rows past the end
    are clamped to M, the zero row the caller appends, so they stage zeros and
    write to a discarded row. The kt/j accumulation order is untouched, so a
    dense group is bit-identical to R dense R=1 GEMV calls.

    This kernel is what makes dense PREFILL viable at all: the per-row GEMV
    reads the entire coded weight once per row, so a 256-token chunk would
    decode every projection 256 times.
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
    # Hoist the 8 decodes, then run r-outer with two half2 threadgroup loads
    # per row: the 8 scalar s_x reads per (lane, r) cover 4 distinct halves at
    # even offsets (xrow, +1, +8, +9), so they merge into 2 vector loads --
    # 8R -> 2R threadgroup accesses per tile.  Loop order changed from j-outer
    # to r-outer, but each accumulator acc{0,1}[r] still receives its four
    # terms in ascending-j order, so every f32 sum -- and every output bit --
    # is unchanged.  (xb, r*16 and xrow are all even => half2 alignment holds.)
    decs = "\n".join(f"            float d{j} = (float){_dec(f's{j}', use_lut)};"
                     for j in range(8))
    accs = [f"""
{decs}
#pragma clang loop unroll(full)
            for (uint r = 0; r < {R}u; ++r) {{
                threadgroup const half2* xp =
                    (threadgroup const half2*)(s_x + xb + r * 16u + xrow);
                half2 xv0 = xp[0];
                half2 xv1 = xp[4];
                acc0[r] += (float)xv0.x * d0;
                acc0[r] += (float)xv0.y * d1;
                acc0[r] += (float)xv1.x * d2;
                acc0[r] += (float)xv1.y * d3;
                acc1[r] += (float)xv0.x * d4;
                acc1[r] += (float)xv0.y * d5;
                acc1[r] += (float)xv1.x * d6;
                acc1[r] += (float)xv1.y * d7;
            }}"""]
    # MoE row addressing, shared by both expert variants; the dense branch
    # replaces it because a dense group's rows are an arithmetic range.
    store_guard = "if ((lane & 3u) == 0u)"
    row_of = "rows_idx[grp * %du + %s]" % (R, "{i}")
    if dense:
        # Row address is computed, not loaded: group grp owns rows grp*R..+R.
        # (uint)M: the template parameter is emitted as an int, and MSL's min()
        # overloads are ambiguous across signedness.
        #
        # No padding sink row.  The MoE contract has the CALLER append a zero
        # row at index M and routes padding slots to it; here the tail group's
        # padding slots stage a real row instead (harmless -- their accumulators
        # are never stored) and the STORE is guarded.  That removes a full
        # [m, IC] copy per call, which the dense path would otherwise pay at
        # every one of ~400 coded linears on every forward: at a 2048-row
        # prefill through a 17408-wide leg that is ~71 MB copied in plus a
        # non-contiguous slice copied out, per linear.  It lands exactly on the
        # prefill path this kernel exists to make viable, and being independent
        # of R it would bias any rows-per-group sweep against blocking.
        row_of = "(grp * %du + %s)" % (R, "{i}")
        store_guard = "if ((lane & 3u) == 0u && grp * %du + r < (uint)M)" % R
        stage_body = ("            s_x[i] = xh[(ulong)min(%s, (uint)M - 1u) * IC\n"
                      "                        + (kb + kk) * 16u + cc];"
                      % row_of.format(i="rr"))
    elif sortx:
        # Sorted-x staging: rows of a group are CONSECUTIVE in xs, so the
        # address is computed (src_row0 + rr) rather than loaded from rows_idx
        # and chased.  The values staged are identical -- padding contributes 0
        # either way -- so the output is bit-identical.  The OUTPUT write keeps
        # the rows_idx indirection: it happens once per group while staging
        # happens TK times, so scattering on the write side is far cheaper than
        # un-permuting the whole mid tensor.
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

{_gemm_group_prologue(R, dense)}
    threadgroup half s_x[{16 * R * KB}];

    device const uint* base = {"code" if dense else f"code + (ulong)e * ((ulong)TK * TN * {wpt}u)"};

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
        {store_guard} {{
            uint col = 2u * (l0 >> 3) + c_off;
            ulong ob = (ulong){row_of.format(i="r")} * OC + ocb * 128u + sg * 16u;
            mid[ob + col] = a0;
            mid[ob + col + 8u] = a1;
        }}
    }}
"""


def _scaled_had_source(rs: float, dense: bool = False) -> str:
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

{_scale_offsets('IC', dense)}
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


def _scaled_had_out_source(rs: float, dense: bool = False) -> str:
    """Fused  f16( H128(mid) * RS * rout[e] )  in one kernel.

    Keep the two post-transform multiplies as separate f32 statements in the
    same left-to-right order as the native MLX chain.  That preserves both f32
    rounding points before the single final f16 cast.
    """
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint blk = thread_position_in_grid.x >> 7;
    uint row = thread_position_in_grid.y;

{_scale_offsets('OC', dense)}
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


# The kernel NAME is part of MLX's compiled-kernel identity, so every source
# variant must contribute to it.  A dense kernel compiled under the MoE name
# would be served from the cache to the MoE path (and vice versa) with no
# error — the signatures differ only by a buffer that is silently absent.
def _variant(dense: bool) -> str:
    return "_dense" if dense else ""


@lru_cache(maxsize=None)
def _scaled_had_kernel(rs: float, dense: bool = False):
    return mx.fast.metal_kernel(
        name=f"escha_scaled_had{_variant(dense)}",
        input_names=["rows", "rin"] + ([] if dense else ["row_expert"]),
        output_names=["out"],
        source=_scaled_had_source(rs, dense),
    )


@lru_cache(maxsize=None)
def _scaled_had_out_kernel(rs: float, dense: bool = False):
    return mx.fast.metal_kernel(
        name=f"escha_scaled_had_out{_variant(dense)}",
        input_names=["mid", "rout"] + ([] if dense else ["row_expert"]),
        output_names=["out"],
        source=_scaled_had_out_source(rs, dense),
    )


def use_fused_had() -> bool:
    """Fused expert Hadamard transforms (ESCHA_MLX_FUSED_HAD=0 disables)."""
    return os.environ.get("ESCHA_MLX_FUSED_HAD", "1") != "0"


def scaled_had(rows: mx.array, rin: mx.array, row_expert: mx.array | None,
               rs: float) -> mx.array:
    """rows [m, IC] f16, rin [E, IC] f32, row_expert [m] i32 -> [m, IC] f16.

    ``row_expert=None`` selects the dense variant: rin is the single [IC]
    vector this linear owns and no per-row gather is issued.
    """
    m, ic = rows.shape
    dense = row_expert is None
    kern = _scaled_had_kernel(float(rs), dense)
    (out,) = kern(
        inputs=[rows, rin] + ([] if dense else [row_expert]),
        template=[("IC", ic)],
        grid=(128 * (ic // 128), m, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, ic)],
        output_dtypes=[mx.float16],
    )
    return out


def scaled_had_out(mid: mx.array, rout: mx.array, row_expert: mx.array | None,
                   rs: float) -> mx.array:
    """mid [m, OC] f32, rout [E, OC] f32 -> transformed [m, OC] f16.

    ``row_expert=None`` selects the dense variant (rout is a single [OC] vector).
    """
    m, oc = mid.shape
    dense = row_expert is None
    kern = _scaled_had_out_kernel(float(rs), dense)
    (out,) = kern(
        inputs=[mid, rout] + ([] if dense else [row_expert]),
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
def _moe_gemv_kernel(K: int, lut: bool, direct: bool = False, shuffle: bool = False,
                     dense: bool = False, pf: int = 1):
    inputs = (["xh", "code"] + ([] if dense else ["row_expert"])
              + (["lut"] if lut else []))
    src = (_moe_gemv_direct_source(K, lut, shuffle, dense, pf) if direct
           else _moe_gemv_source(K, lut, dense))
    # pf is baked into the source, so it MUST be in the name: MLX keys its
    # compiled-kernel cache by name, and a collision would silently serve the
    # wrong kernel.
    tag = (("_direct" if direct else "")
           + ("_shf" if direct and shuffle and pf == 1 else "")
           + (f"_pf{pf}" if direct and pf > 1 else ""))
    stem = "escha_gemv_dense" if dense else "escha_moe_gemv"
    return mx.fast.metal_kernel(
        name=f"{stem}_k{K}{tag}{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["mid"],
        header=_HEADER,
        source=src,
    )


@lru_cache(maxsize=1)
def gemv_pf() -> int:
    """kt-tiles whose code words are fetched ahead in the direct GEMV.

    Resolved ONCE, not per call: this one needs an int parse and a range check,
    and `moe_gemv` runs it on every coded linear of every forward (~400 times
    per token in the 27B). Caching keeps the parse off the hot path and makes a
    malformed value fail on the first forward rather than raising from inside
    the generation loop -- the same reasoning `dense_block_r_pin` gives. The
    consequence is that flipping it mid-process does nothing; nothing sweeps
    it, and a test that varies it must call `gemv_pf.cache_clear()`.

    ESCHA_MLX_GEMV_PF, default 1 (today's single-tile loop).  Values > 1 keep
    2*pf code loads in flight per lane; bit-identical at every depth (only load
    scheduling changes -- see _moe_gemv_direct_source).  Applied per linear
    only when TK divides evenly; ignored by the staged/shuffle variants.

    Measured a wash on M4 base (in-model, medians, A/B/A: pf=1 6.76 tok/s,
    pf=2 6.57, pf=4 6.58, pf=8 6.61 -- all within the drift band and none
    ahead).  Kept because the premise -- more loads in flight on a
    latency-bound kernel -- is hardware-dependent, and this is the cheapest
    lever to re-test on a wider GPU.
    """
    env = os.environ.get("ESCHA_MLX_GEMV_PF")
    if not env:
        return 1
    pf = int(env)
    if pf < 1:
        raise ValueError(f"ESCHA_MLX_GEMV_PF must be >= 1, got {pf}")
    return pf


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
def _moe_gemv_splitk_kernel(K: int, lut: bool, shuffle: bool, S: int,
                            dense: bool = False):
    inputs = (["xh", "code"] + ([] if dense else ["row_expert"])
              + (["lut"] if lut else []))
    stem = "escha_gemv_dense" if dense else "escha_moe_gemv"
    return mx.fast.metal_kernel(
        name=f"{stem}_k{K}_sk{S}{'_shf' if shuffle else ''}"
             f"{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["part"],
        header=_HEADER,
        source=_moe_gemv_splitk_source(K, lut, shuffle, S, dense),
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
                          sortx: bool = False, prefetch: bool = False,
                          dense: bool = False):
    if dense:
        idx_inputs = []           # row addresses are computed, not looked up
    elif sortx:
        idx_inputs = ["src_row0", "n_valid", "rows_idx", "group_expert"]
    else:
        idx_inputs = ["rows_idx", "group_expert"]
    inputs = ["xh", "code"] + idx_inputs + (["lut"] if lut else [])
    stem = "escha_gemm_dense" if dense else "escha_moe_gemm"
    return mx.fast.metal_kernel(
        name=f"{stem}_k{K}_r{R}_kb{KB}"
             f"{'_sx' if sortx and not dense else ''}{'_pf' if prefetch else ''}"
             f"{'_lut' if lut else ''}",
        input_names=inputs,
        output_names=["mid"],
        header=_HEADER,
        source=_moe_gemm_rows_source(K, lut, R, KB, sortx, prefetch, dense),
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


def moe_gemv(xh: mx.array, code_u32: mx.array, row_expert: mx.array | None,
             K: int, IC: int, OC: int, splits: int | None = None) -> mx.array:
    """Fused expert GEMV: mid[r, :] = xh[r, :] @ decode(code[row_expert[r]]).

    xh: [M, IC] f16 (already input-transformed per row).
    code_u32: [E, TK, TN, 8K] uint32 (E-stacked, as loaded), or [TK, TN, 8K]
        for a dense linear.
    row_expert: [M] int32, or None for a dense linear — every row then reads
        the single stream and the expert indirection is compiled out.
    splits: kt-split factor; None = the size policy, 1 = force the sequential
        kernel.  Pass 1 when you need the result to be bit-identical to the
        row-blocked GEMM (split-K reassociates the sum -- see
        _moe_gemv_splitk_source).
    Returns mid [M, OC] f32 (pre output-transform).
    """
    m = xh.shape[0]
    tk, tn = IC // 16, OC // 16
    lut = use_lut()
    dense = row_expert is None
    inputs = ([xh, code_u32.reshape(-1)] + ([] if dense else [row_expert])
              + ([_lut_array()] if lut else []))

    if splits is not None:
        s = splits if splits > 1 and tk % splits == 0 else 1
    else:
        s = split_k_for(m, OC, tk) if use_direct() else 1
    if s > 1:
        kern = _moe_gemv_splitk_kernel(K, lut, use_shuffle(), s, dense)
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

    # pf only exists in the direct source, and its prefetch substitution
    # replaces the same W(i) the shuffle substitution would have; the two are
    # mutually exclusive by construction, so shuffle wins and pf falls back.
    pf = gemv_pf()
    if tk % pf or use_shuffle() or not use_direct():
        pf = 1
    kern = _moe_gemv_kernel(K, lut, use_direct(), use_shuffle(), dense, pf)
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC)],
        grid=(256 * (OC // 128), m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, OC)],
        output_dtypes=[mx.float32],
    )
    return mid


def dense_scaled_had(rows: mx.array, rin: mx.array, rs: float) -> mx.array:
    """f16( H128(rows * rin) * RS ) for a dense linear. rin [IC] f32."""
    return scaled_had(rows, rin, None, rs)


def dense_scaled_had_out(mid: mx.array, rout: mx.array, rs: float) -> mx.array:
    """f16( H128(mid) * RS * rout ) for a dense linear. rout [OC] f32."""
    return scaled_had_out(mid, rout, None, rs)


def dense_gemv(xh: mx.array, code_u32: mx.array, K: int, IC: int, OC: int,
               splits: int | None = None) -> mx.array:
    """Fused dense GEMV: mid = xh @ decode(code). code_u32 [TK, TN, 8K] uint32.

    ``splits`` is carried for parity with ``moe_gemv`` and is not selected by
    anything today: split-K measured negative on M4 (see ``split_k_for``), so
    every caller takes the default.
    """
    return moe_gemv(xh, code_u32, None, K, IC, OC, splits)


#: Largest rows-per-group the dense row-blocked GEMM will choose on its own.
#: Measured on M4 base (24 GB, Qwen3.8-27B W2, ISL 512, chunk 256, medians):
#: R=4 25.8 / R=8 33.3 / R=16 36.6 / R=20 22.1 / R=24 32.6 / R=32 24.0
#: prefill tok/s -- 16 wins, 32 falls off a register/occupancy cliff, and
#: non-power-of-two R is always worse than its pow2 neighbor.  Bit-identity
#: at R=16/32 incl. partial tails is gated (bench/p0_gates.py G0.2b and the
#: high-R check in tests).  See dense_block_r for the policy.
DENSE_R_MAX = 16


def dense_block_r_pin() -> int | None:
    """Resolve ESCHA_MLX_DENSE_BLOCK_R once. None = use the size policy.

    Read at module construction, not per forward: every other knob in this
    runtime is latched the same way, so that flipping an env var mid-process
    cannot make an A/B depend on call order, and a malformed value fails at
    load instead of from inside the hot loop.
    """
    env = os.environ.get("ESCHA_MLX_DENSE_BLOCK_R")
    if not env:
        return None
    r = int(env)
    if r < 1:
        raise ValueError(f"ESCHA_MLX_DENSE_BLOCK_R must be >= 1, got {r}")
    return r


def dense_block_r(m: int, pin: int | None = None) -> int:
    """Rows per group for the dense row-blocked GEMM (1 = the per-row GEMV).

    The dense case is the one the MoE policy in EschaSparseMoeBlock._blocked_R
    could not have: there, R rows must share an EXPERT, so a group is mostly
    padding until the row count is large relative to the expert count, and the
    measured optimum grows slowly and turns harmful early.  Here every row
    shares the single stream, so only the last group is ever partly padding and
    the decoded-stream reads fall by very nearly R across the board.

    So the policy is simply R = min(m, R_MAX), snapped DOWN to a power of two.
    Two consequences worth stating, because the first version of this function
    used a size ladder inherited from the MoE's thresholds and both were wrong:

      * It never pads.  R <= m always, so no group issues MACs for rows that do
        not exist -- the padding tax that shapes the MoE policy is absent here
        by construction, not merely small.
      * It does not treat small row counts as decode-only.  A ladder that
        returned R=1 below m=4 meant a server at concurrency 2 or 3 read the
        entire coded stream two or three times per token, which is precisely
        the cost this kernel exists to remove.  Batch is rows; rows share the
        stream; share them.

    Snapping to a power of two bounds the compiled-kernel count to R in
    {1, 2, 4, 8, 16} -- R is a template parameter, so every distinct value is a
    separate compilation -- at the price of at most halving the sharing.

    R_MAX = 16 is MEASURED on M4 base (see DENSE_R_MAX): the prefill optimum,
    with a hard falloff at 32 (acc0[R] + acc1[R] fp32 accumulators per lane hit
    a register/occupancy cliff) and non-power-of-two values always losing to
    their pow2 neighbor.  ESCHA_MLX_DENSE_BLOCK_R still pins R for re-sweeps on
    other machines (resolved once by ``dense_block_r_pin`` and passed in).
    R=1 is always correct and is what m=1 uses.
    """
    if pin is not None:
        return pin
    r = min(m, DENSE_R_MAX)
    while r & (r - 1):                      # snap down to a power of two
        r &= r - 1
    return max(1, r)


_SGMAT_HEADER = ("#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n"
                 "using namespace metal;\n" + _HEADER)


def _dense_gemm_mat_source(K: int, use_lut: bool) -> str:
    """R=16 dense GEMM whose inner product is 8 simdgroup MMAs per 16x16 tile.

    NOT bit-identical to the scalar kernels -- deterministic, but the sum is
    reassociated.  It is NOT, however, a precision downgrade: probed on M4,
    ``simdgroup_multiply_accumulate`` with half operands and a float8x8
    accumulator computes the product at FULL precision ((1+2^-10)^2 returns
    exact) and accumulates in f32 (2048 additions of 2^-12 return exact).  A
    half x half product needs 22 mantissa bits and f32 has 24, so it is exact
    by construction.  The deviation is therefore f32 reassociation only --
    measured max ~4e-5 and mean ~1e-6 relative to mean |output|, i.e. an order
    of magnitude BELOW the fold_scales deviation this runtime already ships.
    Same class as the split-K path: deterministic, tolerance-gated.

    The matrix units cannot read per-lane registers, so decoded weights make a
    threadgroup round-trip that the scalar kernel avoids; that is the cost this
    trades against 16x fewer FMA instructions.  Accumulators also shrink from
    2R=32 floats per lane to 8, which is why R is fixed at 16 here.
    """
    wpt = 8 * K
    extract = _substitute_fetch(_EXTRACT_K2 if K == 2 else _EXTRACT_K3)
    stores = []
    for j in range(8):
        fi = j >> 1
        row = f"(lane & 3u) * 2u + {j & 1}u + {(fi & 1) * 8}u"
        col = f"2u * ((l0 >> 3) + {4 if j >= 4 else 0}u) + c_off"
        stores.append(f"        s_w[sg * 256u + ({row}) * 16u + ({col})] = "
                      f"{_dec(f's{j}', use_lut)};")
    return f"""
    uint tid  = thread_position_in_threadgroup.x;
    uint ocb  = thread_position_in_grid.x >> 8;
    uint grp  = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;
    if (grp * 16u >= (uint)M) return;

    threadgroup half s_x[256];
    threadgroup half s_w[2048];
    threadgroup float s_c[256];      // tail-group store staging (see below)

    simdgroup_float8x8 C00 = simdgroup_float8x8(0.0f), C01 = simdgroup_float8x8(0.0f),
                       C10 = simdgroup_float8x8(0.0f), C11 = simdgroup_float8x8(0.0f);
    uint l0 = lane & ~4u;
    uint c_off = (lane >> 2) & 1u;
    uint xr = tid >> 4, xc = tid & 15u;
    // Tail rows clamp to the last real row; their stores are guarded below.
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

    // A full group stores straight from the fragments.  A tail group cannot:
    // simdgroup_store writes all 8 rows of a fragment, so rows past M would be
    // written.  It stages through s_c instead and guards the row -- one
    // simdgroup at a time, since 8 x 256 floats will not fit in threadgroup
    // memory and this runs once per group, not once per kt.  `grp` and `M` are
    // uniform, so every thread takes the same branch and the barriers below are
    // reached by the whole threadgroup.
    ulong obase = (ulong)(grp * 16u) * OC + ocb * 128u + sg * 16u;
    if (grp * 16u + 16u <= (uint)M) {{
        device float* o = mid + obase;
        simdgroup_store(C00, o, OC);            simdgroup_store(C01, o + 8u, OC);
        simdgroup_store(C10, o + 8u * OC, OC);  simdgroup_store(C11, o + 8u * OC + 8u, OC);
    }} else {{
        for (uint wv = 0; wv < 8u; ++wv) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == wv) {{
                simdgroup_store(C00, s_c, 16);          simdgroup_store(C01, s_c + 8u, 16);
                simdgroup_store(C10, s_c + 128u, 16);   simdgroup_store(C11, s_c + 136u, 16);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == wv) {{
                for (uint i = lane; i < 256u; i += 32u) {{
                    uint rr = i >> 4, cc = i & 15u;
                    if (grp * 16u + rr < (uint)M)
                        mid[obase + (ulong)rr * OC + cc] = s_c[i];
                }}
            }}
        }}
    }}
"""


@lru_cache(maxsize=None)
def _dense_gemm_mat_kernel(K: int, lut: bool):
    return mx.fast.metal_kernel(
        name=f"escha_gemm_dense_sgmat_k{K}{'_lut' if lut else ''}",
        input_names=["xh", "code"] + (["lut"] if lut else []),
        output_names=["mid"],
        header=_SGMAT_HEADER, source=_dense_gemm_mat_source(K, lut),
    )


#: Rows per group the simdgroup-matrix GEMM is built for.  Not a policy knob:
#: the kernel's fragment shape IS 16 rows (two 8x8 A fragments), so this is a
#: property of the source, not a tunable.  ``EschaLinear`` compares the R the
#: size policy chose against this, so pinning a different R falls back to the
#: scalar kernel at that R instead of silently getting 16-row blocking.
DENSE_MAT_R = 16


def use_dense_mat() -> bool:
    """simdgroup-matrix dense GEMM. DEFAULT OFF -- deterministic, NOT bit-identical.

    Enabled with ESCHA_MLX_DENSE_MAT=1.  Off by default for the same reason
    split-K is: every other kernel in this runtime is bit-identical to the
    goldens, and a path that reassociates the sum cannot be A/B'd against them
    with np.array_equal.  See _dense_gemm_mat_source for why the reassociation
    is the ONLY difference (the MMA product is exact on M4).

    Measured on M4 base, 27B dense, in-model A/B/A/B on a quiet machine:

        prefill ISL 512    38.9 -> 45.5 tok/s  (+17.0%)
        prefill ISL 2048   38.0 -> 44.1 tok/s  (+16.0%)
        decode  bs1         7.05 -> 7.01 tok/s (noise -- see below)

    Decode is untouched by construction: at bs1 the row count is 1, the size
    policy returns R=1, and this kernel is never reached.  It is a prefill/TTFT
    lever only.  Reproduce with
    ``bench/prefill_profile.py --sweep-dense-mat``.

    The obvious next step does NOT pay here.  Swapping the float8x8 accumulator
    for half8x8 -- the variant that wins on GPUs whose matrix units run
    fp16-accumulate at roughly twice the fp32-accumulate rate -- measured 1.23x
    against 1.20x on the GEMM (a 3% gain) for a mean relative error of
    9.2e-3..1.7e-2 instead of 1.3e-6, four orders of magnitude worse and worst
    on the longest reduction (mlp.down, IC=17408), exactly the K-dependence an
    fp16 running sum predicts.  This GPU has no fp16-accumulate rate bonus, so
    that trade is not worth re-testing here; it is worth re-testing on hardware
    that does.
    """
    return os.environ.get("ESCHA_MLX_DENSE_MAT", "0") != "0"


def dense_gemm_mat(xh: mx.array, code_u32: mx.array, K: int, IC: int,
                   OC: int) -> mx.array:
    """Row-blocked dense GEMM at a fixed R=DENSE_MAT_R via simdgroup matrices.

    Deterministic but NOT bit-identical to ``dense_gemm_rows`` -- the f32 sum is
    reassociated.  Callers must gate on ``use_dense_mat()``.
    """
    m = xh.shape[0]
    lut = use_lut()
    kern = _dense_gemm_mat_kernel(K, lut)
    inputs = [xh, code_u32.reshape(-1)] + ([_lut_array()] if lut else [])
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", IC // 16), ("TN", OC // 16), ("IC", IC), ("OC", OC),
                  ("M", m)],
        grid=(256 * (OC // 128), (m + 15) // 16, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, OC)],
        output_dtypes=[mx.float32],
    )
    return mid


def dense_gemm_rows(xh: mx.array, code_u32: mx.array, K: int, IC: int, OC: int,
                    R: int) -> mx.array:
    """Row-blocked dense GEMM: mid = xh @ decode(code), R rows per decode.

    xh: [M, IC] f16 -- no padding row; the kernel guards its stores instead.
    Returns mid [M, OC] f32 -- bit-identical to ``dense_gemv`` on the same input
    (same kt/j accumulation order; only the loop nest around it differs).
    """
    m = xh.shape[0]
    if m < 1:
        raise ValueError("dense_gemm_rows needs at least one row")
    tk, tn = IC // 16, OC // 16
    lut = use_lut()
    kb = kt_block()
    if tk % kb:                      # a partial block would drop kt iterations
        kb = 1
    n_groups = (m + R - 1) // R
    kern = _moe_gemm_rows_kernel(K, lut, R, kb, False, use_prefetch(), True)
    inputs = [xh, code_u32.reshape(-1)] + ([_lut_array()] if lut else [])
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC), ("M", m)],
        grid=(256 * (OC // 128), n_groups, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, OC)],
        output_dtypes=[mx.float32],
    )
    return mid
