"""GPU-busy / frequency sampling around a kernel loop — the one profiling signal
available without Xcode.

WHY. Multiple kernel hypotheses were measured wrong (doc §11, §13.2, §15.4)
because they were reasoned from counters rather than observed. At the time,
whole-step measurements suggested the trellis GEMM sustained only 39-53% of the
bandwidth roofline vs `gather_qmm`'s ~80% (§15); §16 later showed that reading
mis-attributed transform-pipeline cost to the kernel — isolated at matched
shapes, the kernel is faster than `gather_qmm` at every decode row count.
Instruments would have answered why directly, but it needed full Xcode.

What powermetrics CAN answer is the first fork in the tree, and it is the fork
that decides where to look next:

  GPU busy ~100%  -> the GPU is saturated and simply not retiring enough work per
                     cycle: a stall/occupancy/ILP problem INSIDE the kernel.
  GPU busy < ~85% -> the GPU is idle part of the time: a dispatch/launch/
                     serialization problem OUTSIDE the kernel (op overhead,
                     barriers between MLX ops, command-buffer breaks).

Those two lead to completely different fixes, and nothing measured so far
distinguishes them.

Requires sudo (powermetrics is root-only):

    sudo python bench/gpu_busy.py --model <dir> [--arm trellis|dense]

`--arm dense` runs an equivalent-size mx.quantized_matmul loop as the control:
that path is known to hit ~97% of roofline (doc §1.1), so its GPU-busy reading
calibrates what "healthy" looks like on this box.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def sample_powermetrics(seconds: int, interval_ms: int = 500) -> dict:
    """Run powermetrics for `seconds` and parse GPU residency + frequency."""
    n = max(1, int(seconds * 1000 / interval_ms))
    try:
        out = subprocess.run(
            ["powermetrics", "--samplers", "gpu_power",
             "-i", str(interval_ms), "-n", str(n)],
            capture_output=True, text=True, timeout=seconds + 60)
        out = (out.stdout or "") + (out.stderr or "")
    except FileNotFoundError:
        return {"error": "powermetrics not found"}
    except subprocess.TimeoutExpired:
        return {"error": "powermetrics timed out"}

    busy = [float(x) for x in re.findall(r"GPU (?:HW )?active residency:\s+([\d.]+)%", out)]
    if not busy:
        busy = [float(x) for x in re.findall(r"GPU idle residency:\s+([\d.]+)%", out)]
        busy = [100.0 - b for b in busy]
    freq = [float(x) for x in re.findall(r"GPU (?:HW )?active frequency:\s+([\d.]+)", out)]
    pw = [float(x) for x in re.findall(r"GPU Power:\s+([\d.]+)\s*mW", out)]
    return {
        "n_samples": len(busy),
        "gpu_busy_pct_mean": round(sum(busy) / len(busy), 1) if busy else None,
        "gpu_busy_pct_max": round(max(busy), 1) if busy else None,
        "gpu_freq_mhz_mean": round(sum(freq) / len(freq), 1) if freq else None,
        "gpu_power_mw_mean": round(sum(pw) / len(pw), 1) if pw else None,
        # Surface the tool's own complaint (usually "requires root") instead of
        # reporting an empty result that looks like a measurement.
        "raw_head": (out[:300].strip() or "<no output — run under sudo>")
                    if not busy else None,
    }


def trellis_loop(stop: threading.Event, model_path: str) -> None:
    """Hammer ONLY the trellis expert GEMM at a prefill-shaped row count."""
    from escha_mlx import moe, msl

    K, IC, OC, E, R, m = 2, 2048, 1024, 256, 12, 2048
    rng = np.random.default_rng(0)
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt),
                                 dtype=np.uint64).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m + 1, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)
    ng = moe.n_groups_bound(m, E, R)
    ri, ge = moe.build_groups(re, E, R, ng)
    mx.eval(ri, ge)
    while not stop.is_set():
        # a batch of launches per eval, so host round-trips cannot dominate
        outs = [msl.moe_gemm_rows(xh, code, ri, ge, K, IC, OC, R, m)
                for _ in range(8)]
        mx.eval(outs)


def dense_loop(stop: threading.Event, model_path: str) -> None:
    """Control: MLX's own quantized matmul at comparable byte volume."""
    N, Kd, T = 8192, 2048, 2048
    w = mx.random.normal((N, Kd)).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((T, Kd)).astype(mx.float16)
    mx.eval(wq, s, b, x)
    while not stop.is_set():
        outs = [mx.quantized_matmul(x, wq, s, b, transpose=True,
                                    group_size=64, bits=4) for _ in range(8)]
        mx.eval(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--arm", choices=("trellis", "dense", "idle"), default="trellis")
    ap.add_argument("--seconds", type=int, default=8)
    args = ap.parse_args()

    stop = threading.Event()
    fn = {"trellis": trellis_loop, "dense": dense_loop}.get(args.arm)
    t = None
    if fn is not None:
        t = threading.Thread(target=fn, args=(stop, args.model), daemon=True)
        t.start()
        time.sleep(2.0)          # let it reach steady state before sampling

    print(f"sampling GPU for {args.seconds}s during arm={args.arm} ...")
    res = sample_powermetrics(args.seconds)
    stop.set()
    if t:
        t.join(timeout=10)

    for k, v in res.items():
        if v is not None:
            print(f"  {k}: {v}")
    b = res.get("gpu_busy_pct_mean")
    if b is not None:
        print()
        if b >= 90:
            print("  => GPU SATURATED: the kernel occupies the GPU but retires too")
            print("     little per cycle. Look INSIDE: occupancy, register pressure,")
            print("     dependent-load stalls.")
        elif b < 85:
            print("  => GPU PARTLY IDLE: time is being lost between/around kernels.")
            print("     Look OUTSIDE: dispatch overhead, MLX op count, command-buffer")
            print("     breaks, host round-trips.")


if __name__ == "__main__":
    main()
