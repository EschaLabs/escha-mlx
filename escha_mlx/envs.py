"""Central registry for escha-mlx environment variables.

Environment variables are process-level overrides, not the primary model API.
They are grouped by ownership so low-level kernel experiments do not get mixed
with stable runtime behavior or deployment policy:

* ``BUILD``       -- native extension/toolchain settings (none today).
* ``DEPLOYMENT``  -- machine and process resource policy.
* ``RUNTIME``     -- stable behavior with user-visible semantic impact.
* ``KERNEL``      -- implementation selection, fallback, and performance tuning.
* ``DEVELOPMENT`` -- test and benchmark inputs.

Values are deliberately read on every ``get()``.  Benchmark harnesses and a few
tests switch kernel strategies inside one process, and caching here would make
their results depend on import order.  Call sites that need a fixed setting
(for example a model's MoE backend) read once when constructing that object.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Iterable, TypeVar, cast

T = TypeVar("T")

DEFAULT_Q8_GROUP = 128
GDN_STATE_CHOICES = (
    "fp16", "float16",
    "bf16", "bfloat16",
    "fp32", "float32",
)


class EnvLayer(Enum):
    """Ownership layer for an environment variable."""

    BUILD = "build"
    DEPLOYMENT = "deployment"
    RUNTIME = "runtime"
    KERNEL = "kernel"
    DEVELOPMENT = "development"


class _Unset:
    pass


UNSET = _Unset()


@dataclass(frozen=True)
class EnvVar(Generic[T]):
    """Typed, dynamically-read environment variable declaration."""

    name: str
    layer: EnvLayer
    parser: Callable[[str], T]
    description: str
    default: T | _Unset = UNSET
    value_help: str = "a valid value"

    def is_set(self) -> bool:
        return self.name in os.environ

    def get(self, *, default: T | _Unset = UNSET) -> T | None:
        """Return the parsed value, an explicit/default value, or ``None``.

        ``default=`` is useful when the default depends on runtime capability,
        such as selecting the Metal MoE backend only when Metal is available.
        """
        raw = os.environ.get(self.name)
        if raw is None or not raw.strip():
            value = self.default if isinstance(default, _Unset) else default
            return None if isinstance(value, _Unset) else cast(T, value)
        try:
            return self.parser(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.name} must be {self.value_help}, got {raw!r}"
            ) from exc


def _string(value: str) -> str:
    return value


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("invalid boolean")


def _choice(*choices: str) -> Callable[[str], str]:
    allowed = frozenset(choices)

    def parse(value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError("invalid choice")
        return normalized

    return parse


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("not positive")
    return parsed


def _kt_block(value: str) -> int:
    parsed = _positive_int(value)
    if parsed not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("unsupported tile count")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("not positive")
    return parsed


def _split_k(value: str) -> str | int:
    normalized = value.strip().lower()
    return "auto" if normalized == "auto" else _positive_int(normalized)


# Deployment policy ---------------------------------------------------------

ESCHA_MLX_WIRED_GB = EnvVar(
    "ESCHA_MLX_WIRED_GB",
    EnvLayer.DEPLOYMENT,
    _positive_float,
    "Unified-memory amount to wire through MLX, in decimal GB.",
    value_help="a positive number of GB",
)


# Stable runtime behavior ---------------------------------------------------

ESCHA_MLX_GDN_STATE = EnvVar(
    "ESCHA_MLX_GDN_STATE",
    EnvLayer.RUNTIME,
    _choice(*GDN_STATE_CHOICES),
    "Storage dtype for the recurrent GDN state.",
    default="fp16",
    value_help=f"one of {', '.join(GDN_STATE_CHOICES)}",
)

ESCHA_MLX_LAST_LOGIT = EnvVar(
    "ESCHA_MLX_LAST_LOGIT",
    EnvLayer.RUNTIME,
    _bool,
    "Compute only the final prompt position's logits during prefill.",
    default=True,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)

ESCHA_MLX_Q8_GROUP = EnvVar(
    "ESCHA_MLX_Q8_GROUP",
    EnvLayer.RUNTIME,
    _positive_int,
    "Requested MLX affine-Q8 group size.",
    default=DEFAULT_Q8_GROUP,
    value_help="a positive integer",
)

ESCHA_MLX_BIAS = EnvVar(
    "ESCHA_MLX_BIAS",
    EnvLayer.RUNTIME,
    _bool,
    "Apply the per-linear correction shipped by dense exports.",
    default=False,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)


# Kernel implementation and fallback selection -----------------------------

ESCHA_MLX_DENSE = EnvVar(
    "ESCHA_MLX_DENSE",
    EnvLayer.KERNEL,
    _choice("q8", "fp16"),
    "Dense-weight implementation used while loading a checkpoint.",
    default="q8",
    value_help="one of q8 or fp16",
)

ESCHA_MLX_MOE = EnvVar(
    "ESCHA_MLX_MOE",
    EnvLayer.KERNEL,
    _choice("fused", "ops"),
    "MoE implementation; the unset default depends on Metal availability.",
    value_help="one of fused or ops",
)

ESCHA_MLX_FUSED_HAD = EnvVar(
    "ESCHA_MLX_FUSED_HAD",
    EnvLayer.KERNEL,
    _bool,
    "Use fused expert input/output Hadamard kernels.",
    default=True,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)

ESCHA_MLX_LUT = EnvVar(
    "ESCHA_MLX_LUT",
    EnvLayer.KERNEL,
    _bool,
    "Use the codec lookup-table fallback instead of multiply-hash decode.",
    default=False,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)

ESCHA_MLX_GEMV = EnvVar(
    "ESCHA_MLX_GEMV",
    EnvLayer.KERNEL,
    _choice("direct", "staged"),
    "Per-row GEMV implementation.",
    default="direct",
    value_help="one of direct or staged",
)

ESCHA_MLX_FETCH = EnvVar(
    "ESCHA_MLX_FETCH",
    EnvLayer.KERNEL,
    _choice("load", "shuffle"),
    "Code-tile fetch implementation for direct GEMV.",
    default="load",
    value_help="one of load or shuffle",
)

ESCHA_MLX_SPLITK = EnvVar(
    "ESCHA_MLX_SPLITK",
    EnvLayer.KERNEL,
    _split_k,
    "Split-K factor, or auto for the size-based policy.",
    default=1,
    value_help="a positive integer or auto",
)

ESCHA_MLX_BLOCK_R = EnvVar(
    "ESCHA_MLX_BLOCK_R",
    EnvLayer.KERNEL,
    _positive_int,
    "Rows per expert group; unset uses the size-based policy.",
    value_help="a positive integer",
)

ESCHA_MLX_KT_BLOCK = EnvVar(
    "ESCHA_MLX_KT_BLOCK",
    EnvLayer.KERNEL,
    _kt_block,
    "Code tiles staged per barrier pair in row-blocked GEMM.",
    default=4,
    value_help="one of 1, 2, 4, 8, 16, or 32",
)

ESCHA_MLX_PREFETCH = EnvVar(
    "ESCHA_MLX_PREFETCH",
    EnvLayer.KERNEL,
    _bool,
    "Prefetch a block of code tiles into registers.",
    default=False,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)

ESCHA_MLX_SORTX = EnvVar(
    "ESCHA_MLX_SORTX",
    EnvLayer.KERNEL,
    _bool,
    "Pre-sort input rows into expert order for grouped GEMM.",
    default=False,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)

ESCHA_MLX_LINEAR = EnvVar(
    "ESCHA_MLX_LINEAR",
    EnvLayer.KERNEL,
    _choice("fused", "ops"),
    "Dense coded-linear implementation; the unset default depends on Metal availability.",
    value_help="one of fused or ops",
)

ESCHA_MLX_GEMV_PF = EnvVar(
    "ESCHA_MLX_GEMV_PF",
    EnvLayer.KERNEL,
    _positive_int,
    "Code tiles prefetched by the direct GEMV kernel.",
    default=1,
    value_help="a positive integer",
)

ESCHA_MLX_DENSE_BLOCK_R = EnvVar(
    "ESCHA_MLX_DENSE_BLOCK_R",
    EnvLayer.KERNEL,
    _positive_int,
    "Rows per group for dense row-blocked GEMM; unset uses the size policy.",
    value_help="a positive integer",
)

ESCHA_MLX_DENSE_MAT = EnvVar(
    "ESCHA_MLX_DENSE_MAT",
    EnvLayer.KERNEL,
    _bool,
    "Use the deterministic, non-bit-identical simdgroup-matrix dense GEMM.",
    default=False,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)


# Development inputs --------------------------------------------------------

ESCHA_MODEL = EnvVar(
    "ESCHA_MODEL",
    EnvLayer.DEVELOPMENT,
    _string,
    "Checkpoint path used by tests and benchmark examples.",
    value_help="a checkpoint path",
)

ESCHA_DENSE_MODEL = EnvVar(
    "ESCHA_DENSE_MODEL",
    EnvLayer.DEVELOPMENT,
    _string,
    "Dense checkpoint path used by checkpoint-dependent tests.",
    value_help="a checkpoint path",
)

ESCHA_MLX_SLOW_TESTS = EnvVar(
    "ESCHA_MLX_SLOW_TESTS",
    EnvLayer.DEVELOPMENT,
    _bool,
    "Opt in to slow, memory-heavy checkpoint tests.",
    default=False,
    value_help="a boolean (0/1, false/true, no/yes, or off/on)",
)


_DECLARATIONS = (
    ESCHA_MLX_WIRED_GB,
    ESCHA_MLX_GDN_STATE,
    ESCHA_MLX_LAST_LOGIT,
    ESCHA_MLX_Q8_GROUP,
    ESCHA_MLX_BIAS,
    ESCHA_MLX_DENSE,
    ESCHA_MLX_MOE,
    ESCHA_MLX_FUSED_HAD,
    ESCHA_MLX_LUT,
    ESCHA_MLX_GEMV,
    ESCHA_MLX_FETCH,
    ESCHA_MLX_SPLITK,
    ESCHA_MLX_BLOCK_R,
    ESCHA_MLX_KT_BLOCK,
    ESCHA_MLX_PREFETCH,
    ESCHA_MLX_SORTX,
    ESCHA_MLX_LINEAR,
    ESCHA_MLX_GEMV_PF,
    ESCHA_MLX_DENSE_BLOCK_R,
    ESCHA_MLX_DENSE_MAT,
    ESCHA_MODEL,
    ESCHA_DENSE_MODEL,
    ESCHA_MLX_SLOW_TESTS,
)

environment_variables: dict[str, EnvVar[Any]] = {
    variable.name: variable for variable in _DECLARATIONS
}

if len(environment_variables) != len(_DECLARATIONS):  # pragma: no cover
    raise RuntimeError("duplicate environment variable declaration")


def variables_in(layer: EnvLayer) -> tuple[EnvVar[Any], ...]:
    """Return declarations in one ownership layer, in registry order."""
    return tuple(variable for variable in _DECLARATIONS if variable.layer is layer)


def validate_environment(
    layers: EnvLayer | Iterable[EnvLayer] | None = None,
) -> None:
    """Validate all currently set escha variables in the requested layers."""
    if layers is None:
        selected = set(EnvLayer)
    elif isinstance(layers, EnvLayer):
        selected = {layers}
    else:
        selected = set(layers)
    for variable in _DECLARATIONS:
        if variable.layer in selected and variable.is_set():
            variable.get()
