"""Inspect the loaded MTP head instead of treating an environment label as proof."""
from __future__ import annotations


_DRAFT_LINEAR_NAMES = frozenset({
    "fc",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
})


def mtp_head_metadata(
    head, *, expected_precision=None, expected_group_size=64,
) -> dict:
    """Describe and validate the eight draft linears actually loaded in ``head``.

    Shared target vocabulary modules must remain outside the head's registered
    modules and parameters. Imports are lazy so bench tools and their non-model
    tests can be collected on hosts without MLX.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    if expected_precision not in (None, "4", "8", "fp16"):
        raise ValueError(f"Unsupported expected MTP precision: {expected_precision!r}")
    linears = {
        name: module for name, module in head.named_modules()
        if isinstance(module, (nn.Linear, nn.QuantizedLinear))
    }
    if set(linears) != _DRAFT_LINEAR_NAMES:
        raise ValueError(
            "Expected exactly eight MTP draft linears; "
            f"missing={sorted(_DRAFT_LINEAR_NAMES - set(linears))}, "
            f"unexpected={sorted(set(linears) - _DRAFT_LINEAR_NAMES)}"
        )

    precisions = set()
    for name, layer in linears.items():
        if isinstance(layer, nn.QuantizedLinear):
            if (
                layer.bits not in (4, 8)
                or layer.group_size != expected_group_size
                or layer.mode != "affine"
                or layer.weight.dtype != mx.uint32
            ):
                raise ValueError(
                    f"{name}: expected affine Q4/Q8, group size "
                    f"{expected_group_size}, and uint32 packed weights"
                )
            precisions.add(str(layer.bits))
        else:
            if layer.weight.dtype != mx.float16:
                raise ValueError(f"{name}: unquantized MTP weight must be float16")
            precisions.add("fp16")
    if len(precisions) != 1:
        raise ValueError(f"Mixed MTP draft precisions: {sorted(precisions)}")
    precision = precisions.pop()
    if expected_precision is not None and precision != expected_precision:
        raise ValueError(
            f"MTP precision mismatch: expected {expected_precision}, loaded {precision}"
        )
    return {
        "precision": precision,
        "bits": None if precision == "fp16" else int(precision),
        "group_size": None if precision == "fp16" else expected_group_size,
        "head_bytes": sum(value.nbytes for _, value in tree_flatten(head.parameters())),
        "linear_count": len(linears),
    }
