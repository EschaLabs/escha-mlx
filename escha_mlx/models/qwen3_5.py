"""qwen3_5 dense architecture plugin (Qwen3.8 dense, hybrid GDN + attention).

Consumes the public HF export directly (no conversion artifact):
  * every projection in every decoder layer — `mlp.{gate,up,down}_proj`,
    `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` —
    is trellis-coded: `escha_{code,rin,rout,s_in,s_out,config}` (+ `bias`).
  * embed / lm_head: `weight_int8` + `weight_scale` pairs -> exact MLX affine-Q8
    repack (escha_mlx.quant).
  * everything else fp16 (norms, `linear_attn.{A_log,dt_bias,conv1d,norm}` and
    the small `in_proj_a`/`in_proj_b` routers) -> mlx-lm's own sanitize
    (language_model renames, conv1d layout, the (1+w) norm shift) + update.

The model skeleton, GDN kernels, attention, RoPE and KV/state caches are
mlx-lm's; only the coded linears and the quantized embed/head are replaced.
Post-load quirks: GDN recurrent-state dtype cache (escha_mlx.gdn_cache) and the
last-position LM head (escha_mlx.loader.LastPositionHead).

How this differs from the mixture-of-experts sibling. There, the coded weights
are E-stacked expert streams reached through a router, so the whole block is one
fused kernel over `row_expert`-indexed rows. Here each linear owns exactly one
stream and the layer is an ordinary sequence of matmuls — so the plugin installs
`escha_mlx.dense.EschaLinear` modules in place of the skeleton's `nn.Linear`
and never builds a block. The codec underneath is the same; the Metal kernels
take the single-stream case as a compile-time variant (escha_mlx.dense).

Mixed rate. A dense export may code different projections at different rates —
the shipped Qwen3.8-27B runs `mlp.{up,down}_proj` at K=3 and everything else at
K=2. K is a property of one stream, read from its own code tensor and
cross-checked against its own `escha_config`; nothing here assumes a global
rate, and the kernels are already specialised per K.

Per-linear forward (see escha_mlx.ref for the rounding contract):
    xh  = f16( H128(x * rin) * RS )
    mid = xh @ decode(code)                       (f32, fused Metal kernel)
    y   = f16( H128(mid) * RS * rout ) + bias
where rin/rout are the export's transform vectors with the end-to-end scales
s_in/s_out folded in (escha_mlx.ref.fold_scales).
"""
from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from .. import dense, gdn_cache, quant
from ..loader import LastPositionHead, resolve_module, strip_lm_prefix, use_last_logit

logger = logging.getLogger(__name__)

MODEL_TYPE = "qwen3_5"

#: Leaves that make up a coded linear, excluding `bias` — that name is not
#: exclusive to this format (a plain module could carry one), so it is held
#: separately and only claimed by a base that turns out to be coded.
_CODED_LEAVES = frozenset(l for l in dense.LEAVES if l.startswith("escha_"))


class CheckpointLoader:
    """Streaming consumer for escha qwen3_5 dense exports (contract:
    escha_mlx/models/__init__.py)."""

    def __init__(self, config: dict, group_size: int) -> None:
        from mlx_lm.models import qwen3_5 as skel

        self.model = skel.Model(skel.ModelArgs.from_dict(config))
        self.group_size = group_size
        text_args = self.model.language_model.args
        self.n_layers = text_args.num_hidden_layers
        self.hidden_size = text_args.hidden_size
        self.layers = self.model.language_model.model.layers

        # Streaming single pass. A dense export groups its tensors by LEAF name,
        # not by module — every escha_s_in in the model is written before the
        # first escha_code — so a linear cannot be installed the moment its group
        # completes the way the expert loader does. Only the code streams are
        # large, so those are converted to their device form on arrival (the
        # numpy copy is freed immediately) and the rest, a few tens of MB of
        # per-channel vectors in total, is held until finalize.
        self._coded: dict[str, dict[str, object]] = {}
        self._bias: dict[str, tuple[str, np.ndarray]] = {}
        self._int8_np: dict[str, dict[str, np.ndarray]] = {}
        self._base: dict[str, np.ndarray] = {}
        self.n_coded = 0
        self.n_bias = 0
        self.n_q8 = 0
        self.dropped = 0

    def _install_q8(self, base_name: str, pair: dict[str, np.ndarray]) -> None:
        w8, scale = pair["weight_int8"], pair["weight_scale"]
        if base_name == "lm_head":
            self.model.language_model.lm_head = quant.make_linear(w8, scale, self.group_size)
        elif base_name == "embed_tokens":
            self.model.language_model.model.embed_tokens = quant.make_embedding(
                w8, scale, self.group_size)
        elif base_name.startswith("layers."):
            rest = base_name[len("layers."):]
            idx, dotted = rest.split(".", 1)
            parent, attr = resolve_module(self.layers[int(idx)], dotted)
            setattr(parent, attr, quant.make_linear(w8, scale, self.group_size))
        else:
            raise ValueError(f"unexpected int8 tensor: {base_name}")
        self.n_q8 += 1

    def consume(self, name: str, w: np.ndarray) -> None:
        if name.startswith("mtp.") or ".visual." in name or name.startswith("visual."):
            self.dropped += 1
            return
        s = strip_lm_prefix(name)
        base, _, leaf = s.rpartition(".")
        if leaf in _CODED_LEAVES:
            group = self._coded.setdefault(base, {})
            group[leaf] = dense.pack_code(w) if leaf == "escha_code" else w
            return
        if leaf.startswith("escha_"):
            # An escha_* leaf this version does not know is an export written
            # against a newer format — a learned transform replacing the fixed
            # Hadamard, say. Left to fall through it would reach model.update
            # and die as 'Module does not have parameter named ...', which
            # reads as a loader bug rather than a version mismatch. And an
            # unknown TRANSFORM cannot be ignored the way an unknown scale
            # could: the weights would decode in the wrong basis.
            raise ValueError(
                f"{name}: unknown escha tensor {leaf!r}. This export uses a "
                f"format feature this runtime does not implement (known leaves: "
                f"{sorted(dense.LEAVES)}); upgrade escha-mlx.")
        if leaf == "bias":
            self._bias[base] = (name, w)
            return
        if leaf in ("weight_int8", "weight_scale"):
            pair = self._int8_np.setdefault(base, {})
            pair[leaf] = w
            if len(pair) == 2:
                self._install_q8(base, pair)
                del self._int8_np[base]
            return
        self._base[name] = w

    def _install_coded(self, base: str, group: dict[str, object]) -> dense.EschaLinear:
        if not base.startswith("layers."):
            raise ValueError(f"coded linear outside the decoder stack: {base}")
        lin = dense.build(group)
        idx, dotted = base[len("layers."):].split(".", 1)
        parent, attr = resolve_module(self.layers[int(idx)], dotted)
        setattr(parent, attr, lin)
        self.n_coded += 1
        return lin

    def finalize(self) -> list[mx.array]:
        # Raise, not assert: an unpaired weight_int8/weight_scale would leave
        # embed or lm_head at random init, and `python -O` strips asserts.
        if self._int8_np:
            raise ValueError(
                f"incomplete int8 tensor pairs: "
                f"{ {k: sorted(v) for k, v in self._int8_np.items()} }")

        # ---- coded linears -----------------------------------------------
        escha_arrays: list[mx.array] = []
        rates: dict[int, int] = {}
        for base in sorted(self._coded):
            group = self._coded[base]
            missing = dense.REQUIRED - set(group)
            if missing:
                raise ValueError(
                    f"{base}: incomplete escha linear, missing {sorted(missing)}")
            held = self._bias.pop(base, None)
            if held is not None:
                group["bias"] = held[1]
                self.n_bias += 1
            lin = self._install_coded(base, group)
            rates[lin.K] = rates.get(lin.K, 0) + 1
            escha_arrays += lin._w.arrays()
        self._coded.clear()

        # A `bias` nobody claimed belongs to an ordinary module; hand it back
        # under its original name so sanitize/update sees it.
        for orig, w in self._bias.values():
            self._base[orig] = w
        self._bias.clear()

        # ---- fp16 remainder through mlx-lm's own sanitize ----------------
        # mlx-lm only applies the (1+w) norm shift when it sees an unsanitized
        # conv1d (this export carries no mtp weights, the other trigger). If
        # that heuristic stops firing, 130 norms are silently mis-scaled — so
        # this is a raise, not an assert that `python -O` would strip.
        if not any(k.endswith("conv1d.weight") and v.shape[-1] != 1
                   for k, v in self._base.items()):
            raise ValueError(
                "no unsanitized conv1d.weight in the checkpoint: mlx-lm's "
                "(1+w) norm shift would not fire and every RMSNorm would be "
                "off by one")
        sanitized = self.model.sanitize({k: mx.array(v) for k, v in self._base.items()})
        self.model.update(tree_unflatten(list(sanitized.items())))
        self._base.clear()

        # ---- post-load quirks -------------------------------------------
        if use_last_logit():
            self.model.language_model.lm_head = LastPositionHead(
                self.model.language_model.lm_head)
            logger.info("escha_mlx: LM head restricted to the last position "
                        "(ESCHA_MLX_LAST_LOGIT=0 for per-position logits)")
        gdn_cache.install(self.model)

        logger.info("escha_mlx: %d coded linears across %d layers (%s), "
                    "%d Q8 dense, %d dropped", self.n_coded, self.n_layers,
                    ", ".join(f"K={k}: {n}" for k, n in sorted(rates.items())),
                    self.n_q8, self.dropped)
        if self.n_bias and not dense.apply_bias():
            # Say it out loud: these tensors exist, they are not zero, and this
            # runtime is choosing not to apply them because the reference
            # runtime does not either. See escha_mlx.dense.apply_bias.
            logger.info(
                "escha_mlx: %d per-linear bias tensors present and NOT applied "
                "(matching the reference runtime, which has no destination for "
                "them). ESCHA_MLX_BIAS=1 applies them — see "
                "escha_mlx.dense.apply_bias before comparing outputs.",
                self.n_bias)
        return escha_arrays
