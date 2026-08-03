"""OpenAI-compatible serving: python -m escha_mlx.server --model <dir> [mlx_lm.server args]

Thin wrapper over mlx_lm.server (continuous batching via BatchGenerator, prefix
caching via LRUPromptCache, reasoning-content parsing, tool calls): routes model
loading through escha_mlx.loader for eschamoe checkpoints, delegates everything
else verbatim.

Adds one serving extension: ``"ignore_eos": true`` in the request body, which
suppresses end-of-sequence stopping so a request generates exactly max_tokens.
This is required for honest ISL/OSL benchmarking -- without it a batch DRAINS as
sequences finish early and the measurement reports a decaying batch rather than
steady-state throughput (measured: osl_hit_rate 0.04 on random prompts, which
understated aggregate throughput by ~40%).  It matches the flag vLLM and SGLang
expose for the same reason.

Known limitation: mlx_lm.server has no structured output / json_schema
constrained decoding.
"""
from __future__ import annotations

import sys

# A stop-word that cannot occur in real text; carries the ignore_eos intent
# through mlx-lm's (model_key, stop_words, state) state-machine cache key, so
# ignore_eos and normal requests never share a cached stop machine.
_IGNORE_EOS_SENTINEL = "\x00escha_ignore_eos\x00"


class _NoEosTokenizer:
    """Tokenizer view whose eos set is empty; everything else delegates.

    Only `_make_state_machine` sees this, and only to build the stop machine --
    detokenization, thinking and tool-call sequences all pass through unchanged.
    """

    def __init__(self, inner) -> None:
        object.__setattr__(self, "_inner", inner)

    @property
    def eos_token_ids(self):
        return []

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def _install_ignore_eos(server_mod) -> None:
    rg = server_mod.ResponseGenerator
    orig_make = rg._make_state_machine

    def _make_state_machine(self, model_key, tokenizer, stop_words, initial_state="normal"):
        if _IGNORE_EOS_SENTINEL in stop_words:
            stop_words = [w for w in stop_words if w != _IGNORE_EOS_SENTINEL]
            tokenizer = _NoEosTokenizer(tokenizer)
            # keep the sentinel in the cache key so this machine is not reused
            # for a normal request with the same stop words
            return orig_make(self, (model_key, "ignore_eos"), tokenizer,
                             stop_words, initial_state)
        return orig_make(self, model_key, tokenizer, stop_words, initial_state)

    rg._make_state_machine = _make_state_machine

    handler = server_mod.APIHandler
    orig_validate = handler.validate_model_parameters

    def validate_model_parameters(self):
        # runs after self.body is set and before stop_words is read
        if self.body.get("ignore_eos"):
            stop = self.body.get("stop") or []
            stop = [stop] if isinstance(stop, str) else list(stop)
            self.body["stop"] = stop + [_IGNORE_EOS_SENTINEL]
        return orig_validate(self)

    handler.validate_model_parameters = validate_model_parameters


def main() -> None:
    from mlx_lm import server as _server

    from .loader import is_escha_checkpoint, load as escha_load

    _orig_load = _server.load

    def _load(path, *a, **kw):
        if is_escha_checkpoint(path):
            if kw.get("adapter_path"):
                raise ValueError("escha_mlx: adapters are not supported")
            return escha_load(path, tokenizer_config=kw.get("tokenizer_config"))
        return _orig_load(path, *a, **kw)

    _server.load = _load
    _install_ignore_eos(_server)
    sys.argv[0] = "escha_mlx.server"
    _server.main()


if __name__ == "__main__":
    main()
