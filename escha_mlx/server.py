"""OpenAI-compatible serving: python -m escha_mlx.server --model <dir> [mlx_lm.server args]

Thin wrapper over mlx_lm.server (continuous batching via BatchGenerator, prefix
caching via LRUPromptCache, reasoning-content parsing, tool calls): routes model
loading through escha_mlx.loader for escha checkpoints. Qwen3.8's native MTP
head can be enabled without disabling continuous batching.

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
_MTP_DEFAULT_PROMPT_CACHE_BYTES = 512_000_000


def _consume_mtp_args(argv):
    """Remove the escha-only MTP switch before mlx-lm parses its arguments."""
    enabled = False
    cleaned = [argv[0]]
    for arg in argv[1:]:
        if arg == "--mtp":
            enabled = True
        else:
            cleaned.append(arg)
    return cleaned, enabled


def _append_default_option(argv, option: str, value: str) -> bool:
    """Append a server default unless either argparse spelling is present."""
    if any(arg == option or arg.startswith(option + "=") for arg in argv[1:]):
        return False
    argv.extend([option, value])
    return True


def _install_mtp_prompt_cache_limit(server_mod, argv) -> int:
    """Apply a byte cap to mlx-lm's count-bounded prompt LRU.

    A B=8 MTP verification keeps a large GDN state trace.  Ten ordinary
    Qwen3.8 prompt entries add about 0.9 GB and can push a 24 GB machine over
    Metal's recommended working set, where failures surface as unrelated lazy
    graph errors.  mlx-lm's ``--prompt-cache-bytes`` trims against active batch
    caches, but its LRU constructor only receives the entry-count limit.  Apply
    the same byte limit at insertion time as well so completed requests cannot
    temporarily exceed the budget.
    """
    option = "--prompt-cache-bytes"
    raw = None
    for index, arg in enumerate(argv[1:], start=1):
        if arg.startswith(option + "="):
            raw = arg.split("=", 1)[1]
            break
        if arg == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            raw = argv[index + 1]
            break
    if raw is None:
        budget = _MTP_DEFAULT_PROMPT_CACHE_BYTES
        argv.extend([option, str(budget)])
    else:
        budget = server_mod._parse_size(raw)

    original = server_mod.LRUPromptCache

    def bounded_prompt_cache(max_size):
        return original(max_size=max_size, max_bytes=budget)

    server_mod.LRUPromptCache = bounded_prompt_cache
    return budget


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


def _install_exact_hit_guard() -> None:
    """Never let the prompt cache return an empty remainder.

    mlx_lm 0.31.3's ``LRUPromptCache.fetch_nearest_cache`` answers an EXACT
    key hit with ``(deepcopy(cache), [])``.  Both serving paths assume at
    least one prompt token remains: the batched path pops every prompt
    segment and then indexes ``seq[-1]`` on the empty list
    (``BatchGenerator.insert_segments``) -- an IndexError that kills the
    generation thread, after which every request queues forever -- and the
    sequential path would hand ``stream_generate`` an empty prompt.  The key
    is ``prompt + generated`` of a finished request, so a /v1/completions
    continuation of a previous response hits it in ordinary traffic.

    The guard re-establishes the invariant at the source instead of patching
    each consumer:

    * trimmable cache (pure-KV models served through this wrapper): trim one
      token off the copy and re-feed the last prompt token.  This is also the
      byte-equal choice -- feeding the last token against the full-length
      cache would advance its state a second time and shift the first
      generated token's position.
    * non-trimmable cache (both escha models: GDNStateCache holds recurrent
      state that only resumes at its exact stored boundary): decline the
      exact entry and re-resolve the query minus its final token.  That
      returns a stored key equal to ``tokens[:-1]`` (reused as-is, final
      token re-fed), a shorter stored prefix, or a cold start -- the
      full-length entry itself is correctly skipped by the trim gate.  Every
      arm feeds ``>= 1`` real token and resumes state only at a boundary it
      was stored at.
    """
    from mlx_lm.models import cache as cache_mod

    lru = cache_mod.LRUPromptCache
    if getattr(lru.fetch_nearest_cache, "_escha_exact_hit_guard", False):
        return
    orig = lru.fetch_nearest_cache

    def fetch_nearest_cache(self, model, tokens):
        cache, rest = orig(self, model, tokens)
        if cache is None or rest:
            return cache, rest
        if len(tokens) < 2:
            return None, list(tokens)
        if cache_mod.can_trim_prompt_cache(cache):
            cache_mod.trim_prompt_cache(cache, 1)
            return cache, list(tokens[-1:])
        cache, rest = orig(self, model, tokens[:-1])
        if cache is None:
            return None, list(tokens)
        # `rest` is relative to tokens[:-1]; [] here means tokens[:-1] is
        # itself a stored key, reusable as-is because the remainder below
        # feeds the final token against it.
        return cache, list(rest) + list(tokens[-1:])

    fetch_nearest_cache._escha_exact_hit_guard = True
    lru.fetch_nearest_cache = fetch_nearest_cache


def _install_slot_realign_guard() -> None:
    """Keep per-request samplers/logits_processors aligned with uids.

    mlx_lm 0.31.3's ``GenerationBatch.filter`` reindexes ``samplers`` and
    ``logits_processors`` only when ``any(...)`` is truthy. A batch whose
    requests all lack processors carries ``[[], [], ...]`` — all falsy — so
    on request completion the list keeps its STALE length while ``uids``
    shrinks; the next ``extend()`` then appends the incoming request's
    processors positionally, and from that point every per-request processor
    (and, for temp>0, sampler) reads the wrong slot. Reachable today with
    upstream's own per-request ``logit_bias`` / ``repetition_penalty``: a
    plain request finishing ahead of one that carries them leaves theirs
    attached to the wrong sequence. (The sibling
    ``PromptProcessingBatch.filter`` has the correct else-branch
    normalization; this brings ``GenerationBatch`` to parity.) Realigning
    after the fact is safe precisely because the guard only ever fires when
    every entry is falsy — there is no information in the stale list to
    lose.
    """
    # NB: `from mlx_lm import generate` yields the re-exported FUNCTION that
    # shadows the submodule of the same name; the module path form below
    # resolves the module.
    from mlx_lm.generate import GenerationBatch as gb

    if getattr(gb.filter, "_escha_slot_realign", False):
        return
    orig = gb.filter

    def filter(self, keep):
        orig(self, keep)
        n = len(self.uids)
        if len(self.samplers) != n:
            self.samplers = [None] * n
        if len(self.logits_processors) != n:
            self.logits_processors = [[] for _ in range(n)]

    filter._escha_slot_realign = True
    gb.filter = filter


def main() -> None:
    argv, mtp_enabled = _consume_mtp_args(sys.argv)
    sys.argv[:] = argv
    from mlx_lm import server as _server

    from .loader import is_escha_checkpoint, load as escha_load

    _orig_load = _server.load

    def _load(path, *a, **kw):
        if is_escha_checkpoint(path):
            if kw.get("adapter_path"):
                raise ValueError("escha_mlx: adapters are not supported")
            return escha_load(
                path,
                tokenizer_config=kw.get("tokenizer_config"),
                load_mtp=mtp_enabled,
            )
        return _orig_load(path, *a, **kw)

    _server.load = _load
    if mtp_enabled:
        if any(
            arg == "--draft-model" or arg.startswith("--draft-model=")
            for arg in sys.argv
        ):
            raise ValueError("--mtp cannot be combined with an external --draft-model")
        # Tree verification wins at B=1/2 but loses to ordinary batched AR from
        # B=4 onward, and its GDN traces scale linearly with the active batch.
        # Keep the opt-in server safe and fast by default; explicit CLI values
        # still win on machines where the user has measured another policy.
        _append_default_option(sys.argv, "--decode-concurrency", "2")
        _append_default_option(sys.argv, "--prompt-concurrency", "2")
        _install_mtp_prompt_cache_limit(_server, sys.argv)
        from .mtp_batch import MTPBatchGenerator

        _server.BatchGenerator = MTPBatchGenerator
    _install_ignore_eos(_server)
    _install_exact_hit_guard()
    _install_slot_realign_guard()
    sys.argv[0] = "escha_mlx.server"
    _server.main()


if __name__ == "__main__":
    main()
