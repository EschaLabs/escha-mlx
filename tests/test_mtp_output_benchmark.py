from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from bench.mtp_head_output import (
    _HEAD_PRECISION_ENV,
    _load_comparison_heads,
    _require_precision_control,
)

from .conftest import needs_mlx


def test_output_benchmark_rejects_source_without_precision_control():
    with pytest.raises(ValueError, match="cannot compare FP16 and Q4"):
        _require_precision_control(SimpleNamespace(environment_variables={}))
    _require_precision_control(
        SimpleNamespace(environment_variables={_HEAD_PRECISION_ENV: object()})
    )


@needs_mlx
@pytest.mark.parametrize("initial", [None, "4", "8", "fp16"])
def test_output_benchmark_records_actual_arm_precision(tmp_path, monkeypatch, initial):
    from escha_mlx.mtp import load_mtp

    from .test_mtp import _target, _write_mtp

    if initial is None:
        monkeypatch.delenv(_HEAD_PRECISION_ENV, raising=False)
    else:
        monkeypatch.setenv(_HEAD_PRECISION_ENV, initial)
    target = _target()
    _write_mtp(tmp_path, target)
    heads, metadata = _load_comparison_heads(tmp_path, target, load_mtp)

    assert set(heads) == set(metadata) == {"fp16", "q4"}
    for arm, precision, bits, group_size in (
        ("fp16", "fp16", None, None), ("q4", "4", 4, 64),
    ):
        assert metadata[arm]["precision"] == precision
        assert metadata[arm]["bits"] == bits
        assert metadata[arm]["group_size"] == group_size
        assert metadata[arm]["linear_count"] == 8
        assert metadata[arm]["head_bytes"] > 0
    assert metadata["q4"]["head_bytes"] < metadata["fp16"]["head_bytes"]
    assert json.loads(json.dumps(metadata)) == metadata
    assert os.environ.get(_HEAD_PRECISION_ENV) == initial


@needs_mlx
def test_output_benchmark_rejects_loader_ignoring_q4(tmp_path, monkeypatch):
    from escha_mlx.mtp import load_mtp

    from .test_mtp import _target, _write_mtp

    monkeypatch.setenv(_HEAD_PRECISION_ENV, "8")
    target = _target()
    _write_mtp(tmp_path, target)
    requested = []

    def old_loader(path, model):
        requested.append(os.environ[_HEAD_PRECISION_ENV])
        # Simulate a source predating Q4: every load returns the stored FP16.
        with monkeypatch.context() as patch:
            patch.setenv(_HEAD_PRECISION_ENV, "fp16")
            return load_mtp(path, model)

    with pytest.raises(ValueError):
        _load_comparison_heads(tmp_path, target, old_loader)
    assert requested == ["fp16", "4"]
    assert os.environ[_HEAD_PRECISION_ENV] == "8"


@pytest.mark.parametrize("initial", [None, "8"])
def test_output_benchmark_restores_precision_when_loading_fails(monkeypatch, initial):
    if initial is None:
        monkeypatch.delenv(_HEAD_PRECISION_ENV, raising=False)
    else:
        monkeypatch.setenv(_HEAD_PRECISION_ENV, initial)

    def failing_loader(path, model):
        raise RuntimeError("checkpoint unavailable")

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        _load_comparison_heads(None, None, failing_loader)
    assert os.environ.get(_HEAD_PRECISION_ENV) == initial
