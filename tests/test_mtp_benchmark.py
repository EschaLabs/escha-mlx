from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from bench import mtp as benchmark
from bench.mtp_metadata import mtp_head_metadata
from escha_mlx import envs

from .conftest import needs_metal, needs_mlx


def _loaded_head(tmp_path, monkeypatch, precision):
    from escha_mlx.mtp import load_mtp

    from .test_mtp import _target, _write_mtp

    if precision is None:
        monkeypatch.delenv("ESCHA_MLX_MTP_HEAD_BITS", raising=False)
    else:
        monkeypatch.setenv("ESCHA_MLX_MTP_HEAD_BITS", precision)
    target = _target()
    stored = _write_mtp(tmp_path, target)
    return target, load_mtp(tmp_path, target), stored


@needs_mlx
@pytest.mark.parametrize("precision", [None, "4", "8", "fp16"])
def test_metadata_reports_actual_loaded_precision(tmp_path, monkeypatch, precision):
    _, head, stored = _loaded_head(tmp_path, monkeypatch, precision)
    actual_precision = precision or "4"
    metadata = mtp_head_metadata(head, expected_precision=actual_precision)

    assert metadata["precision"] == actual_precision
    assert metadata["bits"] == (None if actual_precision == "fp16" else int(actual_precision))
    assert metadata["group_size"] == (None if actual_precision == "fp16" else 64)
    assert metadata["linear_count"] == 8
    stored_bytes = sum(value.nbytes for value in stored.values())
    if actual_precision == "fp16":
        assert metadata["head_bytes"] == stored_bytes
    else:
        assert 0 < metadata["head_bytes"] < stored_bytes

    # Inspection is independent of a stale process environment label.
    wrong_precision = "fp16" if actual_precision != "fp16" else "4"
    monkeypatch.setenv("ESCHA_MLX_MTP_HEAD_BITS", wrong_precision)
    assert mtp_head_metadata(head) == metadata
    with pytest.raises(ValueError, match="precision mismatch"):
        mtp_head_metadata(head, expected_precision=wrong_precision)


@needs_mlx
@pytest.mark.parametrize("corruption,match", [
    ("mixed_bits", "Mixed MTP draft precisions"),
    ("group_size", "group size 64"),
    ("mode", "affine Q4/Q8"),
    ("packed_dtype", "uint32 packed weights"),
    ("float32", "must be float16"),
    ("missing_linear", "exactly eight"),
])
def test_metadata_rejects_invalid_draft_linears(
    tmp_path, monkeypatch, corruption, match,
):
    import mlx.core as mx
    import mlx.nn as nn

    _, head, _ = _loaded_head(tmp_path, monkeypatch, "4")
    if corruption == "mixed_bits":
        head.fc.bits = 8
    elif corruption == "group_size":
        head.fc.group_size = 128
    elif corruption == "mode":
        head.fc.mode = "mxfp4"
    elif corruption == "packed_dtype":
        head.fc.weight = head.fc.weight.astype(mx.float16)
    elif corruption == "float32":
        head.fc = nn.Linear(256, 128, bias=False)
    else:
        head.fc = nn.Identity()
    with pytest.raises(ValueError, match=match):
        mtp_head_metadata(head)


class _TinyTokenizer:
    def encode(self, text):
        return [1, 2]

    def apply_chat_template(self, *args, **kwargs):
        return "tiny prompt"


@needs_metal
@pytest.mark.parametrize("precision", ["4", "8", "fp16"])
def test_one_shot_records_loaded_head(tmp_path, monkeypatch, precision):
    from escha_mlx import loader

    target, head, _ = _loaded_head(tmp_path, monkeypatch, precision)
    target.mtp = head
    target.eval()
    monkeypatch.setattr(loader, "load", lambda *args, **kwargs: (target, _TinyTokenizer()))
    args = argparse.Namespace(
        model=str(tmp_path), prompt="tiny prompt", warmup_tokens=2, output_tokens=4,
    )

    report = benchmark._one_shot(args)

    assert report["mtp_head"] == mtp_head_metadata(head)
    assert report["workload"]["ar_resident_mtp_head"] is True
    assert all(row["output_tokens"] == 4 for rows in report["runs"].values() for row in rows)


@needs_metal
@pytest.mark.parametrize("path", ["ar", "mtp"])
def test_batch_arm_records_resident_head(tmp_path, monkeypatch, path):
    from escha_mlx import loader

    target, head, _ = _loaded_head(tmp_path, monkeypatch, "4")
    target.mtp = head
    target.eval()
    monkeypatch.setattr(loader, "load", lambda *args, **kwargs: (target, _TinyTokenizer()))
    args = argparse.Namespace(
        model=str(tmp_path), batch=1, path=path, isl=2, prefill_step_size=16,
        warmup_tokens=4, output_tokens=4,
    )

    report = benchmark._batch_arm(args)

    assert report["mtp_head"] == mtp_head_metadata(head)
    assert report["path"] == path
    assert report["reached_max_tokens"] is True


def _continuous_args():
    return argparse.Namespace(
        model="checkpoint", batches="1,2", isl=128, output_tokens=96,
        warmup_tokens=16, prefill_step_size=256, arm_timeout=600,
    )


def _mock_arms(monkeypatch, change=None):
    calls = []
    precision = envs.ESCHA_MLX_MTP_HEAD_BITS.get()
    metadata = {
        "precision": precision,
        "bits": None if precision == "fp16" else int(precision),
        "group_size": None if precision == "fp16" else 64,
        "head_bytes": 1000,
        "linear_count": 8,
    }

    def run(command, **kwargs):
        # No explicit subprocess environment means the configured precision is
        # inherited, as it is when running the real child interpreter.
        assert "env" not in kwargs
        calls.append(command)
        path = command[command.index("--path") + 1]
        row = {
            "batch": int(command[command.index("--batch") + 1]),
            "path": path,
            "mtp_head": dict(metadata),
            "output_tps": 10 if path == "ar" else 12,
            "emission_per_request_round": 1 if path == "ar" else 2,
            "peak_gb": 1,
        }
        if change is not None:
            change(len(calls), row)
        return SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    return metadata, calls


@pytest.mark.parametrize("precision", [None, "4", "8", "fp16"])
def test_continuous_preserves_head_metadata_per_arm_and_summary(monkeypatch, precision):
    if precision is None:
        monkeypatch.delenv("ESCHA_MLX_MTP_HEAD_BITS", raising=False)
    else:
        monkeypatch.setenv("ESCHA_MLX_MTP_HEAD_BITS", precision)
    metadata, calls = _mock_arms(monkeypatch)

    report = benchmark._continuous(_continuous_args())

    assert len(calls) == 8
    assert report["mtp_head"] == metadata
    assert all(row["mtp_head"] == metadata for row in report["rows"])
    assert all(row["mtp_head"] == metadata for row in report["summary"])
    assert all(row["speedup"] == 1.2 for row in report["summary"])
    assert report["workload"]["ar_resident_mtp_head"] is True


@pytest.mark.parametrize("changed_field,value,match", [
    ("precision", "fp16", "expected MTP precision 4"),
    ("bits", 8, "inconsistent MTP head metadata"),
    ("group_size", 128, "inconsistent MTP head metadata"),
    ("head_bytes", 2000, "inconsistent MTP head metadata"),
    ("linear_count", 7, "inconsistent MTP head metadata"),
])
def test_continuous_rejects_cross_arm_mismatch(monkeypatch, changed_field, value, match):
    monkeypatch.setenv("ESCHA_MLX_MTP_HEAD_BITS", "4")

    def change(call, row):
        if call == 2:
            row["mtp_head"][changed_field] = value

    _, calls = _mock_arms(monkeypatch, change)
    with pytest.raises(ValueError, match=match):
        benchmark._continuous(_continuous_args())
    assert len(calls) == 2
