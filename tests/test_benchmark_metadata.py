from __future__ import annotations

import json
import subprocess

from pathlib import Path

from escha_mlx.benchmark_metadata import (
    annotate_report,
    benchmark_metadata,
    escha_mlx_git_revision,
    model_display_name,
    model_hf_revision,
)


REVISION = "0123456789abcdef0123456789abcdef01234567"


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_escha_mlx_git_revision(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked").write_text("one")
    _git(tmp_path, "add", "tracked")
    _git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "initial")

    assert escha_mlx_git_revision(tmp_path) == _git(tmp_path, "rev-parse", "HEAD")


def test_local_dir_hugging_face_revision(tmp_path):
    metadata = tmp_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{REVISION}\nblob\ntimestamp\n")

    assert model_hf_revision(tmp_path) == REVISION


def test_mixed_local_dir_revisions_are_ambiguous(tmp_path):
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text(f"{REVISION}\nblob\n")
    (metadata / "weights.metadata").write_text(f"{'f' * 40}\nblob\n")

    assert model_hf_revision(tmp_path) is None


def test_snapshot_and_config_revisions(tmp_path):
    snapshot = tmp_path / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    assert model_hf_revision(snapshot) == REVISION

    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "config.json").write_text(json.dumps({"_commit_hash": REVISION}))
    assert model_hf_revision(configured) == REVISION


def test_report_annotation_keeps_one_metadata_set(tmp_path):
    metadata = benchmark_metadata(tmp_path, tmp_path)
    metadata["escha_mlx_git_revision"] = REVISION
    metadata["model_hf_revision"] = REVISION

    assert annotate_report({"metric": 1}, metadata) == {**metadata, "metric": 1}
    assert annotate_report([{"metric": 1}], metadata) == {
        **metadata,
        "results": [{"metric": 1}],
    }


def test_model_display_name_never_publishes_a_path():
    """Benchmark reports are committed, so what identifies the checkpoint in one
    must not identify the machine that ran it.

    A Hub cache path is the common case and carries a home directory; the repo
    id recovered from it is both anonymous and more useful. The revision is
    already recorded separately, so nothing is lost.
    """
    hub = ("/Users/somebody/.cache/huggingface/hub/"
           "models--EschaLabs--Qwen3.8-27B-Escha-W2/snapshots/" + REVISION)
    assert model_display_name(hub) == "EschaLabs/Qwen3.8-27B-Escha-W2"
    # the model dir itself, without a snapshot component
    assert model_display_name(
        "/Users/somebody/.cache/huggingface/hub/models--EschaLabs--Qwen3.8-27B-Escha-W2"
    ) == "EschaLabs/Qwen3.8-27B-Escha-W2"

    # A plain directory keeps its own name, which is what the operator chose.
    assert model_display_name("~/Desktop/escha-release-2026-07-16") == \
        "escha-release-2026-07-16"
    assert model_display_name("./escha-w2") == "escha-w2"

    # A bare snapshot dir is named for its revision; the dir above it is the name.
    assert model_display_name("/some/where/snapshots/" + REVISION) == "where"

    # The one component that must never survive, however we arrive at it.
    home = Path.home()
    assert model_display_name(home / "snapshots" / REVISION) == "local-checkpoint"

    for probe in (hub, str(home / "snapshots" / REVISION)):
        assert home.name not in model_display_name(probe)
        assert "/Users/" not in model_display_name(probe)
