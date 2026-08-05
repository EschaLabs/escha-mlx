from __future__ import annotations

import json
import subprocess

from escha_mlx.benchmark_metadata import (
    annotate_report,
    benchmark_metadata,
    escha_mlx_git_revision,
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
