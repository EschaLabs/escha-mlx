"""Reproducibility metadata for benchmark JSON reports."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


_REVISION = re.compile(r"[0-9a-f]{40}")
_ESCHA_MLX_ROOT = Path(__file__).resolve().parent.parent


def _git(args: list[str], path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def escha_mlx_git_revision(repo: str | Path = _ESCHA_MLX_ROOT) -> str | None:
    """Return the commit checked out for the escha-mlx source tree."""
    revision = _git(["rev-parse", "HEAD"], Path(repo))
    return revision if revision and _REVISION.fullmatch(revision) else None


def _metadata_revisions(model: Path) -> set[str]:
    metadata_dir = model / ".cache" / "huggingface" / "download"
    if not metadata_dir.is_dir():
        return set()
    revisions = set()
    for metadata in metadata_dir.rglob("*.metadata"):
        try:
            revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        except (IndexError, OSError, UnicodeError):
            continue
        if _REVISION.fullmatch(revision):
            revisions.add(revision)
    return revisions


def model_hf_revision(model: str | Path) -> str | None:
    """Resolve the Hugging Face revision of a local checkpoint without networking.

    Supports ``hf download --local-dir`` metadata, Hub snapshot-cache paths,
    config files carrying ``_commit_hash``, and Hugging Face git clones.
    ``None`` means that the local files do not identify one unambiguous revision.
    """
    path = Path(model).expanduser()
    if not path.exists():
        return None
    path = path.resolve()

    revisions = _metadata_revisions(path)
    if revisions:
        return next(iter(revisions)) if len(revisions) == 1 else None

    if path.parent.name == "snapshots" and _REVISION.fullmatch(path.name):
        return path.name

    config = path / "config.json"
    if config.is_file():
        try:
            revision = json.loads(config.read_text(encoding="utf-8")).get("_commit_hash")
        except (json.JSONDecodeError, OSError, UnicodeError):
            revision = None
        if isinstance(revision, str) and _REVISION.fullmatch(revision):
            return revision

    origin = _git(["config", "--get", "remote.origin.url"], path)
    if origin and "huggingface.co" in origin:
        revision = _git(["rev-parse", "HEAD"], path)
        if revision and _REVISION.fullmatch(revision):
            return revision
    return None


def model_display_name(model: str | Path) -> str:
    """Identify a checkpoint without publishing where it sits on disk.

    Benchmark reports are committed under bench/results/, so anything they
    record about the machine that produced them is published with them. The
    raw ``--model`` argument is usually an absolute path -- on a Hub cache that
    means a home directory -- and it carries no information the report does not
    already have: ``model_hf_revision`` pins the exact revision, which is
    strictly more precise than a path.

    So report the repo id instead. A Hub cache lays out
    ``models--Org--Name/snapshots/<revision>``, which recovers ``Org/Name``
    exactly; anything else falls back to the directory name.
    """
    path = Path(model).expanduser()
    for part in (path, *path.parents):
        if part.name.startswith("models--"):
            return part.name[len("models--"):].replace("--", "/", 1)
    name = path.name
    # A bare snapshot dir is named for its revision, which says nothing useful;
    # the directory holding snapshots/ is the checkpoint's real name.
    if _REVISION.fullmatch(name) and path.parent.name == "snapshots":
        name = path.parent.parent.name
    # Whatever component we landed on, it must not be the operator's home
    # directory -- that is the one name this function exists to keep out of a
    # committed report, and a layout like <home>/snapshots/<rev> would reach it.
    if not name or name == Path.home().name:
        return "local-checkpoint"
    return name


def benchmark_metadata(
    model: str | Path,
    repo: str | Path = _ESCHA_MLX_ROOT,
) -> dict[str, str | None]:
    """Build the version fields embedded in every benchmark JSON record."""
    return {
        "escha_mlx_git_revision": escha_mlx_git_revision(repo),
        "model_hf_revision": model_hf_revision(model),
    }


def annotate_report(
    report: dict | list[dict],
    metadata: dict[str, object],
) -> dict:
    """Add one top-level set of version fields to a benchmark report."""
    if isinstance(report, dict):
        return {**metadata, **report}
    return {**metadata, "results": report}
