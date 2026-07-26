"""Tests for VLM weight-availability checking.

These guard a failure that already cost a full run: Lingshu-7B had all four shard
files present but 1.7GB short, passed an existence-only check, and then failed all
171 cases with ``SafetensorError: Error while deserializing header: incomplete
metadata``. Availability must therefore be decided on bytes, not on stat().

The HF cache stores shards as symlinks into ``blobs/``, so size checks have to
resolve the link; asserting that here keeps a future refactor from silently
measuring link sizes instead of payload sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.vlms import VLMMethod

MB = 1024 * 1024
#: Shortfall the check is allowed to ignore, mirroring benchmark.vlms. Test sizes
#: must straddle this: a gap below it is legitimate header overhead, not damage.
TOLERANCE_MB = 50


def _sparse(path: Path, size: int) -> None:
    """Create a file that reports ``size`` bytes without allocating them.

    Real shards are gigabytes and the truncation threshold is 50MB, so the test
    has to work at that scale. Sparse files give the size that getsize() reports
    while consuming almost no disk.
    """
    with path.open("wb") as handle:
        handle.truncate(size)


def _snapshot(root: Path, repo_id: str) -> Path:
    """Build an HF-cache-shaped snapshot directory and return it.

    A config.json is written because check_available requires one before it looks
    at weights; without it every case would fail for the wrong reason.
    """
    cache = root / "hub" / f"models--{repo_id.replace('/', '--')}"
    snapshot = cache / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (cache / "blobs").mkdir(exist_ok=True)
    with (snapshot / "config.json").open("w") as handle:
        json.dump({"model_type": "probe"}, handle)
    return snapshot


def _write_sharded(
    snapshot: Path,
    *,
    declared_total: int,
    shard_sizes: dict[str, int],
    symlink: bool = True,
) -> None:
    """Write a shard index plus shard payloads of the given sizes.

    Payloads go into ``blobs/`` with a symlink from the snapshot when ``symlink``
    is set, mirroring how huggingface_hub materialises a download.
    """
    index = {
        "metadata": {"total_size": declared_total},
        "weight_map": {f"layer.{i}.weight": name for i, name in enumerate(shard_sizes)},
    }
    with (snapshot / "model.safetensors.index.json").open("w") as handle:
        json.dump(index, handle)

    blobs = snapshot.parent.parent / "blobs"
    for name, size in shard_sizes.items():
        if symlink:
            blob = blobs / f"blob-{name}"
            _sparse(blob, size)
            (snapshot / name).symlink_to(blob)
        else:
            _sparse(snapshot / name, size)


def _check(monkeypatch, root: Path, repo_id: str) -> tuple[bool, str]:
    """Run check_available against a cache rooted at ``root``.

    The method is built without __init__ so the test does not have to satisfy
    unrelated required fields such as provenance.
    """
    from benchmark import vlms

    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vlms, "HF_CACHE", hub)

    method = VLMMethod.__new__(VLMMethod)
    object.__setattr__(method, "name", "probe")
    object.__setattr__(method, "repo_id", repo_id)
    return vlms.VLMMethod.check_available(method)


def test_complete_download_is_available(monkeypatch, tmp_path: Path) -> None:
    repo = "org/complete"
    snapshot = _snapshot(tmp_path, repo)
    _write_sharded(
        snapshot,
        declared_total=8000 * MB,
        shard_sizes={"model-00001-of-00002.safetensors": 4000 * MB,
                     "model-00002-of-00002.safetensors": 4000 * MB},
    )
    ok, message = _check(monkeypatch, tmp_path, repo)
    assert ok, message


def test_truncated_shard_is_rejected_even_though_every_file_exists(
    monkeypatch, tmp_path: Path
) -> None:
    """The Lingshu-7B failure: all shards present, total short.

    The gap here (3000MB) is far above the tolerance, matching the real 1.7GB
    shortfall rather than a size that rounding could explain.
    """
    repo = "org/truncated"
    snapshot = _snapshot(tmp_path, repo)
    _write_sharded(
        snapshot,
        declared_total=8000 * MB,
        shard_sizes={"model-00001-of-00002.safetensors": 4000 * MB,
                     "model-00002-of-00002.safetensors": 1000 * MB},
    )
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        assert (snapshot / name).exists(), "precondition: files must all exist"

    ok, message = _check(monkeypatch, tmp_path, repo)
    assert not ok
    assert "truncated" in message.lower()


def test_rejection_message_reports_the_shortfall(monkeypatch, tmp_path: Path) -> None:
    """The operator needs the size gap to decide whether to resume a download."""
    repo = "org/shortfall"
    snapshot = _snapshot(tmp_path, repo)
    _write_sharded(
        snapshot,
        declared_total=2000 * MB,
        shard_sizes={"model-00001-of-00001.safetensors": 1000 * MB},
    )
    ok, message = _check(monkeypatch, tmp_path, repo)
    assert not ok
    assert "on disk" in message and "declared" in message


def test_small_overhead_is_tolerated(monkeypatch, tmp_path: Path) -> None:
    """total_size excludes some header bytes, so exact equality must not be required.

    A gap of a few MB is normal; only a gap beyond the tolerance means damage.
    """
    repo = "org/slightly-off"
    snapshot = _snapshot(tmp_path, repo)
    _write_sharded(
        snapshot,
        declared_total=8000 * MB,
        shard_sizes={"model-00001-of-00001.safetensors": (8000 - TOLERANCE_MB + 10) * MB},
    )
    ok, message = _check(monkeypatch, tmp_path, repo)
    assert ok, message


def test_symlinked_shards_are_measured_by_payload_not_link(
    monkeypatch, tmp_path: Path
) -> None:
    """A symlink's own size is a few dozen bytes; measuring it would fail everything."""
    repo = "org/symlinked"
    snapshot = _snapshot(tmp_path, repo)
    _write_sharded(
        snapshot,
        declared_total=8000 * MB,
        shard_sizes={"model-00001-of-00001.safetensors": 8000 * MB},
        symlink=True,
    )
    assert (snapshot / "model-00001-of-00001.safetensors").is_symlink()
    ok, message = _check(monkeypatch, tmp_path, repo)
    assert ok, message


def test_missing_shard_is_reported_as_missing_not_truncated(
    monkeypatch, tmp_path: Path
) -> None:
    repo = "org/missing"
    snapshot = _snapshot(tmp_path, repo)
    _write_sharded(
        snapshot,
        declared_total=8000 * MB,
        shard_sizes={"model-00001-of-00002.safetensors": 4000 * MB},
    )
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["layer.99.weight"] = "model-00002-of-00002.safetensors"
    index_path.write_text(json.dumps(index))

    ok, message = _check(monkeypatch, tmp_path, repo)
    assert not ok
    assert "missing" in message.lower()


def test_index_without_total_size_falls_back_to_existence(
    monkeypatch, tmp_path: Path
) -> None:
    """Some repos omit metadata.total_size; that must not become a hard failure."""
    repo = "org/no-total"
    snapshot = _snapshot(tmp_path, repo)
    index = {
        "metadata": {},
        "weight_map": {"layer.0.weight": "model-00001-of-00001.safetensors"},
    }
    with (snapshot / "model.safetensors.index.json").open("w") as handle:
        json.dump(index, handle)
    _sparse(snapshot / "model-00001-of-00001.safetensors", MB)

    ok, message = _check(monkeypatch, tmp_path, repo)
    assert ok, message
