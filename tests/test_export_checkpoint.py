import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from export_checkpoint import export_checkpoint


def test_exact_bytes_only_current_allowlisted_metadata(tmp_path):
    source = tmp_path / "processed"
    male = source / "male"
    male.mkdir(parents=True)
    payload = b'{  "inputs": {"manifest": "opaque-hash"}, "knots": [1,2] }\r\n'
    (male / "reviewed-alignment.json").write_bytes(payload)
    (male / "image.png").write_bytes(b"not an export")
    (male / "unknown.json").write_text('{"private":true}')
    for folder in ("rgb", "ct", "volume-v1", "rgb-volume-v1", "history", "cache"):
        (male / folder).mkdir()
        (male / folder / "manifest.json").write_text('{"never":"copy"}')
    output = tmp_path / "checkpoints" / "test"
    index = export_checkpoint(source, output)
    assert (output / "male" / "reviewed-alignment.json").read_bytes() == payload
    assert (male / "reviewed-alignment.json").read_bytes() == payload
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()) == [
        "index.json", "male/reviewed-alignment.json",
    ]
    record = index["subjects"]["male"]["files"][0]
    assert record == {"path": "male/reviewed-alignment.json", "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert index["metadata_file_count"] == 1
    assert str(source) not in json.dumps(index)
    assert index["subjects"]["male"]["status"] == "experimental"


def test_existing_output_never_overwritten(tmp_path):
    source = tmp_path / "processed"
    (source / "male").mkdir(parents=True)
    (source / "male" / "manifest.json").write_text('{}')
    output = tmp_path / "checkpoint"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        export_checkpoint(source, output)
    assert (output / "keep.txt").read_text() == "keep"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


def test_female_warning_and_missing_optional_metadata(tmp_path):
    source = tmp_path / "processed"
    (source / "female").mkdir(parents=True)
    (source / "female" / "manifest.json").write_text('{}')
    index = export_checkpoint(source, tmp_path / "checkpoint")
    assert index["subjects"]["female"]["status"] == "stale-geometry-unvalidated"
    assert "alignment-candidate-v2.json" in index["subjects"]["female"]["missing_optional_files"]


def test_subject_filter_exports_only_requested_subject(tmp_path):
    source = tmp_path / "processed"
    for subject in ("male", "female"):
        (source / subject).mkdir(parents=True)
        (source / subject / "manifest.json").write_text('{}')
    index = export_checkpoint(source, tmp_path / "checkpoint", subjects=("male",))
    assert set(index["subjects"]) == {"male"}
    assert not (tmp_path / "checkpoint/female").exists()


def test_public_checkpoint_can_include_geometry_manifests_without_bricks(tmp_path):
    source = tmp_path / "processed"
    volume = source / "male/volume-v1"
    rgb = source / "male/rgb-volume-v1"
    volume.mkdir(parents=True)
    rgb.mkdir(parents=True)
    (volume / "manifest.json").write_text('{"dimensions": [1,2,3]}')
    (rgb / "manifest.json").write_text('{"inputs": {"alignment": "hash"}}')
    (volume / "brick.u16le").write_bytes(b"never copy")
    output = tmp_path / "checkpoint"
    index = export_checkpoint(source, output, subjects=("male",), include_volume_manifests=True)
    assert (output / "male/volume-v1/manifest.json").is_file()
    assert (output / "male/rgb-volume-v1/manifest.json").is_file()
    assert not (output / "male/volume-v1/brick.u16le").exists()
    assert index["metadata_file_count"] == 2


def test_invalid_json_does_not_publish_partial_checkpoint(tmp_path):
    source = tmp_path / "processed"
    (source / "male").mkdir(parents=True)
    (source / "male" / "manifest.json").write_text('{}')
    (source / "male" / "alignment.json").write_text('broken')
    output = tmp_path / "checkpoint"
    with pytest.raises(ValueError):
        export_checkpoint(source, output)
    assert not output.exists()
    assert not (tmp_path / ".checkpoint.export.lock").exists()
    assert list(tmp_path.glob(".checkpoint.export-*")) == []


def test_refuses_output_inside_source(tmp_path):
    source = tmp_path / "processed"
    source.mkdir()
    with pytest.raises(ValueError, match="outside"):
        export_checkpoint(source, source / "checkpoint")
    assert list(source.iterdir()) == []


def test_does_not_follow_source_symlinks(tmp_path):
    source = tmp_path / "processed"
    (source / "male").mkdir(parents=True)
    target = tmp_path / "private.json"
    target.write_text('{}')
    (source / "male" / "manifest.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        export_checkpoint(source, tmp_path / "checkpoint")
    assert not (tmp_path / "checkpoint").exists()


def test_cooperating_exports_cannot_race(tmp_path):
    source = tmp_path / "processed"
    source.mkdir()
    lock = tmp_path / ".checkpoint.export.lock"
    lock.write_text("another exporter owns this")
    with pytest.raises(FileExistsError):
        export_checkpoint(source, tmp_path / "checkpoint")
    assert lock.read_text() == "another exporter owns this"
