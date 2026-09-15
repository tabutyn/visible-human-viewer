import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_public_release import build_release
from build_public_site import build_site


COUNTS = {0: 240, 1: 32, 2: 4}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return digest(path.read_bytes())


def make_fixture(tmp_path):
    processed = tmp_path / "processed"
    delivery = tmp_path / "delivery"
    subject = processed / "male"
    blobs = delivery / "blobs"
    blobs.mkdir(parents=True)
    hashes = {}
    for name, value in (
        ("manifest.json", {"subject": "male"}),
        ("corrections.json", {"version": 1}),
        ("alignment.json", {"depth_knots": []}),
        ("alignment-candidate-v2.json", {"status": "reviewed"}),
        ("reviewed-alignment.json", {"anchors": [{"color_frame": 0}]}),
        ("medical-view.json", {"preset": "lung", "yaw": -1.2}),
    ):
        hashes[name] = write_json(subject / name, value)

    index = {"version": 1, "lossless": True, "files": {}}

    def volume(folder, modality):
        levels = []
        for level, count in COUNTS.items():
            bricks = []
            for number in range(count):
                decoded = f"{modality}-{level}-{number}".encode()
                decoded_sha = digest(decoded)
                relative = f"male/{folder}/{level}-{number}.bin"
                source = processed / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(decoded)
                packed = gzip.compress(decoded, compresslevel=9, mtime=0)
                packed_sha = digest(packed)
                packed_relative = f"blobs/{packed_sha}.gz"
                (delivery / packed_relative).write_bytes(packed)
                info = source.stat()
                index["files"][relative] = {
                    "encoding": "gzip",
                    "file": packed_relative,
                    "bytes": len(packed),
                    "sha256": packed_sha,
                    "source_bytes": len(decoded),
                    "source_sha256": decoded_sha,
                    "source_mtime_ns": str(info.st_mtime_ns),
                }
                bricks.append({
                    "file": f"{level}-{number}.bin",
                    "x": number,
                    "y": 0,
                    "z": 0,
                    "extent": [1, 1, 1],
                    "bytes": len(decoded),
                    "sha256": decoded_sha,
                    **({"min_hu": -1024, "max_hu": 3071} if modality == "ct" else {}),
                })
            levels.append({
                "level": level,
                "factor": 2 ** level,
                "dimensions": [8, 8, 8],
                "grid": [count, 1, 1],
                "bytes": sum(brick["bytes"] for brick in bricks),
                "stored_bytes": sum(brick["bytes"] for brick in bricks),
                "bricks": bricks,
            })
        return {
            "version": 1,
            "subject": "male",
            "format": "u16le" if modality == "ct" else "rgb8-planar",
            "bytes_per_voxel": 2 if modality == "ct" else 3,
            "dimensions": [8, 8, 8],
            "spacing_mm": [1, 1, 1],
            "brick_size": 1,
            "levels": levels,
        }

    ct = volume("volume-v1", "ct")
    ct.update({
        "hu_offset": 1024,
        "origin_mm": [0, 0, 0],
        "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "slice_frames": list(range(8)),
        "histogram": {"min_hu": -1024, "max_hu": 3071, "bins": [0, 1]},
        "acquisition_segments": [{"start_index": 0, "end_index": 7}],
    })
    ct["inputs"] = {"corrections_sha256": hashes["corrections.json"]}
    hashes["volume-v1/manifest.json"] = write_json(subject / "volume-v1/manifest.json", ct)
    rgb = volume("rgb-volume-v1", "rgb")
    rgb["inputs"] = {
        "source_manifest_sha256": hashes["manifest.json"],
        "ct_volume_manifest_sha256": hashes["volume-v1/manifest.json"],
        "alignment_sha256": hashes["alignment.json"],
        "candidate_sha256": hashes["alignment-candidate-v2.json"],
        "reviewed_sha256": hashes["reviewed-alignment.json"],
    }
    rgb["alignment_knots"] = [
        {"color_frame": 0, "source": "human"},
        {"color_frame": 100, "source": "generated"},
    ]
    write_json(subject / "rgb-volume-v1/manifest.json", rgb)
    write_json(delivery / "index.json", index)
    # These files are explicitly outside the first public release.
    (subject / "rgb").mkdir()
    (subject / "rgb/0.png").write_bytes(b"private photograph")
    (processed / "female").mkdir()
    (processed / "female/manifest.json").write_text('{"private": true}')
    return processed, delivery


def test_release_is_deterministic_male_only_and_excludes_slices(tmp_path):
    processed, delivery = make_fixture(tmp_path)
    output = tmp_path / "public-release"
    first = build_release(processed, delivery, output, "https://visiblehuman-data.ballrollergames.com")
    second = build_release(processed, delivery, output, "https://visiblehuman-data.ballrollergames.com")
    assert first["release_sha256"] == second["release_sha256"]
    assert first["brick_references"] == 552
    assert first["objects"] == 552
    assert first["manifest"]["inputs"]["valid_reviewed_anchor_count"] == 1
    assert first["manifest"]["ct"]["hu_offset"] == 1024
    assert first["manifest"]["ct"]["slice_frames"] == list(range(8))
    assert all(
        brick["min_hu"] == -1024 and brick["max_hu"] == 3071
        for level in first["manifest"]["ct"]["levels"]
        for brick in level["bricks"]
    )
    encoded = Path(first["manifest_path"]).read_text()
    assert str(processed) not in encoded
    assert '"female"' not in encoded
    assert "private photograph" not in encoded
    assert "alignment_knots" not in encoded
    assert all(
        brick["url"].startswith("https://visiblehuman-data.ballrollergames.com/v1/objects/")
        for modality in ("ct", "rgb")
        for level in first["manifest"][modality]["levels"]
        for brick in level["bricks"]
    )


def test_release_refuses_stale_rgb_and_corrupt_delivery(tmp_path):
    processed, delivery = make_fixture(tmp_path)
    (processed / "male/reviewed-alignment.json").write_text('{"anchors": []}\n')
    with pytest.raises(ValueError, match="Registered RGB volume is stale"):
        build_release(processed, delivery, tmp_path / "stale-output", "https://visiblehuman-data.ballrollergames.com")

    processed, delivery = make_fixture(tmp_path / "corrupt")
    index = json.loads((delivery / "index.json").read_text())
    entry = next(iter(index["files"].values()))
    (delivery / entry["file"]).write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="Compressed delivery hash differs"):
        build_release(processed, delivery, tmp_path / "corrupt-output", "https://visiblehuman-data.ballrollergames.com")


def test_release_refuses_missing_ct_brick_hu_bounds(tmp_path):
    processed, delivery = make_fixture(tmp_path)
    ct_path = processed / "male/volume-v1/manifest.json"
    ct = json.loads(ct_path.read_text())
    del ct["levels"][2]["bricks"][0]["min_hu"]
    ct_hash = write_json(ct_path, ct)
    rgb_path = processed / "male/rgb-volume-v1/manifest.json"
    rgb = json.loads(rgb_path.read_text())
    rgb["inputs"]["ct_volume_manifest_sha256"] = ct_hash
    write_json(rgb_path, rgb)
    with pytest.raises(ValueError, match="HU bounds"):
        build_release(processed, delivery, tmp_path / "missing-hu-output", "https://visiblehuman-data.ballrollergames.com")


def test_release_refuses_incomplete_lod_wrong_encoding_and_changed_brick(tmp_path):
    processed, delivery = make_fixture(tmp_path / "incomplete")
    rgb_path = processed / "male/rgb-volume-v1/manifest.json"
    rgb = json.loads(rgb_path.read_text())
    rgb["levels"][2]["bricks"].pop()
    write_json(rgb_path, rgb)
    with pytest.raises(ValueError, match="expected 4"):
        build_release(processed, delivery, tmp_path / "incomplete-output", "https://visiblehuman-data.ballrollergames.com")

    processed, delivery = make_fixture(tmp_path / "metadata")
    index_path = delivery / "index.json"
    index = json.loads(index_path.read_text())
    next(iter(index["files"].values()))["encoding"] = "identity"
    write_json(index_path, index)
    with pytest.raises(ValueError, match="gzip delivery copy is missing"):
        build_release(processed, delivery, tmp_path / "metadata-output", "https://visiblehuman-data.ballrollergames.com")

    processed, delivery = make_fixture(tmp_path / "changed")
    source = next((processed / "male/volume-v1").glob("*.bin"))
    source.write_bytes(b"x" * source.stat().st_size)
    with pytest.raises(ValueError, match="Delivery copy is stale|Decoded source hash differs"):
        build_release(processed, delivery, tmp_path / "changed-output", "https://visiblehuman-data.ballrollergames.com")


def test_static_site_contains_only_the_allowlisted_runtime(tmp_path):
    release = tmp_path / "release"
    write_json(release / "release-config.json", {
        "readOnly": True,
        "automaticVolume": "preview",
        "consentVersion": 1,
        "releaseManifestUrl": "https://visiblehuman-data.ballrollergames.com/v1/releases/a/manifest.json",
    })
    viewer = Path(__file__).resolve().parents[1]
    destination = tmp_path / "site"
    result = build_site(viewer, release, destination)
    assert result["release_manifest_url"].endswith("/manifest.json")
    assert not (destination / "viewer.js").exists()
    assert not (destination / "server.mjs").exists()
    assert not (destination / "review-server.mjs").exists()
    assert "real human cadaver" in (destination / "index.html").read_text()
    assert "releaseManifestUrl" in (destination / "release-config.json").read_text()
    scripts = "\n".join(path.read_text() for path in destination.glob("*.mjs"))
    assert "/viewer.js" not in scripts
    assert "/api/review" not in scripts
    assert "/rgb/layers/" not in scripts
    assert 'method:"PUT"' not in scripts
    assert 'method:"POST"' not in scripts


def test_release_output_cannot_contain_or_wrap_private_roots(tmp_path):
    processed, delivery = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        build_release(processed, delivery, processed / "public", "https://example.test")
    with pytest.raises(ValueError, match="separate"):
        build_release(processed, delivery, tmp_path, "https://example.test")
