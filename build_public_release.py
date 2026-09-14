#!/usr/bin/env python3
"""Build a deterministic male-only public manifest from verified lossless bricks."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from urllib.parse import urlparse


EXPECTED_BRICKS = {0: 240, 1: 32, 2: 4}
SCHEMA = "visible-human-public-release/v1"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular JSON file is unavailable: {path.name}")
    payload = path.read_bytes()
    return json.loads(payload), payload


def canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def inside(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def public_base(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Public data base URL must be an HTTPS origin")
    if parsed.path not in ("", "/"):
        raise ValueError("Public data base URL must not include a path")
    return f"https://{parsed.netloc}"


def current_hash(path: Path) -> str | None:
    return digest_file(path) if path.is_file() and not path.is_symlink() else None


def validate_inputs(subject: Path, ct_manifest: dict, rgb_manifest: dict) -> dict:
    expected_ct = {"ct_sha256": None, "corrections_sha256": current_hash(subject / "corrections.json")}
    for key, value in expected_ct.items():
        if value is not None and ct_manifest.get("inputs", {}).get(key) != value:
            raise ValueError(f"CT volume is stale: {key} differs")
    sources = {
        "source_manifest_sha256": subject / "manifest.json",
        "ct_volume_manifest_sha256": subject / "volume-v1/manifest.json",
        "alignment_sha256": subject / "alignment.json",
        "candidate_sha256": subject / "alignment-candidate-v2.json",
        "reviewed_sha256": subject / "reviewed-alignment.json",
    }
    actual = {key: current_hash(path) for key, path in sources.items()}
    for key, value in actual.items():
        if rgb_manifest.get("inputs", {}).get(key) != value:
            raise ValueError(f"Registered RGB volume is stale: {key} differs; rebuild it before publication")
    return actual


def verified_brick(processed: Path, delivery: Path, index: dict, subject: Path, folder: str, brick: dict, base: str) -> dict:
    source_candidate = subject / folder / brick["file"]
    if source_candidate.is_symlink():
        raise ValueError(f"Unsafe or missing source brick: {brick['file']}")
    source = source_candidate.resolve()
    if not inside(processed, source) or not source.is_file():
        raise ValueError(f"Unsafe or missing source brick: {brick['file']}")
    relative = source.relative_to(processed).as_posix()
    entry = index.get("files", {}).get(relative)
    if not entry or entry.get("encoding") != "gzip":
        raise ValueError(f"Lossless gzip delivery copy is missing: {relative}")
    info = source.stat()
    if entry.get("source_mtime_ns") != str(info.st_mtime_ns) or entry.get("source_bytes") != info.st_size:
        raise ValueError(f"Delivery copy is stale: {relative}")
    if entry.get("source_sha256") != brick.get("sha256") or digest_file(source) != brick.get("sha256"):
        raise ValueError(f"Decoded source hash differs: {relative}")
    packed = (delivery / entry["file"]).resolve()
    if not inside(delivery, packed) or packed.is_symlink() or not packed.is_file() or packed.suffix != ".gz":
        raise ValueError(f"Unsafe or missing delivery blob: {relative}")
    payload = packed.read_bytes()
    if len(payload) != entry.get("bytes") or digest_bytes(payload) != entry.get("sha256"):
        raise ValueError(f"Compressed delivery hash differs: {relative}")
    decoded = gzip.decompress(payload)
    if len(decoded) != brick.get("bytes") or digest_bytes(decoded) != brick.get("sha256"):
        raise ValueError(f"Gzip round-trip differs: {relative}")
    object_key = f"v1/objects/{entry['sha256']}.gz"
    return {
        "x": int(brick["x"]), "y": int(brick["y"]), "z": int(brick["z"]),
        "extent": [int(value) for value in brick["extent"]],
        "bytes": int(brick["bytes"]), "sha256": brick["sha256"],
        "transfer_bytes": int(entry["bytes"]), "transfer_sha256": entry["sha256"],
        "object_key": object_key, "url": f"{base}/{object_key}",
    }


def volume_record(processed: Path, delivery: Path, index: dict, subject: Path, folder: str, source: dict, base: str, modality: str) -> dict:
    levels = []
    found = {int(level["level"]): level for level in source.get("levels", [])}
    if set(found) != set(EXPECTED_BRICKS):
        raise ValueError(f"{modality} must contain exactly L0, L1, and L2")
    for number in sorted(found):
        level = found[number]
        if len(level.get("bricks", [])) != EXPECTED_BRICKS[number]:
            raise ValueError(f"{modality} L{number} has {len(level.get('bricks', []))} bricks; expected {EXPECTED_BRICKS[number]}")
        bricks = [verified_brick(processed, delivery, index, subject, folder, brick, base) for brick in level["bricks"]]
        levels.append({
            "level": number, "factor": int(level.get("factor", 2**number)),
            "dimensions": [int(value) for value in level["dimensions"]],
            "grid": [int(value) for value in level["grid"]],
            "bytes": int(level["bytes"]), "stored_bytes": int(level["stored_bytes"]),
            "transfer_bytes": sum(item["transfer_bytes"] for item in bricks),
            "transfer_size_exact": True, "bricks": bricks,
        })
    allowed = {key: source[key] for key in ("version", "subject", "format", "bytes_per_voxel", "dimensions", "spacing_mm", "brick_size") if key in source}
    if modality.lower() == "ct":
        for key in ("hu_offset", "origin_mm", "direction", "slice_frames", "histogram", "acquisition_segments"):
            if key in source:
                allowed[key] = source[key]
    return {**allowed, "status": "ready", "levels": levels}


def default_view(subject: Path) -> dict:
    path = subject / "medical-view.json"
    value = read_json(path)[0] if path.exists() else {}
    defaults = {"preset": "lung", "yaw": -1.5707963267948966, "pitch": 0, "zoom": 1.15, "cameraPan": [0, 0], "planeOffset": 0, "planeNormal": [0, 0, 1]}
    return {key: value.get(key, fallback) for key, fallback in defaults.items()}


def build_release(processed_root, delivery_root, output_root, base_url: str) -> dict:
    processed, delivery = Path(processed_root).resolve(strict=True), Path(delivery_root).resolve(strict=True)
    output = Path(output_root).absolute()
    if (inside(processed, output) or inside(delivery, output)
            or inside(output, processed) or inside(output, delivery)):
        raise ValueError("Public release output must be separate from processed and delivery roots")
    subject = processed / "male"
    source_manifest, source_bytes = read_json(subject / "manifest.json")
    alignment, alignment_bytes = read_json(subject / "alignment.json")
    reviewed, reviewed_bytes = read_json(subject / "reviewed-alignment.json")
    candidate, candidate_bytes = read_json(subject / "alignment-candidate-v2.json")
    ct_manifest, ct_bytes = read_json(subject / "volume-v1/manifest.json")
    rgb_manifest, rgb_bytes = read_json(subject / "rgb-volume-v1/manifest.json")
    index, _ = read_json(delivery / "index.json")
    if index.get("version") != 1 or index.get("lossless") is not True:
        raise ValueError("Unsupported or lossy delivery index")
    inputs = validate_inputs(subject, ct_manifest, rgb_manifest)
    if ct_manifest.get("subject") != "male" or rgb_manifest.get("subject") != "male":
        raise ValueError("Only the male subject may be published")
    base = public_base(base_url)
    ct = volume_record(processed, delivery, index, subject, "volume-v1", ct_manifest, base, "CT")
    rgb = volume_record(processed, delivery, index, subject, "rgb-volume-v1", rgb_manifest, base, "RGB")
    for left, right in zip(ct["levels"], rgb["levels"]):
        if left["level"] != right["level"] or left["dimensions"] != right["dimensions"]:
            raise ValueError("CT and RGB level dimensions differ")
    brick_references = [item for volume in (ct, rgb) for level in volume["levels"] for item in level["bricks"]]
    if len(brick_references) != 552:
        raise ValueError(f"Release has {len(brick_references)} brick references; expected 552")
    objects = {item["transfer_sha256"] for item in brick_references}
    manifest = {
        "schema": SCHEMA, "subject": "male", "experimental": True,
        "attribution": {"credit": "Courtesy of the U.S. National Library of Medicine.", "source_url": "https://www.nlm.nih.gov/research/visible/visible_human.html", "terms_url": "https://www.nlm.nih.gov/databases/download/terms_and_conditions.html", "endorsed_by_nlm": False},
        "limitations": ["Experimental registration; anatomical correspondence is not assured.", "Not for diagnosis, treatment, surgical planning, or decisions about an individual.", "Modified data do not represent NLM's most current or most accurate data."],
        "inputs": {
            "source_manifest_sha256": digest_bytes(source_bytes), "alignment_sha256": digest_bytes(alignment_bytes),
            "reviewed_sha256": digest_bytes(reviewed_bytes), "candidate_sha256": digest_bytes(candidate_bytes),
            "ct_volume_manifest_sha256": digest_bytes(ct_bytes), "rgb_volume_manifest_sha256": digest_bytes(rgb_bytes),
            "valid_reviewed_anchor_count": sum(
                1 for knot in rgb_manifest.get("alignment_knots", [])
                if knot.get("source") == "human"
            ),
            "baked_alignment_knot_count": len(rgb_manifest.get("alignment_knots", [])),
        },
        "delivery": {
            "brick_references": len(brick_references),
            "unique_objects": len(objects),
            "deduplicated_bricks": len(brick_references) - len(objects),
        },
        "default_view": default_view(subject), "ct": ct, "rgb": rgb,
    }
    payload = canonical(manifest)
    serialized = payload.decode()
    for forbidden in (str(processed), str(delivery), '"female"', '"alignment_knots"', '"editor"'):
        if forbidden in serialized:
            raise ValueError(f"Forbidden release-manifest content: {forbidden}")
    release_sha = digest_bytes(payload)
    manifest_path = output / "v1" / "releases" / release_sha / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise ValueError("Content-addressed release manifest already differs")
    manifest_path.write_bytes(payload)
    config = {"readOnly": True, "automaticVolume": "preview", "consentVersion": 1, "releaseManifestUrl": f"{base}/v1/releases/{release_sha}/manifest.json"}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as temporary:
        json.dump(config, temporary, indent=2);temporary.write("\n");temporary_path = Path(temporary.name)
    os.replace(temporary_path, output / "release-config.json")
    return {
        "manifest": manifest, "manifest_path": str(manifest_path),
        "release_sha256": release_sha, "config": config,
        "brick_references": len(brick_references), "objects": len(objects),
        "deduplicated_bricks": len(brick_references) - len(objects),
        "transfer_bytes": sum(item["transfer_bytes"] for item in brick_references),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-base-url", required=True)
    args = parser.parse_args()
    try:
        result = build_release(args.processed_root, args.delivery_root, args.output, args.public_base_url)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"Public release not built: {error}\n")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    if peak_bytes >= 4 * 1024**3:
        parser.exit(1, f"Public release exceeded 4 GiB peak RSS: {peak_bytes} bytes\n")
    print(json.dumps({**{key: value for key, value in result.items() if key != "manifest"}, "peak_rss_bytes": peak_bytes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
