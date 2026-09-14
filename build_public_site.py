#!/usr/bin/env python3
"""Build the allowlisted static Pages bundle for a pinned public release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


FILES = (
    "release-data.mjs", "volume-math.mjs",
    "volume-alignment.mjs", "viewer-math.mjs", "viewer.css", "LICENSE", "NOTICE.md",
)


def replace_function(source: str, start: str, end: str, replacement: str) -> str:
    left = source.find(start)
    right = source.find(end, left + len(start))
    if left < 0 or right < 0:
        raise ValueError(f"Public renderer source marker is missing: {start}")
    return source[:left] + replacement + "\n\n" + source[right:]


def public_renderer(source: str) -> str:
    source = replace_function(
        source, "function saveMedicalView() {", "const SHADER =",
        "function saveMedicalView() {}",
    )
    source = replace_function(
        source, "async function loadKnotTexture() {", "function updateLiveAlignment() {",
        "async function loadKnotTexture() {}",
    )
    source = replace_function(
        source, "async function buildVolume() {", "function releaseVolume(",
        'async function buildVolume() { setMessage("Volume builds are unavailable in the public release."); }',
    )
    forbidden = ('method:"PUT"', 'method:"POST"', 'method:"DELETE"', "/api/review", "/rgb/layers/")
    for value in forbidden:
        if value in source:
            raise ValueError(f"Private network operation remains in public renderer: {value}")
    return source


def build_site(viewer_root, release_root, output):
    viewer = Path(viewer_root).resolve(strict=True)
    release = Path(release_root).resolve(strict=True)
    destination = Path(output).absolute()
    if destination.resolve().is_relative_to(viewer) and destination.resolve() == viewer:
        raise ValueError("Site output cannot replace the viewer source")
    config_path = release / "release-config.json"
    config = json.loads(config_path.read_text())
    if config.get("readOnly") is not True or config.get("automaticVolume") != "preview" or not str(config.get("releaseManifestUrl", "")).startswith("https://"):
        raise ValueError("Release config is not a pinned HTTPS read-only preview")
    if destination.exists():
        raise FileExistsError(f"Site output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "site";staging.mkdir()
        shutil.copyfile(viewer / "public-index.html", staging / "index.html")
        shutil.copyfile(viewer / "public-entry.mjs", staging / "entry.mjs")
        for name in FILES:
            source = viewer / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Static source is unavailable: {name}")
            shutil.copyfile(source, staging / name)
        renderer = viewer / "medical-renderer.mjs"
        if renderer.is_symlink() or not renderer.is_file():
            raise ValueError("Static source is unavailable: medical-renderer.mjs")
        (staging / "medical-renderer.mjs").write_text(public_renderer(renderer.read_text()), encoding="utf-8")
        shutil.copyfile(config_path, staging / "release-config.json")
        (staging / "_headers").write_text(
            "/*\n"
            "  X-Content-Type-Options: nosniff\n"
            "  Referrer-Policy: no-referrer\n"
            "  Permissions-Policy: camera=(), microphone=(), geolocation=()\n"
            "  Cross-Origin-Resource-Policy: same-origin\n"
            "  Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' https://visiblehuman-data.ballrollergames.com; img-src 'self' blob:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'\n"
            "/release-config.json\n"
            "  Cache-Control: no-store\n",
            encoding="utf-8",
        )
        names = {path.name for path in staging.iterdir()}
        forbidden = {"viewer.js", "review-server.mjs", "review-workflow.mjs", "review-math.mjs"}
        if names & forbidden:
            raise ValueError("Editor runtime entered the public site")
        html = (staging / "index.html").read_text()
        for forbidden_text in ("Knot / slice editor", "Auto Align", "Save Human knot", "rgb/layers"):
            if forbidden_text in html:
                raise ValueError(f"Editor UI entered the public site: {forbidden_text}")
        scripts = "\n".join(path.read_text() for path in staging.glob("*.mjs"))
        for forbidden_text in ('method:"PUT"', 'method:"POST"', 'method:"DELETE"', "/api/review", "/rgb/layers/", "/viewer.js"):
            if forbidden_text in scripts:
                raise ValueError(f"Private runtime entered the public site: {forbidden_text}")
        os.rename(staging, destination)
    return {"output": str(destination), "files": sorted(path.name for path in destination.iterdir()), "release_manifest_url": config["releaseManifestUrl"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = build_site(args.viewer_root, args.release_root, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"Public site not built: {error}\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
