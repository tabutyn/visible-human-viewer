#!/usr/bin/env python3
"""Create resumable, lossless delivery copies; never rewrite source images or hashes."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


def sha(value):
    return hashlib.sha256(value).hexdigest()


def source_file(root, relative):
    value = root / relative
    if not value.resolve().is_relative_to(root) or value.is_symlink():
        raise ValueError(f"Unsafe source path: {relative}")
    return value


def pack_one(root, output, relative, encoding):
    source = source_file(root, relative)
    before = source.stat()
    raw = source.read_bytes()
    if encoding == "webp-lossless":
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode not in ("RGB", "RGBA"):
                raise ValueError("Only RGB/RGBA photograph pixels may be WebP encoded")
            buffer = io.BytesIO()
            image.save(buffer, "WEBP", lossless=True, quality=100, method=0, exact=True)
            packed = buffer.getvalue()
            with Image.open(io.BytesIO(packed)) as restored:
                if restored.convert(image.mode).tobytes() != image.tobytes():
                    raise ValueError(f"Lossless pixel verification failed: {relative}")
        extension = "webp"
    else:
        packed = gzip.compress(raw, compresslevel=3, mtime=0)
        if gzip.decompress(packed) != raw:
            raise ValueError(f"Lossless byte verification failed: {relative}")
        extension = "gz"
    after = source.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError(f"Source changed while packing: {relative}")
    # Do not make a delivery copy larger than the original photograph.
    if encoding == "webp-lossless" and len(packed) >= len(raw):
        return None
    digest = sha(packed)
    target = output / "blobs" / f"{digest}.{extension}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(packed)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    elif sha(target.read_bytes()) != digest:
        raise ValueError(f"Existing delivery blob is corrupt: {target.name}")
    return {"file": target.relative_to(output).as_posix(), "encoding": encoding,
            "source_sha256": sha(raw), "source_bytes": len(raw),
            "source_mtime_ns": str(after.st_mtime_ns), "bytes": len(packed), "sha256": digest}


def entries(root, subject, levels, include_slices, max_slices=None):
    result = []
    for folder in ("volume-v1", "rgb-volume-v1"):
        manifest = json.loads((root / subject / folder / "manifest.json").read_text())
        for level in sorted(manifest["levels"], key=lambda item: -item["level"]):
            if level["level"] not in levels:
                continue
            for brick in level["bricks"]:
                result.append((f"{subject}/{folder}/{brick['file']}", "gzip"))
    if include_slices:
        manifest = json.loads((root / subject / "manifest.json").read_text())
        for modality in ("rgb", "density"):
            layers = manifest[modality]["layers"]
            if max_slices is not None:
                # Representative selection across anatomy, not only air at the head.
                count = min(max_slices, len(layers))
                layers = [layers[round(i * (len(layers) - 1) / max(1, count - 1))] for i in range(count)]
            for layer in layers:
                name = layer["file"]
                if modality == "rgb" and name.endswith(".png"):
                    result.append((f"{subject}/{name}", "webp-lossless"))
                elif modality == "density" and name.endswith(".u16be"):
                    result.append((f"{subject}/{name}", "gzip"))
    return result


def pack(processed_root, destination, subject="male", levels=(2,), include_slices=False, max_slices=None, workers=4):
    root = Path(processed_root).resolve(strict=True)
    output = Path(destination).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Delivery output must be separate from the processed source tree")
    if subject not in ("male", "female"):
        raise ValueError("Unknown subject")
    if not 1 <= workers <= 8:
        raise ValueError("Use 1–8 compression workers")
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    lock = output / ".packing.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {"version": 1, "lossless": True, "files": {}}
        if index.get("version") != 1 or not isinstance(index.get("files"), dict):
            raise ValueError("Unsupported delivery index")
        jobs = entries(root, subject, set(levels), include_slices, max_slices)
        skipped = 0
        started = time.monotonic()
        def checkpoint():
            with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as temp:
                json.dump(index, temp, indent=2)
                temp.write("\n")
                temp_path = Path(temp.name)
            os.replace(temp_path, index_path)
        def encode(job):
            relative, encoding = job
            previous = index["files"].get(relative)
            source = source_file(root, relative)
            info = source.stat()
            target = output / previous["file"] if previous else None
            valid_target = target and target.resolve().is_relative_to(output) and target.is_file()
            if (previous and valid_target and previous["source_mtime_ns"] == str(info.st_mtime_ns)
                    and previous["source_bytes"] == info.st_size
                    and sha(target.read_bytes()) == previous["sha256"]):
                return relative, previous
            return relative, pack_one(root, output, relative, encoding)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for number, (relative, value) in enumerate(executor.map(encode, jobs), 1):
                if value:index["files"][relative] = value
                else:
                    index["files"].pop(relative, None);skipped += 1
                if number % 20 == 0 or number == len(jobs):
                    checkpoint()
                    print(json.dumps({"completed": number, "total": len(jobs), "seconds": round(time.monotonic()-started, 1)}), flush=True)
        saved = [index["files"][name] for name, _ in jobs if name in index["files"]]
        report = {"subject": subject, "requested_files": len(jobs), "packed_files": len(saved),
                  "original_bytes": sum(item["source_bytes"] for item in saved),
                  "delivery_bytes": sum(item["bytes"] for item in saved), "unchanged_pngs": skipped,
                  "lossless": True, "seconds": round(time.monotonic()-started, 2)}
        print(json.dumps(report), flush=True)
        return report
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", choices=("male", "female"), default="male")
    parser.add_argument("--levels", type=int, nargs="*", default=[2])
    parser.add_argument("--include-slices", action="store_true")
    parser.add_argument("--max-slices", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_slices is not None and args.max_slices < 1:
        parser.error("--max-slices must be positive")
    pack(args.processed_root, args.output, args.subject, args.levels, args.include_slices, args.max_slices, args.workers)
