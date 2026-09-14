#!/usr/bin/env python3
"""Build a bounded-memory, WebGPU-ready CT volume pyramid.

The alignment viewer's baked density images are 1024x608 display rasters.  A
medical volume must remain square in-plane, so this builder applies the same
CT-only corrections in the 512x304 analysis coordinate system and conjugates
them back into the source 512x512 raster before writing the volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from preprocess import (
    CT_ANALYSIS_SIZE,
    CT_SOURCE_SIZE,
    Similarity,
    affine_from_similarity,
    ct_metadata_transforms,
    discover_ct,
    full_ct,
    json_text,
)

HU_OFFSET = 1024
HISTOGRAM_BINS = 4096
DEFAULT_BRICK_SIZE = 128
DEFAULT_LEVELS = 3


def volume_bytes(dimensions: tuple[int, int, int], bytes_per_voxel: int = 2) -> int:
    return math.prod(dimensions) * bytes_per_voxel


def square_affine(transform: Similarity) -> np.ndarray:
    """Convert a 512x304 analysis-space inverse warp into 512x512 space."""
    analysis = np.eye(3, dtype=np.float64)
    analysis[:2] = affine_from_similarity(
        transform, (CT_ANALYSIS_SIZE[0] / 2, CT_ANALYSIS_SIZE[1] / 2)
    )
    square_to_analysis = np.diag(
        [1.0, CT_ANALYSIS_SIZE[1] / CT_SOURCE_SIZE[1], 1.0]
    )
    return (np.linalg.inv(square_to_analysis) @ analysis @ square_to_analysis)[:2]


def corrected_ct(item, transforms: dict[str, dict]) -> np.ndarray:
    image = full_ct(item)
    transform = Similarity(**transforms.get(str(item.depth), {}))
    return cv2.warpAffine(
        image,
        square_affine(transform),
        CT_SOURCE_SIZE,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def load_transforms(subject_output: Path, slices) -> tuple[dict[str, dict], str | None]:
    corrections = subject_output / "corrections.json"
    if corrections.exists():
        raw = corrections.read_bytes()
        value = json.loads(raw)
        transforms = value.get("transforms", {}).get("density", {})
        if transforms:
            return transforms, hashlib.sha256(raw).hexdigest()
    generated = ct_metadata_transforms(slices)
    return {key: vars(value) for key, value in generated.items()}, None


def source_hash(slices) -> str:
    digest = hashlib.sha256()
    for item in slices:
        path = Path(item.path)
        digest.update(item.name.encode("utf8"))
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def brick_name(level: int, x: int, y: int, z: int) -> str:
    return f"level-{level}/{x:03d}-{y:03d}-{z:03d}.u16le"


def level_dimensions(slice_count: int, factor: int) -> tuple[int, int, int]:
    return (
        math.ceil(CT_SOURCE_SIZE[0] / factor),
        math.ceil(CT_SOURCE_SIZE[1] / factor),
        math.ceil(slice_count / factor),
    )


def build_level(
    destination: Path,
    level: int,
    factor: int,
    slices,
    transforms: dict[str, dict],
    brick_size: int,
) -> tuple[dict, list[int] | None]:
    width, height, depth = level_dimensions(len(slices), factor)
    grid = tuple(math.ceil(value / brick_size) for value in (width, height, depth))
    level_dir = destination / f"level-{level}"
    level_dir.mkdir(parents=True, exist_ok=True)
    bricks: list[dict] = []
    histogram = np.zeros(HISTOGRAM_BINS, dtype=np.uint64) if level == 0 else None

    for bz in range(grid[2]):
        buffers = {
            (bx, by): np.zeros((brick_size, brick_size, brick_size), dtype=np.uint16)
            for by in range(grid[1])
            for bx in range(grid[0])
        }
        output_z_start = bz * brick_size
        output_z_end = min(depth, output_z_start + brick_size)
        for output_z in range(output_z_start, output_z_end):
            source_start = output_z * factor
            source_end = min(len(slices), source_start + factor)
            images = [
                corrected_ct(slices[index], transforms).astype(np.uint32)
                for index in range(source_start, source_end)
            ]
            image = np.rint(np.mean(images, axis=0)).astype(np.uint16)
            if factor > 1:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if histogram is not None:
                histogram += np.bincount(
                    np.clip(image, 0, HISTOGRAM_BINS - 1).ravel(),
                    minlength=HISTOGRAM_BINS,
                ).astype(np.uint64)
            local_z = output_z - output_z_start
            for (bx, by), buffer in buffers.items():
                x0, y0 = bx * brick_size, by * brick_size
                tile = image[y0 : min(y0 + brick_size, height), x0 : min(x0 + brick_size, width)]
                buffer[local_z, : tile.shape[0], : tile.shape[1]] = tile
            progress_interval = max(1, depth // 100)
            if (output_z + 1) % progress_interval == 0 or output_z + 1 == depth:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "level": level,
                            "completed": output_z + 1,
                            "total": depth,
                        }
                    ),
                    flush=True,
                )

        valid_z = output_z_end - output_z_start
        for (bx, by), buffer in buffers.items():
            valid_x = min(brick_size, width - bx * brick_size)
            valid_y = min(brick_size, height - by * brick_size)
            valid = buffer[:valid_z, :valid_y, :valid_x]
            # Keep full brick rows/planes for aligned GPU copies, but omit
            # unused Z planes from the final brick in each level.
            payload = buffer[:valid_z].astype("<u2", copy=False).tobytes()
            relative = brick_name(level, bx, by, bz)
            (destination / relative).write_bytes(payload)
            bricks.append(
                {
                    "x": bx,
                    "y": by,
                    "z": bz,
                    "file": relative,
                    "extent": [valid_x, valid_y, valid_z],
                    "bytes": len(payload),
                    "min_hu": int(valid.min()) - HU_OFFSET,
                    "max_hu": int(valid.max()) - HU_OFFSET,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

    return (
        {
            "level": level,
            "factor": factor,
            "dimensions": [width, height, depth],
            "grid": list(grid),
            "bytes": volume_bytes((width, height, depth)),
            "stored_bytes": sum(item["bytes"] for item in bricks),
            "bricks": bricks,
        },
        histogram.tolist() if histogram is not None else None,
    )


def acquisition_segments(slices) -> list[dict]:
    if not slices:
        return []
    segments: list[dict] = []
    start = 0
    for index in range(1, len(slices)):
        left, right = slices[index - 1], slices[index]
        changed = (left.exam, left.series) != (right.exam, right.series)
        if changed:
            segments.append(
                {
                    "start_index": start,
                    "end_index": index - 1,
                    "exam": slices[start].exam,
                    "series": slices[start].series,
                }
            )
            start = index
    segments.append(
        {
            "start_index": start,
            "end_index": len(slices) - 1,
            "exam": slices[start].exam,
            "series": slices[start].series,
        }
    )
    return segments


def build_volume(
    root: Path,
    subject_output: Path,
    subject: str,
    brick_size: int = DEFAULT_BRICK_SIZE,
    levels: int = DEFAULT_LEVELS,
    memory_limit_gib: float = 8,
) -> dict:
    slices = discover_ct(root, subject)
    if not slices:
        raise ValueError(f"No {subject} CT slices found under {root}")
    if brick_size < 32 or brick_size > 256 or brick_size & (brick_size - 1):
        raise ValueError("brick size must be a power of two from 32 through 256")
    dimensions = level_dimensions(len(slices), 1)
    hard_limit = int(memory_limit_gib * 1024**3)
    if volume_bytes(dimensions) > hard_limit // 2:
        raise ValueError(
            f"native volume requires {volume_bytes(dimensions) / 1024**3:.2f} GiB; "
            "the voxel payload may use at most half the renderer memory limit"
        )

    transforms, corrections_sha = load_transforms(subject_output, slices)
    output = subject_output / "volume-v1"
    temporary = subject_output / f".volume-v1-building-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        pyramid = []
        histogram = None
        for level in range(levels):
            factor = 2**level
            built, values = build_level(
                temporary, level, factor, slices, transforms, brick_size
            )
            pyramid.append(built)
            if values is not None:
                histogram = values
        spacing_xy = max(
            [item.pixel_size_x for item in slices if item.pixel_size_x] or [1.0]
        )
        manifest = {
            "version": 1,
            "subject": subject,
            "format": "u16le-hu-plus-1024",
            "bytes_per_voxel": 2,
            "hu_offset": HU_OFFSET,
            "dimensions": list(dimensions),
            "spacing_mm": [spacing_xy, spacing_xy, 1.0],
            "origin_mm": [0.0, 0.0, 0.0],
            "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "slice_frames": [item.depth for item in slices],
            "brick_size": brick_size,
            "histogram": {
                "min_hu": -HU_OFFSET,
                "max_hu": HISTOGRAM_BINS - HU_OFFSET - 1,
                "bins": histogram,
            },
            "levels": pyramid,
            "acquisition_segments": acquisition_segments(slices),
            "inputs": {
                "ct_sha256": source_hash(slices),
                "corrections_sha256": corrections_sha,
            },
        }
        (temporary / "manifest.json").write_text(json_text(manifest) + "\n")
        previous = subject_output / ".volume-v1-previous"
        if previous.exists():
            shutil.rmtree(previous)
        if output.exists():
            output.rename(previous)
        temporary.rename(output)
        if previous.exists():
            shutil.rmtree(previous)
        print(json.dumps({"event": "complete", "output": str(output)}), flush=True)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("VISIBLE_HUMAN_ROOT", Path.home() / "Visible-Human-Project")),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subject", choices=("male", "female"), required=True)
    parser.add_argument("--brick-size", type=int, default=DEFAULT_BRICK_SIZE)
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--memory-limit-gib", type=float, default=8)
    args = parser.parse_args(argv)
    subject_output = args.output or args.root / "Processed" / "v1" / args.subject
    subject_output.mkdir(parents=True, exist_ok=True)
    build_volume(
        args.root,
        subject_output,
        args.subject,
        args.brick_size,
        args.levels,
        args.memory_limit_gib,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
