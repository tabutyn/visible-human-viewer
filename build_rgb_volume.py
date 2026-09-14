#!/usr/bin/env python3
"""Build an alignment-aware 8-bit RGB volume on the existing CT voxel grid."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def current_knots(subject_output: Path, maximum_frame: int) -> list[dict]:
    candidate_path = subject_output / "alignment-candidate-v2.json"
    baseline_path = subject_output / "alignment.json"
    profile = read_json(candidate_path if candidate_path.exists() else baseline_path)
    deleted = {int(value) for value in profile.get("deleted_knot_frames", [])}
    source = profile.get("parameter_knots") or [
        {**item, "z_position": item.get("ct_frame")}
        for item in profile.get("spatial_knots", [])
    ]
    by_frame: dict[int, dict] = {}
    for item in source:
        frame = item.get("color_frame")
        if frame is None and item.get("color_depth") is not None:
            frame = round(float(item["color_depth"]) * maximum_frame)
        if frame is None or int(frame) in deleted:
            continue
        z_position = item.get("z_position", item.get("ct_frame"))
        matrix = item.get("inverse_uv")
        if z_position is None or not isinstance(matrix, list) or len(matrix) != 6:
            continue
        by_frame[int(frame)] = {
            "color_frame": int(frame),
            "color_depth": float(item.get("color_depth", int(frame) / maximum_frame)),
            "z_position": float(z_position),
            "inverse_uv": [float(value) for value in matrix],
            "source": "candidate",
        }

    reviewed_path = subject_output / "reviewed-alignment.json"
    if reviewed_path.exists():
        expected = {
            "manifest_sha256": digest(subject_output / "manifest.json"),
            "alignment_sha256": digest(baseline_path),
        }
        for item in read_json(reviewed_path).get("anchors", []):
            frame = int(item.get("color_frame", -1))
            matrix = item.get("inverse_uv")
            if (
                frame < 0
                or frame in deleted
                or item.get("z_confirmed") is False
                or item.get("inputs") != expected
                or not isinstance(matrix, list)
                or len(matrix) != 6
            ):
                continue
            by_frame[frame] = {
                "color_frame": frame,
                "color_depth": float(item.get("color_depth", frame / maximum_frame)),
                "z_position": float(item["ct_frame"]),
                "inverse_uv": [float(value) for value in matrix],
                "source": "human",
            }

    knots = sorted(by_frame.values(), key=lambda item: item["color_depth"])
    if len(knots) < 2:
        raise ValueError("At least two valid spatial/Z alignment knots are required")
    if any(left["z_position"] >= right["z_position"] for left, right in zip(knots, knots[1:])):
        raise ValueError("RGB volume requires strictly increasing Z correspondence")
    return knots


def interpolate_knot(knots: list[dict], ct_frame: float) -> tuple[float, np.ndarray] | None:
    z_values = [item["z_position"] for item in knots]
    if ct_frame < z_values[0] or ct_frame > z_values[-1]:
        return None
    right = bisect.bisect_left(z_values, ct_frame)
    if right == 0:
        left = right = 0
        mix = 0.0
    elif right == len(knots):
        left = right = len(knots) - 1
        mix = 0.0
    else:
        left, right = right - 1, right
        span = z_values[right] - z_values[left]
        mix = (ct_frame - z_values[left]) / span if span else 0.0
    a, b = knots[left], knots[right]
    depth = a["color_depth"] + (b["color_depth"] - a["color_depth"]) * mix
    matrix = np.asarray(a["inverse_uv"], dtype=np.float64)
    matrix += (np.asarray(b["inverse_uv"], dtype=np.float64) - matrix) * mix
    return depth, matrix.reshape(2, 3)


def destination_to_source(matrix: np.ndarray, source_size: tuple[int, int], output_size: tuple[int, int]) -> np.ndarray:
    color_to_ct = np.eye(3, dtype=np.float64)
    color_to_ct[:2] = matrix
    ct_to_color = np.linalg.inv(color_to_ct)
    source_pixels = np.diag([source_size[0] - 1.0, source_size[1] - 1.0, 1.0])
    output_uv = np.diag([1.0 / max(1, output_size[0] - 1), 1.0 / max(1, output_size[1] - 1), 1.0])
    return (source_pixels @ ct_to_color @ output_uv)[:2]


class PlanarRgbLevel:
    """Read arbitrary slices from a brick-backed planar RGB level."""

    def __init__(self, root: Path, level: dict, brick_size: int):
        self.root = root
        self.level = level
        self.brick_size = brick_size
        self.width, self.height, self.depth = map(int, level["dimensions"])
        self.by_z: dict[int, list[dict]] = {}
        for brick in level["bricks"]:
            self.by_z.setdefault(int(brick["z"]), []).append(brick)

    @lru_cache(maxsize=32)
    def _brick(self, relative: str, valid_z: int) -> np.ndarray:
        values = np.frombuffer((self.root / relative).read_bytes(), dtype=np.uint8)
        expected = 3 * valid_z * self.brick_size * self.brick_size
        if values.size != expected:
            raise ValueError(f"RGB brick {relative} has {values.size} bytes; expected {expected}")
        return values.reshape(3, valid_z, self.brick_size, self.brick_size)

    def slice(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.depth:
            raise IndexError(index)
        output = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bz, local_z = divmod(index, self.brick_size)
        for brick in self.by_z.get(bz, []):
            valid_x, valid_y, valid_z = map(int, brick["extent"])
            if local_z >= valid_z:
                continue
            bx, by = int(brick["x"]), int(brick["y"])
            x0, y0 = bx * self.brick_size, by * self.brick_size
            tile = self._brick(brick["file"], valid_z)[:, local_z, :valid_y, :valid_x]
            output[y0 : y0 + valid_y, x0 : x0 + valid_x] = np.moveaxis(tile, 0, -1)
        return output


def build_downsampled_level(
    destination: Path,
    source: PlanarRgbLevel,
    ct_level: dict,
    brick_size: int,
) -> dict:
    level_number = int(ct_level["level"])
    factor = int(ct_level.get("factor", 2**level_number))
    width, height, depth = map(int, ct_level["dimensions"])
    grid = [math.ceil(value / brick_size) for value in (width, height, depth)]
    level_dir = destination / f"level-{level_number}"
    level_dir.mkdir(parents=True)
    bricks: list[dict] = []

    for bz in range(grid[2]):
        buffers = {
            (bx, by): np.zeros((brick_size, brick_size, brick_size, 3), dtype=np.uint8)
            for by in range(grid[1])
            for bx in range(grid[0])
        }
        z_start, z_end = bz * brick_size, min(depth, (bz + 1) * brick_size)
        for z in range(z_start, z_end):
            source_start, source_end = z * factor, min(source.depth, (z + 1) * factor)
            accumulated = np.zeros((source.height, source.width, 3), dtype=np.uint32)
            for source_z in range(source_start, source_end):
                accumulated += source.slice(source_z)
            averaged = np.rint(accumulated / max(1, source_end - source_start)).astype(np.uint8)
            image = cv2.resize(averaged, (width, height), interpolation=cv2.INTER_AREA)
            local_z = z - z_start
            for (bx, by), buffer in buffers.items():
                x0, y0 = bx * brick_size, by * brick_size
                tile = image[y0 : min(y0 + brick_size, height), x0 : min(x0 + brick_size, width)]
                buffer[local_z, : tile.shape[0], : tile.shape[1]] = tile
            if (z + 1) % max(1, depth // 100) == 0 or z + 1 == depth:
                print(json.dumps({"event": "progress", "level": level_number, "completed": z + 1, "total": depth}), flush=True)

        valid_z = z_end - z_start
        for (bx, by), buffer in buffers.items():
            valid_x, valid_y = min(brick_size, width - bx * brick_size), min(brick_size, height - by * brick_size)
            payload = b"".join(buffer[:valid_z, ..., channel].tobytes() for channel in range(3))
            name = f"level-{level_number}/{bx:03d}-{by:03d}-{bz:03d}.rgb8p"
            (destination / name).write_bytes(payload)
            bricks.append({"x": bx, "y": by, "z": bz, "file": name, "extent": [valid_x, valid_y, valid_z],
                           "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})

    return {"level": level_number, "factor": factor, "dimensions": [width, height, depth], "grid": grid,
            "bytes": width * height * depth * 3, "stored_bytes": sum(item["bytes"] for item in bricks), "bricks": bricks}


def build_rgb_volume(subject_output: Path, brick_size: int | None = None) -> dict:
    source_manifest_path = subject_output / "manifest.json"
    ct_manifest_path = subject_output / "volume-v1" / "manifest.json"
    source_manifest, ct_manifest = read_json(source_manifest_path), read_json(ct_manifest_path)
    rgb_layers = sorted(source_manifest["rgb"]["layers"], key=lambda item: item["frame"])
    if not rgb_layers:
        raise ValueError("No processed RGB layers are available")
    frames = [int(item["frame"]) for item in rgb_layers]
    by_frame = {int(item["frame"]): subject_output / item["file"] for item in rgb_layers}
    maximum_frame = max(frames)
    knots = current_knots(subject_output, maximum_frame)
    level = min(ct_manifest["levels"], key=lambda item: item["level"])
    width, height, depth = map(int, level["dimensions"])
    if [width, height, depth] != list(map(int, ct_manifest["dimensions"])):
        raise ValueError("Native CT level is required for same-resolution RGB")
    brick_size = int(brick_size or ct_manifest["brick_size"])
    grid = [math.ceil(value / brick_size) for value in (width, height, depth)]
    slice_frames = ct_manifest.get("slice_frames", list(range(depth)))

    @lru_cache(maxsize=4)
    def image(frame: int) -> np.ndarray:
        value = cv2.imread(str(by_frame[frame]), cv2.IMREAD_COLOR)
        if value is None:
            raise ValueError(f"Cannot decode RGB frame {frame}")
        return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)

    def source_at(frame: float) -> np.ndarray:
        right = bisect.bisect_left(frames, frame)
        if right <= 0:
            return image(frames[0])
        if right >= len(frames):
            return image(frames[-1])
        a, b = frames[right - 1], frames[right]
        mix = (frame - a) / max(1, b - a)
        if mix <= 1e-6:
            return image(a)
        if mix >= 1 - 1e-6:
            return image(b)
        return cv2.addWeighted(image(a), 1 - mix, image(b), mix, 0)

    output = subject_output / "rgb-volume-v1"
    temporary = subject_output / f".rgb-volume-v1-building-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    level_dir = temporary / "level-0"
    level_dir.mkdir(parents=True)
    bricks: list[dict] = []
    try:
        for bz in range(grid[2]):
            buffers = {
                (bx, by): np.zeros((brick_size, brick_size, brick_size, 3), dtype=np.uint8)
                for by in range(grid[1])
                for bx in range(grid[0])
            }
            z_start, z_end = bz * brick_size, min(depth, (bz + 1) * brick_size)
            for z in range(z_start, z_end):
                mapped = interpolate_knot(knots, float(slice_frames[z]))
                if mapped is not None:
                    color_depth, matrix = mapped
                    source = source_at(color_depth * maximum_frame)
                    affine = destination_to_source(matrix, (source.shape[1], source.shape[0]), (width, height))
                    aligned = cv2.warpAffine(
                        source, affine, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
                    )
                    local_z = z - z_start
                    for (bx, by), buffer in buffers.items():
                        x0, y0 = bx * brick_size, by * brick_size
                        tile = aligned[y0:min(y0 + brick_size, height), x0:min(x0 + brick_size, width)]
                        buffer[local_z, :tile.shape[0], :tile.shape[1]] = tile
                if (z + 1) % max(1, depth // 100) == 0 or z + 1 == depth:
                    print(json.dumps({"event": "progress", "level": 0, "completed": z + 1, "total": depth}), flush=True)

            valid_z = z_end - z_start
            for (bx, by), buffer in buffers.items():
                valid_x, valid_y = min(brick_size, width - bx * brick_size), min(brick_size, height - by * brick_size)
                payload = b"".join(buffer[:valid_z, ..., channel].tobytes() for channel in range(3))
                name = f"level-0/{bx:03d}-{by:03d}-{bz:03d}.rgb8p"
                (temporary / name).write_bytes(payload)
                bricks.append({"x": bx, "y": by, "z": bz, "file": name, "extent": [valid_x, valid_y, valid_z],
                               "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})

        native_level = {"level": 0, "factor": 1, "dimensions": [width, height, depth], "grid": grid,
                        "bytes": width * height * depth * 3, "stored_bytes": sum(item["bytes"] for item in bricks), "bricks": bricks}
        pyramid = [native_level]
        native_source = PlanarRgbLevel(temporary, native_level, brick_size)
        for ct_level in sorted(ct_manifest["levels"], key=lambda item: int(item["level"])):
            if int(ct_level["level"]) > 0:
                pyramid.append(build_downsampled_level(temporary, native_source, ct_level, brick_size))

        inputs = {
            "source_manifest_sha256": digest(source_manifest_path),
            "ct_volume_manifest_sha256": digest(ct_manifest_path),
            "alignment_sha256": digest(subject_output / "alignment.json"),
            "candidate_sha256": digest(subject_output / "alignment-candidate-v2.json") if (subject_output / "alignment-candidate-v2.json").exists() else None,
            "reviewed_sha256": digest(subject_output / "reviewed-alignment.json") if (subject_output / "reviewed-alignment.json").exists() else None,
        }
        manifest = {
            "version": 1, "subject": ct_manifest["subject"], "format": "rgb8-planar", "bytes_per_voxel": 3,
            "dimensions": [width, height, depth], "spacing_mm": ct_manifest["spacing_mm"], "brick_size": brick_size,
            "levels": pyramid,
            "alignment_knots": knots, "inputs": inputs,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        previous = subject_output / ".rgb-volume-v1-previous"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--subject", choices=("male", "female"), required=True)
    parser.add_argument("--brick-size", type=int)
    args = parser.parse_args()
    build_rgb_volume(args.processed / args.subject, args.brick_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
