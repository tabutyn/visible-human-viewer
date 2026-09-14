#!/usr/bin/env python3
"""Build a registered, reviewable Visible Human slice pyramid.

This program intentionally has a useful no-data mode: `--synthetic-test` and
`--dry-run` exercise discovery/registration without reading or writing a full
dataset.  Source files are never modified.
"""
from __future__ import annotations

import argparse
import html
import hashlib
import json
import math
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from register_modalities import register_modalities

# Male archives use a_vmNNNN; female archives use avfNNNNa/b/c.
RGB_RE = re.compile(r"a_?v([mf])(\d{4})([abc]?)\.raw$", re.I)
CT_RE = re.compile(r"c_v([mf])(\d{4})\.fre$", re.I)
FROZEN_RE = re.compile(r"cvm(\d{4})f\.png$", re.I)
LOC_RE = re.compile(r"@\s*loc\s+([-+]?\d+(?:\.\d+)?)\s*mm", re.I)
SECTION_RE = re.compile(r"(?:section|image)\s*(?:number)?\s*[:=]?\s*(\d+)", re.I)
RECON_RE = re.compile(r"CT\s+Recon\s+[^/\r\n]+/(\d+)/(\d+)/(\d+)\s+@\s*loc", re.I)

RGB_SIZE = (2048, 1216)  # (width, height)
CT_SOURCE_SIZE = (512, 512)
CT_ANALYSIS_SIZE = (512, 304)
CT_OUTPUT_SIZE = (1024, 608)
GE_HEADER_BYTES = 3416


@dataclass
class Slice:
    path: str
    name: str
    depth: int
    kind: str
    position_mm: float | None = None
    section: int | None = None
    series: str | None = None
    exam: str | None = None
    pixel_size_x: float | None = None
    pixel_size_y: float | None = None
    fov_x: float | None = None
    fov_y: float | None = None
    dimension_x: int | None = None
    dimension_y: int | None = None
    center_r: float | None = None
    center_a: float | None = None


@dataclass
class Similarity:
    tx: float = 0.0
    ty: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0


def json_safe(value):
    """Return strict-JSON data; JavaScript rejects Python Infinity/NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_text(value, *, indent: int | None = 2) -> str:
    return json.dumps(json_safe(value), indent=indent, allow_nan=False)


def parse_rgb_filename(name: str) -> tuple[str, int] | None:
    """Return (subject, canonical colour-plane depth)."""
    match = RGB_RE.search(name)
    if not match:
        return None
    subject, number, plane = match.groups()
    base = int(number) - 1001
    # Male contains one image per millimetre. Female has a/b/c planes.
    depth = base if subject.lower() == "m" else base * 3 + {"a": 0, "b": 1, "c": 2}.get(plane.lower(), 0)
    return subject.lower(), depth


def _header_value(text: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}.*?:\s*([^\r\n]+)", text, re.I | re.M)
    return match.group(1).strip() if match else None


def _as_number(value: str | None, cast):
    try:
        return cast(value) if value is not None else None
    except ValueError:
        return None


def parse_ge_header(path: Path) -> dict:
    """Read only the GE header; CT payload is never loaded here."""
    with path.open("rb") as source:
        # Extracted .fre payloads start after 3,416 bytes.  NLM's companion
        # *.txt dumps are longer and put the human-readable image fields near
        # the end, so truncating those files silently drops exam/FOV metadata.
        size = None if path.suffix.lower() == ".txt" else GE_HEADER_BYTES
        raw_header = source.read(size)
        text = raw_header.decode("latin1", "replace")
    loc = LOC_RE.search(text)
    position = _as_number(_header_value(text, "Image location"), float)
    if position is None and loc:
        position = float(loc.group(1))
    section = SECTION_RE.search(text)
    recon = RECON_RE.search(text)
    exam = _header_value(text, "Exam number for this image") or (recon.group(1) if recon else None)
    series = _header_value(text, "Series number for this image") or (recon.group(2) if recon else None)
    result = {
        "position_mm": position,
        "section": int(section.group(1)) if section else int(recon.group(3)) if recon else None,
        "series": series,
        "exam": exam,
        "dimension_x": _as_number(_header_value(text, "Image dimension - X"), int),
        "dimension_y": _as_number(_header_value(text, "Image dimension - Y"), int),
        "pixel_size_x": _as_number(_header_value(text, "Image pixel size - X"), float),
        "pixel_size_y": _as_number(_header_value(text, "Image pixel size - Y"), float),
        "center_r": _as_number(_header_value(text, "Center R coord of plane image"), float),
        "center_a": _as_number(_header_value(text, "Center A coord of plane image"), float),
    }
    # Extracted Genesis/GE `.fre` files usually contain a binary image header
    # and no useful ASCII geometry.  The big-endian uint32 at byte 148 points
    # at the image-header block.  Decode only documented geometry fields and
    # retain text values when companion header dumps supplied them.
    if path.suffix.lower() != ".txt" and len(raw_header) >= 152:
        image_header = struct.unpack_from(">I", raw_header, 148)[0]
        try:
            binary = {
                "dimension_x": struct.unpack_from(">h", raw_header, image_header + 30)[0],
                "dimension_y": struct.unpack_from(">h", raw_header, image_header + 32)[0],
                "fov_x": struct.unpack_from(">f", raw_header, image_header + 42)[0],
                "fov_y": struct.unpack_from(">f", raw_header, image_header + 46)[0],
                "pixel_size_x": struct.unpack_from(">f", raw_header, image_header + 50)[0],
                "pixel_size_y": struct.unpack_from(">f", raw_header, image_header + 54)[0],
            }
        except struct.error:
            binary = {}
        plausible = (
            0 <= image_header < len(raw_header)
            and 1 <= binary.get("dimension_x", 0) <= 8192
            and 1 <= binary.get("dimension_y", 0) <= 8192
            and all(math.isfinite(binary.get(field, float("nan"))) and 0 < binary[field] < 5000
                    for field in ("fov_x", "fov_y", "pixel_size_x", "pixel_size_y"))
        )
        if plausible:
            for field in ("dimension_x", "dimension_y", "pixel_size_x", "pixel_size_y"):
                if result[field] is None:
                    result[field] = binary[field]
            result["fov_x"] = binary["fov_x"]
            result["fov_y"] = binary["fov_y"]
    return result


def parse_ct_filename(name: str, subject: str) -> int | None:
    frozen = FROZEN_RE.search(name)
    if frozen and subject == "male":
        return int(frozen.group(1)) - 1001
    match = CT_RE.search(name)
    if not match or match.group(1).lower() != subject[0].lower():
        return None
    number = int(match.group(2))
    # Female CT is acquired at 1 mm while the RGB coordinate has 3 planes/mm.
    return (number - 1001) * 3 if subject == "female" else number - 1001


def discover_rgb(root: Path, subject: str) -> list[Slice]:
    directory = root / subject.title() / "Fullcolor" / "fullbody"
    found: list[Slice] = []
    if not directory.is_dir():
        return found
    for path in directory.glob("*.raw"):
        parsed = parse_rgb_filename(path.name)
        if parsed and parsed[0] == subject[0]:
            found.append(Slice(str(path), path.name, parsed[1], "rgb"))
    return sorted(found, key=lambda item: item.depth)


def discover_ct(root: Path, subject: str, male_ct: str = "frozenCT") -> list[Slice]:
    # Some archive mirrors keep payloads in extracted/, others directly here.
    name = male_ct if subject == "male" else "normalCT"
    base = root / subject.title() / "Radiological" / name
    directories = [base / "extracted", base]
    found: list[Slice] = []
    seen: set[Path] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        patterns = ["*.fre"] if subject != "male" or male_ct != "frozenCT" else ["*.fre", "png/*.png"]
        for pattern in patterns:
          for path in directory.glob(pattern):
            if path in seen:
                continue
            depth = parse_ct_filename(path.name, subject)
            if depth is None:
                continue
            seen.add(path)
            header_path = base / "headers" / f"{path.stem}.txt" if path.suffix.lower() == ".png" else path
            header = parse_ge_header(header_path) if header_path.exists() else {}
            found.append(Slice(str(path), path.name, depth, "density", **header))
    return sorted(found, key=lambda item: item.depth)


def read_rgb(item: Slice, analysis_scale: float = 0.25) -> np.ndarray:
    data = np.fromfile(item.path, dtype=np.uint8)
    pixels = RGB_SIZE[0] * RGB_SIZE[1]
    if data.size != pixels * 3:
        raise ValueError(f"invalid RGB payload: {item.path}")
    # Visible Human RGB is planar R, G, B.
    planes = (data[:pixels], data[pixels:2*pixels], data[2*pixels:])
    image = np.stack([plane.reshape(RGB_SIZE[1], RGB_SIZE[0]) for plane in planes], axis=-1)
    return cv2.resize(image, None, fx=analysis_scale, fy=analysis_scale, interpolation=cv2.INTER_AREA)


def read_ct(item: Slice, analysis_size: tuple[int, int] = CT_ANALYSIS_SIZE) -> np.ndarray:
    if Path(item.path).suffix.lower() == ".png":
        raw = cv2.imread(item.path, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError(f"invalid 16-bit frozen CT PNG: {item.path}")
        # Frozen PNG represents unsigned stored density, matching baked U16BE.
        hu = raw.astype(np.float32) - 1024.0
        return cv2.resize(hu, analysis_size, interpolation=cv2.INTER_AREA)
    raw = np.fromfile(item.path, dtype=">u2", offset=GE_HEADER_BYTES)
    if raw.size != CT_SOURCE_SIZE[0] * CT_SOURCE_SIZE[1]:
        raise ValueError(f"invalid CT payload: {item.path}")
    hu = raw.reshape(CT_SOURCE_SIZE[1], CT_SOURCE_SIZE[0]).astype(np.float32) - 1024.0
    return cv2.resize(hu, analysis_size, interpolation=cv2.INTER_AREA)


def foreground(image: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "rgb":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.astype(np.uint8)
        border = np.concatenate((gray[:4].ravel(), gray[-4:].ravel(), gray[:, :4].ravel(), gray[:, -4:].ravel()))
        threshold = float(np.median(border) + 12)
        mask = (np.abs(gray.astype(np.float32) - np.median(border)) > max(8, threshold - np.median(border))).astype(np.uint8)
    else:
        gray = image.astype(np.float32)
        mask = (gray > -500).astype(np.uint8)
        gray = np.clip((gray + 1000) / 2500 * 255, 0, 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return gray, mask


def image_metric(image: np.ndarray, kind: str) -> float:
    gray, mask = foreground(image, kind)
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
    return float(edges[mask > 0].mean()) if np.any(mask) else float(edges.mean())


def registration_feature(image: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Modality-tolerant geometry image used for residuals and alignment."""
    gray, mask = foreground(image, kind)
    gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tissue_edges = cv2.Canny(gray_u8, 40, 120)
    silhouette_edges = cv2.Canny(mask * 255, 20, 80)
    edges = np.maximum(tissue_edges, silhouette_edges)
    # A smooth distance-to-edge field gives ECC useful gradients and measures
    # geometry rather than RGB/CT intensity differences.
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    feature = np.exp(-distance / 4.0).astype(np.float32)
    return feature, mask


def robust_boundaries(slices: list[Slice], metrics: list[float], expected_step: int) -> list[dict]:
    """Metadata + local MAD discontinuity detector, with ±8-slice context."""
    result: list[dict] = []
    for index in range(1, len(slices)):
        left, right = slices[index - 1], slices[index]
        metadata_gap = right.depth - left.depth > expected_step
        position_jump = left.position_mm is not None and right.position_mm is not None and abs(right.position_mm - left.position_mm) > expected_step * 2.1
        lo, hi = max(0, index - 8), min(len(metrics), index + 8)
        local = np.asarray([value for value in metrics[lo:hi] if math.isfinite(value)], dtype=float)
        if not local.size or not math.isfinite(metrics[index]):
            continue
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median))) or 1e-6
        score = abs(metrics[index] - median) / (1.4826 * mad)
        acquisition_change = (left.exam, left.series) != (right.exam, right.series)
        fov_change = any(abs((getattr(left, field) or 0) - (getattr(right, field) or 0)) > 1e-6 for field in ("pixel_size_x", "pixel_size_y", "dimension_x", "dimension_y"))
        if metadata_gap or position_jump or score > 4.5 or acquisition_change or fov_change:
            result.append({"boundary": index, "left": left.depth, "right": right.depth,
                           "reason": {"metadata_gap": metadata_gap, "position_jump": position_jump,
                                      "acquisition_change": acquisition_change, "fov_change": fov_change, "metric_z": round(score, 3)},
                           "gap": bool(metadata_gap)})
    # Residual spikes often span two or three adjacent anatomical planes.
    # Keep the strongest image-only candidate in each ±8-plane neighbourhood,
    # while never suppressing an independently known acquisition/FOV boundary.
    metadata = [entry for entry in result if any(entry["reason"].get(key) for key in ("metadata_gap", "position_jump", "acquisition_change", "fov_change"))]
    image_only = sorted((entry for entry in result if entry not in metadata), key=lambda entry: entry["reason"]["metric_z"], reverse=True)
    selected: list[dict] = []
    for entry in image_only:
        if all(abs(entry["boundary"] - kept["boundary"]) > 8 for kept in selected + metadata):
            selected.append(entry)
    return sorted(metadata + selected, key=lambda entry: entry["boundary"])


def affine_from_similarity(sim: Similarity, center: tuple[float, float]) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(center, sim.rotation_deg, sim.scale)
    matrix[:, 2] += (sim.tx, sim.ty)
    return matrix.astype(np.float32)


def similarity_from_affine(matrix: np.ndarray, center: tuple[float, float]) -> Similarity:
    """Decompose an affine already projected to a centered similarity."""
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = math.hypot(a, b)
    rotation = math.degrees(math.atan2(b, a))
    base = cv2.getRotationMatrix2D(center, rotation, scale)
    return Similarity(
        float(matrix[0, 2] - base[0, 2]),
        float(matrix[1, 2] - base[1, 2]),
        scale,
        rotation,
    )


def compose_similarity(parent: Similarity, local: Similarity, center: tuple[float, float] = (0, 0)) -> Similarity:
    """Compose two constrained transforms; no shear/reflection is introduced."""
    a = np.eye(3, dtype=float); b = np.eye(3, dtype=float)
    a[:2] = affine_from_similarity(parent, center); b[:2] = affine_from_similarity(local, center)
    result = a @ b
    return similarity_from_affine(result[:2], center)


def inverse_similarity(value: Similarity, center: tuple[float, float]) -> Similarity:
    matrix = np.eye(3, dtype=float)
    matrix[:2] = affine_from_similarity(value, center)
    return similarity_from_affine(np.linalg.inv(matrix)[:2], center)


def bake_scale(sim: Similarity, factor: float) -> Similarity:
    return Similarity(sim.tx * factor, sim.ty * factor, sim.scale, sim.rotation_deg)


def ct_metadata_transforms(slices: list[Slice]) -> dict[str, Similarity]:
    """Map scanner FOV/center changes into one physical CT canvas."""
    known = [item for item in slices if item.pixel_size_x and item.pixel_size_y]
    if not known:
        return {}
    canonical_pixel_x = max(item.pixel_size_x for item in known if item.pixel_size_x)
    canonical_pixel_y = max(item.pixel_size_y for item in known if item.pixel_size_y)
    widest = [item for item in known if abs((item.pixel_size_x or 0) - canonical_pixel_x) < 1e-6]
    canonical_r = float(np.median([item.center_r or 0.0 for item in widest]))
    canonical_a = float(np.median([item.center_a or 0.0 for item in widest]))
    result: dict[str, Similarity] = {}
    for item in slices:
        if not item.pixel_size_x or not item.pixel_size_y:
            continue
        # OpenCV applies this matrix with WARP_INVERSE_MAP: destination
        # canonical pixels are sampled in source scanner coordinates.
        scale = ((canonical_pixel_x / item.pixel_size_x) + (canonical_pixel_y / item.pixel_size_y)) / 2
        tx = (canonical_r - (item.center_r or 0.0)) / item.pixel_size_x
        # Scanner A increases opposite raster rows.  The source CT is square,
        # while registration compresses it to the RGB aspect ratio, so express
        # the physical row offset in that 304/512 analysis coordinate system.
        ty = ((item.center_a or 0.0) - canonical_a) / item.pixel_size_y
        ty *= CT_ANALYSIS_SIZE[1] / CT_SOURCE_SIZE[1]
        result[str(item.depth)] = Similarity(tx, ty, scale, 0.0)
    return result


def estimate_similarity(fixed: np.ndarray, moving: np.ndarray, kind: str = "rgb") -> tuple[Similarity, float]:
    """Rigid-in-plane alignment. ECC handles subpixel tx/ty/scale/rotation.

    Phase correlation is the robust initial translation. ECC's Euclidean
    pyramid is deliberately constrained: no shear, reflection or warp.
    """
    f_feature, fmask = registration_feature(fixed, kind)
    m_feature, mmask = registration_feature(moving, kind)
    fgray = foreground(fixed, kind)[0].astype(np.float32)
    mgray = foreground(moving, kind)[0].astype(np.float32)
    f = cv2.GaussianBlur(cv2.normalize(fgray, None, 0, 1, cv2.NORM_MINMAX), (0, 0), 1.0)
    m = cv2.GaussianBlur(cv2.normalize(mgray, None, 0, 1, cv2.NORM_MINMAX), (0, 0), 1.0)
    # Phase correlation supplies a translation-only fallback. Affine ECC is
    # used solely to measure scale/rotation; its result is projected back to a
    # true similarity matrix, so no affine shear reaches a baked output.
    shift, _ = cv2.phaseCorrelate(f_feature, m_feature)
    warp = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 75, 1e-6)
        _, warp = cv2.findTransformECC(f, m, warp, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        warp[:, 2] = shift
    a, b, tx = (float(warp[0, 0]), float(warp[0, 1]), float(warp[0, 2]))
    scale = (math.hypot(a, b) + math.hypot(float(warp[1, 0]), float(warp[1, 1]))) / 2
    rotation = math.degrees(math.atan2(b, a))
    ty = float(warp[1, 2])
    # Project ECC's affine result to scale + rotation only.
    angle = math.radians(rotation)
    warp = np.array([[scale * math.cos(angle), scale * math.sin(angle), tx], [-scale * math.sin(angle), scale * math.cos(angle), ty]], dtype=np.float32)
    aligned = cv2.warpAffine(m_feature, warp, (f.shape[1], f.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    overlap = np.logical_and(fmask > 0, cv2.warpAffine(mmask, warp, (f.shape[1], f.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0)
    residual = float(np.mean(np.abs(f_feature[overlap] - aligned[overlap]))) if np.any(overlap) else float("inf")
    return similarity_from_affine(warp, (f.shape[1] / 2, f.shape[0] / 2)), residual


def synthetic_transform_test() -> dict:
    image = np.zeros((256, 256), np.uint8)
    cv2.ellipse(image, (128, 130), (68, 91), 7, 0, 360, 180, -1)
    cv2.circle(image, (104, 103), 14, 0, -1)
    truth = Similarity(3.5, -2.0, 1.015, 2.0)
    forward = affine_from_similarity(truth, (128, 128))
    moving = cv2.warpAffine(image, forward, (256, 256))
    recovered, _ = estimate_similarity(image, moving)
    expected = truth
    warp = affine_from_similarity(recovered, (128, 128))
    corrected = cv2.warpAffine(moving, warp, (256, 256), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    mae = float(np.mean(np.abs(image.astype(float) - corrected.astype(float))))
    tolerances = {"translation_px": .5, "rotation_deg": .1, "scale": .002}
    passed = abs(recovered.tx - expected.tx) <= tolerances["translation_px"] and abs(recovered.ty - expected.ty) <= tolerances["translation_px"] and abs(recovered.rotation_deg - expected.rotation_deg) <= tolerances["rotation_deg"] and abs(recovered.scale - expected.scale) <= tolerances["scale"]
    return {"truth": asdict(expected), "recovered": asdict(recovered), "mae": mae, "tolerances": tolerances, "pass": passed}


def analyse_stack(
    slices: list[Slice],
    kind: str,
    expected_step: int,
    base_transforms: dict[str, Similarity] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """Detect breaks and calculate one cumulative similarity per segment.

    The image analysis is deliberately bounded to small images. The full-size
    image is opened only later, while baking the final transform.
    """
    if not slices:
        return [], {}
    reader = read_rgb if kind == "rgb" else read_ct
    # Registration needs ±8 windows, not an in-memory copy of the full body.
    # The bounded cache keeps a 5k-slice female stack below ~30 MiB analysis RAM.
    @lru_cache(maxsize=20)
    def image_at(index: int) -> np.ndarray:
        image = reader(slices[index])
        base = (base_transforms or {}).get(str(slices[index].depth))
        if base:
            center = (image.shape[1] / 2, image.shape[0] / 2)
            image = cv2.warpAffine(image, affine_from_similarity(base, center), (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=-1024.0 if kind == "density" else 0)
        return image
    pair: list[dict] = []
    for index in range(1, len(slices)):
        fixed, moving = image_at(index - 1), image_at(index)
        fg, mask = registration_feature(fixed, kind); mg, moving_mask = registration_feature(moving, kind)
        overlap_mask = np.logical_and(mask > 0, moving_mask > 0)
        raw = float(np.mean(np.abs(fg.astype(float)[overlap_mask] - mg.astype(float)[overlap_mask]))) if np.any(overlap_mask) else float("inf")
        overlap = float(np.count_nonzero(overlap_mask) / max(1, min(np.count_nonzero(mask), np.count_nonzero(moving_mask))))
        pair.append({"raw": raw, "overlap": overlap})
        if index % 250 == 0 or index == len(slices) - 1:
            print(f"analyse {kind}: compared {index}/{len(slices) - 1} adjacent layers", file=sys.stderr, flush=True)
    residuals = [0.0] + [item["raw"] for item in pair]
    boundaries = robust_boundaries(slices, residuals, expected_step)
    print(f"analyse {kind}: solving {len(boundaries)} candidate boundaries", file=sys.stderr, flush=True)
    finite_residuals = np.asarray([value for value in residuals[1:] if math.isfinite(value)], dtype=float)
    raw95 = float(np.percentile(finite_residuals, 95)) if finite_residuals.size else 0.0
    first_base = (base_transforms or {}).get(str(slices[0].depth), Similarity())
    transforms: dict[str, dict] = {str(slices[0].depth): asdict(first_base)}
    segment = Similarity()
    flagged = {entry["boundary"]: entry for entry in boundaries}
    for index, item in enumerate(slices):
        if index == 0:
            continue
        if index in flagged and not flagged[index]["gap"]:
            # Median projections from the two ±8-slice windows reduce a single
            # anatomy change dominating a slab-boundary estimate.
            left = np.median(np.stack([image_at(i) for i in range(max(0, index - 8), index)]), axis=0).astype(image_at(0).dtype)
            right = np.median(np.stack([image_at(i) for i in range(index, min(len(slices), index + 8))]), axis=0).astype(image_at(0).dtype)
            found, residual = estimate_similarity(left, right, kind)
            center = (left.shape[1] / 2, left.shape[0] / 2)
            flagged[index]["transform"] = asdict(found)
            left_gray, lm = registration_feature(left, kind); right_gray, rm = registration_feature(right, kind)
            raw_mask = np.logical_and(lm > 0, rm > 0)
            raw = float(np.mean(np.abs(left_gray.astype(float)[raw_mask] - right_gray.astype(float)[raw_mask]))) if np.any(raw_mask) else float("inf")
            improvement = max(0.0, (raw - residual) / raw) if raw and math.isfinite(raw) else 0.0
            warp = affine_from_similarity(found, center)
            corrected_mask = cv2.warpAffine(rm, warp, (lm.shape[1], lm.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            flagged[index].update({"raw_residual": raw, "corrected_residual": residual, "improvement": improvement, "raw_95th": raw95})
            intersection = np.count_nonzero(np.logical_and(lm > 0, corrected_mask > 0))
            flagged[index]["overlap"] = float(intersection / max(1, min(np.count_nonzero(lm), np.count_nonzero(corrected_mask))))
            fov_only = flagged[index]["reason"].get("fov_change") and not flagged[index]["reason"].get("acquisition_change")
            known_acquisition = bool(flagged[index]["reason"].get("acquisition_change") or flagged[index]["reason"].get("position_jump"))
            required_improvement = .30 if known_acquisition else .70
            accepted = bool(fov_only or (improvement >= required_improvement and flagged[index]["overlap"] >= .60))
            flagged[index]["accepted"] = accepted
            flagged[index]["flagged"] = not accepted
            applied = found if accepted and not fov_only else Similarity()
            flagged[index]["applied_transform"] = asdict(applied)
            if accepted and not fov_only:
                segment = compose_similarity(segment, found, center)
        base = (base_transforms or {}).get(str(item.depth), Similarity())
        transforms[str(item.depth)] = asdict(compose_similarity(segment, base, (image_at(index).shape[1] / 2, image_at(index).shape[0] / 2)))
    return boundaries, transforms


def align_ct_to_corrected_rgb(
    rgb: list[Slice],
    ct: list[Slice],
    rgb_transforms: dict[str, dict],
    ct_transforms: dict[str, dict],
) -> Similarity:
    """One cross-modality canonical-frame alignment after each stack is stable."""
    if not rgb or not ct:
        return Similarity()
    candidate = ct[len(ct) // 2]
    nearest = min(rgb, key=lambda item: abs(item.depth - candidate.depth))
    rgb_image = cv2.resize(read_rgb(nearest), (512, 304), interpolation=cv2.INTER_AREA)
    ct_image = read_ct(candidate)
    center = (rgb_image.shape[1] / 2, rgb_image.shape[0] / 2)
    rgb_segment = Similarity(**rgb_transforms.get(str(nearest.depth), {}))
    ct_segment = Similarity(**ct_transforms.get(str(candidate.depth), {}))
    rgb_image = cv2.warpAffine(
        rgb_image,
        affine_from_similarity(rgb_segment, center),
        (rgb_image.shape[1], rgb_image.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    ct_image = cv2.warpAffine(
        ct_image,
        affine_from_similarity(ct_segment, center),
        (ct_image.shape[1], ct_image.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderValue=-1024.0,
    )
    # Gradient images make RGB/CT metric-compatible while retaining anatomy.
    rgb_edges = cv2.Canny(foreground(rgb_image, "rgb")[0], 40, 120)
    ct_edges = cv2.Canny(foreground(ct_image, "density")[0], 40, 120)
    # A single cross-modality rotation/scale solve is under-constrained by the
    # scanner cradle and the cryosection block.  Use their shared anatomical
    # edge translation here; within-modality slab solves still estimate the
    # full similarity transform.  This avoids a false global body rotation.
    rgb_phase = cv2.GaussianBlur(foreground(rgb_edges, "rgb")[0].astype(np.float32), (0, 0), 1.0)
    ct_phase = cv2.GaussianBlur(foreground(ct_edges, "rgb")[0].astype(np.float32), (0, 0), 1.0)
    shift, response = cv2.phaseCorrelate(rgb_phase, ct_phase)
    if response < .10 or abs(shift[0]) > rgb_image.shape[1] / 3 or abs(shift[1]) > rgb_image.shape[0] / 3:
        shift = (0.0, 0.0)
    cross = Similarity(float(shift[0]), float(shift[1]), 1.0, 0.0)
    for depth, value in list(ct_transforms.items()):
        ct_transforms[depth] = asdict(compose_similarity(cross, Similarity(**value), center))
    return cross


def full_rgb(item: Slice) -> np.ndarray:
    data = np.fromfile(item.path, dtype=np.uint8); pixels = RGB_SIZE[0] * RGB_SIZE[1]
    if data.size != pixels * 3:
        raise ValueError(f"invalid RGB payload: {item.path}")
    planes = (data[:pixels], data[pixels:2 * pixels], data[2 * pixels:])
    return np.stack([plane.reshape(RGB_SIZE[1], RGB_SIZE[0]) for plane in planes], axis=-1)


def full_ct(item: Slice) -> np.ndarray:
    if Path(item.path).suffix.lower() == ".png":
        raw = cv2.imread(item.path, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError(f"invalid 16-bit frozen CT PNG: {item.path}")
        return raw
    raw = np.fromfile(item.path, dtype=">u2", offset=GE_HEADER_BYTES)
    if raw.size != CT_SOURCE_SIZE[0] * CT_SOURCE_SIZE[1]:
        raise ValueError(f"invalid CT payload: {item.path}")
    # OpenCV expects native-endian uint16; leaving the GE big-endian dtype in
    # place makes resize/warp treat values as byte-swapped on Apple Silicon.
    return raw.reshape(CT_SOURCE_SIZE[1], CT_SOURCE_SIZE[0]).astype(np.uint16)


def bake(subject: str, output: Path, rgb: list[Slice], ct: list[Slice], rgb_transforms: dict[str, dict], ct_transforms: dict[str, dict], resume: bool = False) -> tuple[list[dict], list[dict]]:
    """Bake lossless RGB PNG + big-endian uint16 density into common UV space."""
    rgb_dir, ct_dir = output / "rgb", output / "density"
    rgb_dir.mkdir(parents=True, exist_ok=True); ct_dir.mkdir(parents=True, exist_ok=True)
    rgb_layers: list[dict] = []; ct_layers: list[dict] = []

    def bake_rgb(item: Slice) -> dict:
        name = f"{item.depth:06d}.png"; destination = rgb_dir / name
        if resume and destination.exists():
            try:
                with Image.open(destination) as existing:
                    if existing.size == RGB_SIZE:
                        existing.verify()
                        return {"frame": item.depth, "file": f"rgb/{name}"}
            except (OSError, ValueError):
                pass
        image = full_rgb(item); sim = bake_scale(Similarity(**rgb_transforms.get(str(item.depth), {})), 4.0)
        corrected = cv2.warpAffine(image, affine_from_similarity(sim, (RGB_SIZE[0] / 2, RGB_SIZE[1] / 2)), RGB_SIZE, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        Image.fromarray(corrected).save(destination, compress_level=1)
        return {"frame": item.depth, "file": f"rgb/{name}"}

    def bake_density(item: Slice) -> dict:
        name = f"{item.depth:06d}.u16be"; destination = ct_dir / name
        if resume and destination.exists() and destination.stat().st_size == CT_OUTPUT_SIZE[0] * CT_OUTPUT_SIZE[1] * 2:
            return {"frame": item.depth, "file": f"density/{name}"}
        image = full_ct(item); sim = bake_scale(Similarity(**ct_transforms.get(str(item.depth), {})), 2.0)
        # CT's output raster is 1024×608, the RGB aspect ratio. This makes UV
        # coordinates identical while retaining unsigned scanner values.
        image = cv2.resize(image, CT_OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        corrected = cv2.warpAffine(image, affine_from_similarity(sim, (CT_OUTPUT_SIZE[0] / 2, CT_OUTPUT_SIZE[1] / 2)), CT_OUTPUT_SIZE, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        corrected.astype(">u2", copy=False).tofile(destination)
        return {"frame": item.depth, "file": f"density/{name}"}

    worker_limit = max(1, int(os.environ.get("VISIBLE_HUMAN_BAKE_WORKERS", "8")))
    with ThreadPoolExecutor(max_workers=min(worker_limit, os.cpu_count() or 1)) as executor:
        for layer in executor.map(bake_rgb, rgb):
            rgb_layers.append(layer)
            if len(rgb_layers) % 100 == 0 or len(rgb_layers) == len(rgb):
                print(f"bake rgb: {len(rgb_layers)}/{len(rgb)}", file=sys.stderr, flush=True)
        for layer in executor.map(bake_density, ct):
            ct_layers.append(layer)
            if len(ct_layers) % 100 == 0 or len(ct_layers) == len(ct):
                print(f"bake density: {len(ct_layers)}/{len(ct)}", file=sys.stderr, flush=True)
    return rgb_layers, ct_layers


def serialise_slices(slices: Iterable[Slice]) -> list[dict]:
    return [asdict(item) for item in slices]


def existing_baked_layers(output: Path) -> tuple[list[dict], list[dict]] | None:
    """Keep a completed bake addressable while registration is re-analysed."""
    path = output / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if not manifest.get("baked"):
        return None
    return manifest["rgb"]["layers"], manifest["density"]["layers"]


def modality_input_fingerprint(rgb: list[Slice], ct: list[Slice]) -> tuple[str, dict]:
    """Stable source revision without hashing tens of gigabytes of pixels."""
    digest = hashlib.sha256()
    for label, slices in (("rgb", rgb), ("ct", ct)):
        digest.update(label.encode())
        for item in slices:
            path = Path(item.path)
            try:
                stat = path.stat()
                token = f"{item.depth}:{item.name}:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                token = f"{item.depth}:{item.name}:missing"
            digest.update(token.encode())
    return digest.hexdigest(), {"rgb_layers": len(rgb), "ct_layers": len(ct), "fingerprint_method": "path-depth-size-mtime-sha256"}


def exact_sha256(path: Path) -> str:
    """Hash the exact on-disk bytes used to establish alignment freshness."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baked_modality_layers(output: Path, manifest_bytes: bytes) -> tuple[list[Slice], list[Slice], dict]:
    """Use corrected baked files only; source raws are not registration inputs."""
    manifest = json.loads(manifest_bytes)
    if not manifest.get("baked"):
        raise ValueError("modality registration requires a completed baked manifest")
    rgb = [Slice(str(output / item["file"]), Path(item["file"]).name, int(item["frame"]), "rgb") for item in manifest.get("rgb", {}).get("layers", [])]
    ct = [Slice(str(output / item["file"]), Path(item["file"]).name, int(item["frame"]), "density") for item in manifest.get("density", {}).get("layers", [])]
    if not rgb or not ct:
        raise ValueError("baked manifest must contain corrected RGB and density layers")
    return rgb, ct, manifest


def read_baked_rgb(item: Slice) -> np.ndarray:
    with Image.open(item.path) as source:
        image = np.asarray(source.convert("RGB"))
    return cv2.resize(image, CT_ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)


def read_baked_ct(item: Slice, width: int = CT_OUTPUT_SIZE[0], height: int = CT_OUTPUT_SIZE[1]) -> np.ndarray:
    values = np.fromfile(item.path, dtype=">u2")
    if values.size != width * height:
        raise ValueError(f"invalid baked U16BE CT payload: {item.path}")
    hu = values.reshape(height, width).astype(np.uint16).astype(np.float32) - 1024.0
    return cv2.resize(hu, CT_ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)


def modality_alignment(subject: str, rgb: list[Slice], ct: list[Slice], output: Path, sample_count: int = 32) -> dict:
    """Build the viewer-facing CT inverse-UV alignment; never bake pixels."""
    # These byte digests are captured before opening any representative images.
    # A server can therefore reject a profile after any manifest/correction edit.
    manifest = output / "manifest.json"
    corrections = output / "corrections.json"
    missing = [str(path) for path in (manifest, corrections) if not path.is_file()]
    if missing:
        raise ValueError(f"modality registration requires existing processed inputs: {', '.join(missing)}")
    manifest_bytes = manifest.read_bytes()
    corrections_bytes = corrections.read_bytes()
    inputs = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "corrections_sha256": hashlib.sha256(corrections_bytes).hexdigest()}
    rgb, ct, baked_manifest = baked_modality_layers(output, manifest_bytes)
    fingerprint, revision = modality_input_fingerprint(rgb, ct)
    result = register_modalities(rgb, ct, read_baked_rgb, read_baked_ct, sample_count=sample_count, holdout_count=24)
    return {
        "version": 1,
        "subject": subject,
        "reference": "color",
        "input_fingerprint": fingerprint,
        "input_revision": revision,
        "inputs": inputs,
        "orientation": result["orientation"],
        "coverage": {"color_depth": result["coverage_color"], "ct_frame": result["coverage_ct"]},
        "depth_knots": sorted(result["depth_knots"], key=lambda item: item["color_depth"]),
        "spatial_knots": sorted(result["spatial_knots"], key=lambda item: item["color_depth"]),
        "qc": {**result["qc"], "orientation_scores": result["orientation_scores"], "representatives": result["representatives"], "manifest_dimensions": {"rgb": baked_manifest.get("rgb", {}).get("width"), "density": baked_manifest.get("density", {}).get("width")}},
    }


def write_modality_alignment(output: Path, alignment: dict) -> None:
    """Activate a validated profile and emit compact machine/human QC."""
    output.mkdir(parents=True, exist_ok=True)
    qc = alignment["qc"]
    if not qc.get("accepted"):
        raise ValueError(
            f"modality registration failed QC: improvement={qc.get('improvement', 0):.1%}, "
            f"coverage={qc.get('coverage', 0):.1%}, "
            f"regional_failures={qc.get('regional_failures', [])}"
        )
    (output / "alignment.json").write_text(json_text(alignment))
    (output / "alignment-qc.json").write_text(json_text(qc))
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value.get('levels', 0)}</td>"
        f"<td>{value.get('before_symmetric_edge_distance', 0):.2f}</td>"
        f"<td>{value.get('after_symmetric_edge_distance', 0):.2f}</td>"
        f"<td>{value.get('internal_landmark_before_distance', 0):.2f}</td>"
        f"<td>{value.get('internal_landmark_after_distance', 0):.2f}</td>"
        f"<td>{value.get('coverage', 0):.1%}</td>"
        f"<td>{'yes' if value.get('accepted') else html.escape(', '.join(value.get('failures', [])))}</td></tr>"
        for name, value in qc.get("regional", {}).items()
    )
    style = "body{font:14px system-ui;margin:2rem;background:#111;color:#eee}table{border-collapse:collapse}th,td{padding:.45rem .8rem;border-bottom:1px solid #333;text-align:left}code{color:#bded85}"
    body = (
        f"<!doctype html><meta charset='utf-8'><title>{alignment['subject']} modality alignment QC</title><style>{style}</style>"
        f"<h1>{alignment['subject'].title()} CT → color alignment</h1>"
        f"<p>orientation: <code>{alignment['orientation']}</code>; improvement: {qc.get('improvement', 0):.1%}; "
        f"coverage: {qc.get('coverage', 0):.1%}; held out: {qc.get('holdout_count', 0)}; "
        f"orientation margin: {qc.get('orientation_margin', 0):.4f}</p>"
        f"<table><tr><th>Region</th><th>Levels</th><th>Silhouette before</th><th>Silhouette after</th>"
        f"<th>Landmarks before</th><th>Landmarks after</th><th>Overlap</th><th>Accepted</th></tr>{rows}</table>"
    )
    (output / "alignment-qc.html").write_text(body)


def incremental_rebuild(subject: str, output: Path, rgb: list[Slice], ct: list[Slice], boundary_id: str) -> dict:
    """Apply one reviewed boundary override and rebake only its right segment."""
    manifest_path = output / "manifest.json"
    corrections_path = output / "corrections.json"
    qc_path = output / "qc.json"
    overrides_path = output / "overrides.json"
    if not all(path.exists() for path in (manifest_path, corrections_path, qc_path, overrides_path)):
        raise ValueError("incremental rebuild requires an existing analysed and baked subject")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("baked"):
        raise ValueError("incremental rebuild requires an existing full bake")
    corrections = json.loads(corrections_path.read_text())
    qc = json.loads(qc_path.read_text())
    overrides = json.loads(overrides_path.read_text())
    entry = next((item for item in qc.get("flagged_boundaries", []) if item.get("id") == boundary_id), None)
    saved = overrides.get("boundaries", {}).get(boundary_id, {}).get("transform")
    if not entry or not saved:
        raise ValueError(f"missing boundary or override: {boundary_id}")
    modality = entry["modality"]
    key = "rgb" if modality == "rgb" else "density"
    transforms = corrections["transforms"][key]
    items = rgb if modality == "rgb" else ct
    center = (256.0, 152.0)
    baseline = Similarity(**entry.get("applied_transform", asdict(Similarity())))
    delta = compose_similarity(Similarity(**saved), inverse_similarity(baseline, center), center)
    for item in items[entry["boundary"]:]:
        transforms[str(item.depth)] = asdict(compose_similarity(delta, Similarity(**transforms[str(item.depth)]), center))
    if modality == "rgb":
        bake(subject, output, items[entry["boundary"]:], [], transforms, {})
    else:
        bake(subject, output, [], items[entry["boundary"]:], {}, transforms)
    entry.update({"accepted": True, "flagged": False, "override": True, "applied_transform": saved})
    for item in manifest.get("boundaries", []):
        if item.get("id") == boundary_id:
            item.update(entry)
    corrections_path.write_text(json_text(corrections))
    manifest_path.write_text(json_text(manifest))
    qc_path.write_text(json_text(qc))
    return {"subject": subject, "boundary": boundary_id, "modality": modality, "rebaked": len(items) - entry["boundary"]}


def write_report(output: Path, subject: str, rgb: list[Slice], ct: list[Slice], boundaries: list[dict], dry_run: bool, baked: tuple[list[dict], list[dict]] | None = None, transforms: dict[str, dict] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rgb_layers, ct_layers = baked if baked else (serialise_slices(rgb), serialise_slices(ct))
    manifest = {"version": 1, "subject": subject, "dry_run": dry_run, "baked": baked is not None,
                "rgb": {"width": 2048, "height": 1216, "format": "PNG" if baked else "planar-rgb-raw", "layers": rgb_layers},
                "density": {"width": 1024 if baked else 512, "height": 608 if baked else 512, "format": "U16BE", "layers": ct_layers}, "boundaries": boundaries,
                "gaps": [item for item in boundaries if item["gap"]]}
    (output / "manifest.json").write_text(json_text(manifest))
    (output / "corrections.json").write_text(json_text({"version": 1, "transforms": transforms or {}}))
    overrides = output / "overrides.json"
    if not overrides.exists():
        overrides.write_text(json_text({"version": 1, "boundaries": {}}))
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(b.get('id', b['boundary'])))}</td>"
        f"<td>{b.get('modality', '')}</td><td>{b['left']}→{b['right']}</td>"
        f"<td>{'yes' if b.get('accepted') else 'review'}</td>"
        f"<td>{b.get('improvement', 0):.1%}</td><td>{b.get('overlap', 0):.1%}</td>"
        f"<td><code>{html.escape(json.dumps(b.get('reason', {})))}</code></td></tr>"
        for b in boundaries
    )
    style = "body{font:14px system-ui;margin:2rem;background:#111;color:#eee}table{border-collapse:collapse;width:100%}th,td{padding:.45rem;border-bottom:1px solid #333;text-align:left}code{font-size:11px;color:#bded85}"
    (output / "qc.html").write_text(f"<!doctype html><meta charset='utf-8'><title>{subject} QC</title><style>{style}</style><h1>{subject.title()} registration QC</h1><p>dry run: {dry_run}; boundaries: {len(boundaries)}</p><table><tr><th>ID</th><th>modality</th><th>depth</th><th>status</th><th>improvement</th><th>overlap</th><th>reason</th></tr>{rows}</table>")
    (output / "qc.json").write_text(json_text({"subject": subject, "flagged_boundaries": boundaries}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("VISIBLE_HUMAN_ROOT", Path.home() / "Visible-Human-Project")),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--subject", choices=("male", "female"), default="male")
    parser.add_argument("--male-ct", default="frozenCT")
    parser.add_argument("--dry-run", action="store_true", help="scan metadata/emit report; do not bake pixels")
    parser.add_argument("--analyse", action="store_true", help="detect/solve boundaries and emit QC without baking full-size pixels")
    parser.add_argument("--analyse-modalities", action="store_true", help="analyse CT→color alignment and print JSON; never write/bake")
    parser.add_argument("--register-modalities", action="store_true", help="write alignment.json beside this subject without baking")
    parser.add_argument("--modality-samples", type=int, default=32, help="representative CT/color pairs for modality registration")
    parser.add_argument("--bake", action="store_true", help="analyse, register and bake corrected pixels")
    parser.add_argument("--bake-existing", action="store_true", help="bake using corrections.json and qc.json already present in --output")
    parser.add_argument("--modality", choices=("both", "rgb", "density"), default="both", help="limit --bake-existing to one modality")
    parser.add_argument("--resume", action="store_true", help="reuse already valid baked layer files")
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--rebuild-boundary", help="incrementally rebake the right segment of one reviewed boundary id")
    args = parser.parse_args()
    if args.synthetic_test:
        outcome = synthetic_transform_test()
        print(json.dumps(outcome, indent=2))
        return 0 if outcome["pass"] else 1
    output = args.output or args.root / "Processed" / "v1" / args.subject
    rgb = discover_rgb(args.root, args.subject)
    ct = discover_ct(args.root, args.subject, args.male_ct)
    if args.analyse_modalities or args.register_modalities:
        if args.analyse_modalities and args.register_modalities:
            parser.error("choose one of --analyse-modalities or --register-modalities")
        if any((args.bake, args.bake_existing, args.analyse, args.dry_run, args.rebuild_boundary)):
            parser.error("modality registration is non-destructive; do not combine it with bake/analyse/rebuild modes")
        alignment = modality_alignment(args.subject, rgb, ct, output, max(2, args.modality_samples))
        if args.register_modalities:
            write_modality_alignment(output, alignment)
            print(json_text({"subject": args.subject, "alignment": str(output / "alignment.json"), "orientation": alignment["orientation"], "improvement": alignment["qc"]["improvement"], "coverage": alignment["qc"]["coverage"]}, indent=None))
        else:
            print(json_text(alignment))
        return 0
    preserved_bake = existing_baked_layers(output)
    if args.rebuild_boundary:
        outcome = incremental_rebuild(args.subject, output, rgb, ct, args.rebuild_boundary)
        print(json.dumps(outcome))
        return 0
    if sum((args.bake, args.bake_existing, args.analyse, args.dry_run)) > 1:
        parser.error("--bake, --bake-existing, --analyse, and --dry-run are mutually exclusive")
    if args.bake_existing:
        corrections = json.loads((output / "corrections.json").read_text())
        qc = json.loads((output / "qc.json").read_text())
        transforms = corrections["transforms"]
        bake_rgb = rgb if args.modality in ("both", "rgb") else []
        bake_ct = ct if args.modality in ("both", "density") else []
        baked = bake(args.subject, output, bake_rgb, bake_ct, transforms["rgb"], transforms["density"], resume=args.resume)
        if args.modality != "both":
            if preserved_bake is None:
                raise ValueError("partial --bake-existing requires an existing full baked manifest")
            merged = (
                baked[0] if args.modality == "rgb" else preserved_bake[0],
                baked[1] if args.modality == "density" else preserved_bake[1],
            )
            write_report(output, args.subject, rgb, ct, qc["flagged_boundaries"], False, merged, transforms)
            print(json.dumps({"subject": args.subject, "modality": args.modality, "layers": len(bake_rgb) + len(bake_ct), "output": str(output), "baked": True, "reused_analysis": True}))
            return 0
        write_report(output, args.subject, rgb, ct, qc["flagged_boundaries"], False, baked, transforms)
        print(json.dumps({"subject": args.subject, "rgb": len(rgb), "density": len(ct), "output": str(output), "analysed": True, "baked": True, "reused_analysis": True}))
        return 0
    if args.bake or args.analyse:
        rgb_boundaries, rgb_transforms = analyse_stack(rgb, "rgb", 1)
        ct_step = 3 if args.subject == "female" else 1
        ct_boundaries, ct_transforms = analyse_stack(ct, "density", ct_step, ct_metadata_transforms(ct))
        for entry in rgb_boundaries:
            entry.update({"id": f"rgb-{entry['boundary']}", "modality": "rgb"})
        for entry in ct_boundaries:
            entry.update({"id": f"density-{entry['boundary']}", "modality": "density"})
        cross_transform = align_ct_to_corrected_rgb(rgb, ct, rgb_transforms, ct_transforms)
        # Overrides are segment transforms selected in the browser. Applying a
        # boundary value to its right-hand segment keeps prior baked slices
        # untouched and makes rebuilds deterministic/auditable.
        override_path = output / "overrides.json"
        override_data = json.loads(override_path.read_text()) if override_path.exists() else {"boundaries": {}}
        for entry in rgb_boundaries + ct_boundaries:
            saved = override_data.get("boundaries", {}).get(entry["id"], {}).get("transform")
            if not saved:
                continue
            target = rgb_transforms if entry in rgb_boundaries else ct_transforms
            all_items = rgb if target is rgb_transforms else ct
            center = (256.0, 152.0)
            baseline = Similarity(**entry.get("applied_transform", asdict(Similarity())))
            manual = Similarity(**saved)
            delta = compose_similarity(manual, inverse_similarity(baseline, center), center)
            for item in all_items[entry["boundary"]:]:
                current = Similarity(**target[str(item.depth)])
                target[str(item.depth)] = asdict(compose_similarity(delta, current, center))
            entry.update({"accepted": True, "flagged": False, "override": True, "applied_transform": saved})
        transforms = {"rgb": rgb_transforms, "density": ct_transforms, "ct_to_rgb": asdict(cross_transform)}
        baked = bake(args.subject, output, rgb, ct, rgb_transforms, ct_transforms) if args.bake else preserved_bake
        boundaries = rgb_boundaries + ct_boundaries
    else:
        boundaries, transforms, baked = [], {}, None
    write_report(output, args.subject, rgb, ct, boundaries, args.dry_run, baked, transforms)
    print(json.dumps({"subject": args.subject, "rgb": len(rgb), "density": len(ct), "output": str(output), "dry_run": args.dry_run, "analysed": bool(args.analyse or args.bake), "baked": bool(args.bake)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
