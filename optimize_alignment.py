#!/usr/bin/env python3
"""Build a bounded-memory six-curve RGB <-> CT registration candidate.

The baseline ``alignment.json`` is read-only.  This command writes a separate
candidate which must be reviewed and promoted explicitly by the viewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.format import open_memmap
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

import register_modalities as registration
from preprocess import CT_ANALYSIS_SIZE, Slice, read_baked_ct, read_baked_rgb
from suggest_alignment import color_mask, density_mask

LEVELS = ((128, 76), (256, 152), (512, 304))
PARAMETER_NAMES = ("z_position", "y_position", "x_position", "rotation_deg", "y_scale", "x_scale")
REGIONS = (("head", 0, .12), ("neck", .12, .18), ("chest", .18, .35),
           ("abdomen", .35, .52), ("pelvis", .52, .68), ("knee", .68, .84), ("ankle", .84, 1.01))


@dataclass
class SpatialParameters:
    x_position: float = 0.0
    y_position: float = 0.0
    rotation_deg: float = 0.0
    x_scale: float = 1.0
    y_scale: float = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def orientation_pixels(name: str, width: int, height: int) -> np.ndarray:
    return registration._orientation_pixels(name, width, height)


def display_forward(parameters: SpatialParameters, width=512, height=304) -> np.ndarray:
    """Oriented CT pixels -> displayed color pixels, centered, clockwise."""
    if not (.25 <= parameters.x_scale <= 4 and .25 <= parameters.y_scale <= 4):
        raise ValueError("scale is outside .25..4")
    angle = math.radians(parameters.rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    linear = np.array([[cosine * parameters.x_scale, -sine * parameters.y_scale],
                       [sine * parameters.x_scale, cosine * parameters.y_scale]], dtype=float)
    center = np.array([(width - 1) / 2, (height - 1) / 2], dtype=float)
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = linear
    matrix[:2, 2] = center + np.array([parameters.x_position, parameters.y_position]) - linear @ center
    return matrix


def parameters_to_inverse_uv(parameters: SpatialParameters, orientation="flip_y", width=512, height=304) -> list[float]:
    forward = display_forward(parameters, width, height)
    sample_pixels = orientation_pixels(orientation, width, height) @ np.linalg.inv(forward)
    to_pixels = np.diag([width - 1., height - 1., 1.])
    to_uv = np.diag([1 / max(1., width - 1), 1 / max(1., height - 1), 1.])
    uv = to_uv @ sample_pixels @ to_pixels
    answer = [float(uv[0, 0]), float(uv[0, 1]), float(uv[0, 2]),
              float(uv[1, 0]), float(uv[1, 1]), float(uv[1, 2])]
    if not np.all(np.isfinite(answer)) or abs(np.linalg.det(sample_pixels[:2, :2])) < 1e-4:
        raise ValueError("singular transform")
    return answer


def inverse_uv_to_parameters(inverse_uv, orientation="flip_y", width=512, height=304) -> SpatialParameters:
    values = np.asarray(inverse_uv, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("inverse_uv must have six finite values")
    uv = np.eye(3); uv[:2] = values.reshape(2, 3)
    to_pixels = np.diag([width - 1., height - 1., 1.])
    sample = to_pixels @ uv @ np.linalg.inv(to_pixels)
    oriented_sample = np.linalg.inv(orientation_pixels(orientation, width, height)) @ sample
    if abs(np.linalg.det(oriented_sample[:2, :2])) < 1e-6:
        raise ValueError("singular transform")
    forward = np.linalg.inv(oriented_sample)
    sx, sy = np.linalg.norm(forward[:2, 0]), np.linalg.norm(forward[:2, 1])
    dot = float(np.dot(forward[:2, 0], forward[:2, 1]) / max(1e-9, sx * sy))
    if not (.1 <= sx <= 10 and .1 <= sy <= 10) or abs(dot) > .08:
        raise ValueError("transform contains implausible scale or shear")
    rotation = math.degrees(math.atan2(forward[1, 0], forward[0, 0]))
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    translated_center = forward[:2, :2] @ center + forward[:2, 2]
    shift = translated_center - center
    return SpatialParameters(float(shift[0]), float(shift[1]), rotation, float(sx), float(sy))


def interpolate_reviewed_parameters(reviewed_by_frame, color_frame):
    """Interpolate manual six-parameter ground truth inside its covered range."""
    rows = sorted((int(frame), item.get("parameters")) for frame, item in reviewed_by_frame.items()
                  if isinstance(item.get("parameters"), dict))
    if len(rows) < 2 or color_frame < rows[0][0] or color_frame > rows[-1][0]: return None
    frames = [row[0] for row in rows]

    def value(name): return [float(row[1][name]) for row in rows]

    return SpatialParameters(
        float(np.interp(color_frame, frames, value("x_position"))),
        float(np.interp(color_frame, frames, value("y_position"))),
        float(np.interp(color_frame, frames, value("rotation_deg"))),
        float(math.exp(np.interp(color_frame, frames, np.log(value("x_scale"))))),
        float(math.exp(np.interp(color_frame, frames, np.log(value("y_scale"))))))


def _layer(directory: Path, item: dict, kind: str) -> Slice:
    path = (directory / item["file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("manifest layer escapes processed directory")
    return Slice(str(path), path.name, int(item["frame"]), kind)


def _read(directory: Path, manifest: dict, item: dict, kind: str) -> np.ndarray:
    layer = _layer(directory, item, kind)
    if kind == "rgb":
        return read_baked_rgb(layer)
    density = manifest["density"]
    return read_baked_ct(layer, int(density["width"]), int(density["height"]))


def feature_planes(image: np.ndarray, kind: str, subject: str, depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if kind == "rgb":
        mask, warnings = color_mask(image, subject)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        mask, warnings = density_mask(image, subject, depth)
        gray = np.clip((image.astype(np.float32) + 300.) * (255. / 1600.), 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray, 35, 110)
    edges = np.maximum(edges, cv2.Canny(mask * 255, 20, 80))
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    sdf = np.exp(-distance / 4.).astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    scale = float(np.percentile(gradient[mask > 0], 95)) if np.any(mask) else 1.
    gradient = np.clip(gradient / max(1., scale), 0, 1)
    return mask.astype(np.uint8), sdf, gradient, warnings


def descriptor(mask: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if not len(xs):
        return np.full(18, 4., np.float32)
    box_width, box_height = np.ptp(xs) + 1, np.ptp(ys) + 1
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    moments = cv2.HuMoments(cv2.moments(crop)).ravel()[:3]
    base = [box_width / max(1, box_height), len(xs) / max(1, box_width * box_height),
            *np.sign(moments) * np.log1p(np.abs(moments))]
    row_counts = crop.sum(1).astype(float) / max(1, box_width)
    col_counts = crop.sum(0).astype(float) / max(1, box_height)
    base.extend(np.interp(np.linspace(0, len(row_counts) - 1, 4), np.arange(len(row_counts)), row_counts))
    base.extend(np.interp(np.linspace(0, len(col_counts) - 1, 4), np.arange(len(col_counts)), col_counts))
    values = gradient[mask > 0]
    base.extend(np.histogram(values, bins=5, range=(0, 1), density=True)[0] / 5)
    return np.asarray(base, np.float32)


class FeatureCache:
    """Disk-backed uint8/float16 pyramids; only a slice window is resident."""
    def __init__(self, directory: Path, manifest: dict, subject: str, rebuild=False):
        self.directory, self.manifest, self.subject = directory, manifest, subject
        self.root = directory / ".alignment-cache-v2"
        self.layers = {"rgb": manifest["rgb"]["layers"], "density": manifest["density"]["layers"]}
        fingerprint = hashlib.sha256(json.dumps({
            kind: [(item["frame"], item["file"]) for item in layers] for kind, layers in self.layers.items()
        }, sort_keys=True).encode()).hexdigest()
        metadata_file = self.root / "metadata.json"
        ready = False
        if not rebuild and metadata_file.exists():
            try: ready = json.loads(metadata_file.read_text()).get("fingerprint") == fingerprint
            except (OSError, ValueError): pass
        if not ready:
            if self.root.exists(): shutil.rmtree(self.root)
            self.root.mkdir(parents=True)
            self._build(fingerprint)
        try: self.cache_build_peak_gib = float(json.loads(metadata_file.read_text()).get("peak_working_set_gib", 0))
        except (OSError, ValueError, TypeError): self.cache_build_peak_gib = 0.
        self.arrays = {}
        for kind, layers in self.layers.items():
            for width, height in LEVELS:
                for plane in ("mask", "sdf", "gradient"):
                    self.arrays[(kind, width, plane)] = np.load(self.root / f"{kind}-{width}-{plane}.npy", mmap_mode="r")
        self.descriptors = {kind: np.load(self.root / f"{kind}-descriptors.npy", mmap_mode="r") for kind in self.layers}

    def _build(self, fingerprint):
        warnings = []
        for kind, layers in self.layers.items():
            maps = {}
            for width, height in LEVELS:
                maps[(width, "mask")] = open_memmap(self.root / f"{kind}-{width}-mask.npy", mode="w+", dtype=np.uint8, shape=(len(layers), height, width))
                for plane in ("sdf", "gradient"):
                    maps[(width, plane)] = open_memmap(self.root / f"{kind}-{width}-{plane}.npy", mode="w+", dtype=np.float16, shape=(len(layers), height, width))
            desc = open_memmap(self.root / f"{kind}-descriptors.npy", mode="w+", dtype=np.float32, shape=(len(layers), 18))
            max_frame = max(1, max(item["frame"] for item in layers))
            for index, item in enumerate(layers):
                image = _read(self.directory, self.manifest, item, kind)
                depth = item["frame"] / max_frame if kind == "rgb" else 0.
                if kind == "density": image = registration.orient(image, "flip_y")
                mask, sdf, gradient, found = feature_planes(image, kind, self.subject, depth)
                warnings.extend(found)
                for width, height in LEVELS:
                    maps[(width, "mask")][index] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    maps[(width, "sdf")][index] = cv2.resize(sdf, (width, height), interpolation=cv2.INTER_AREA).astype(np.float16)
                    maps[(width, "gradient")][index] = cv2.resize(gradient, (width, height), interpolation=cv2.INTER_AREA).astype(np.float16)
                desc[index] = descriptor(maps[(128, "mask")][index], maps[(128, "gradient")][index].astype(np.float32))
                if index % 100 == 0 or index + 1 == len(layers):
                    print(f"cache {kind}: {index + 1}/{len(layers)}", file=sys.stderr, flush=True)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
        json_write(self.root / "metadata.json", {"version": 2, "fingerprint": fingerprint, "levels": LEVELS, "peak_working_set_gib": peak,
                   "warnings": list(dict.fromkeys(warnings)), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def planes(self, kind, ordinal, width):
        return tuple(np.asarray(self.arrays[(kind, width, plane)][ordinal], dtype=np.float32 if plane != "mask" else np.uint8)
                     for plane in ("mask", "sdf", "gradient"))


def _normalize_descriptors(color, density):
    combined = np.vstack([color, density]).astype(float)
    median = np.median(combined, axis=0)
    scale = np.median(np.abs(combined - median), axis=0) * 1.4826
    return (color - median) / np.maximum(scale, .05), (density - median) / np.maximum(scale, .05)


def solve_monotonic_z(color_frames, ct_frames, color_desc, ct_desc, anchors, sample_count=160, band_frames=48):
    """Monotonic dynamic program around physical depth; anchors are exact."""
    color_frames, ct_frames = np.asarray(color_frames, float), np.asarray(ct_frames, float)
    if any(anchors[i][0] >= anchors[i + 1][0] or anchors[i][1] >= anchors[i + 1][1] for i in range(len(anchors) - 1)):
        raise ValueError("Z anchors must be strictly increasing")
    selected = sorted(set(np.linspace(0, len(color_frames) - 1, min(sample_count, len(color_frames))).round().astype(int).tolist()) |
                      {int(np.argmin(abs(color_frames - rgb))) for rgb, _ in anchors})
    cdesc, tdesc = _normalize_descriptors(np.asarray(color_desc), np.asarray(ct_desc))
    anchor_map = {int(np.argmin(abs(color_frames - rgb))): int(np.argmin(abs(ct_frames - ct))) for rgb, ct in anchors}
    states, parents, local_costs = [], [], []
    for row, ci in enumerate(selected):
        if ci in anchor_map:
            candidates = np.array([anchor_map[ci]], dtype=int)
        else:
            # Piecewise physical prior through hard anchors and both stack ends.
            control = {float(color_frames[0]): float(ct_frames[0]), float(color_frames[-1]): float(ct_frames[-1])}
            control.update({float(x): float(y) for x, y in anchors})
            ax = sorted(control)
            ay = [control[x] for x in ax]
            prior = float(np.interp(color_frames[ci], ax, ay))
            candidates = np.flatnonzero(abs(ct_frames - prior) <= band_frames)
            if not len(candidates): candidates = np.array([int(np.argmin(abs(ct_frames - prior)))])
            if len(candidates) > 81: candidates = candidates[np.linspace(0, len(candidates) - 1, 81).round().astype(int)]
        costs = np.linalg.norm(tdesc[candidates] - cdesc[ci], axis=1)
        if ci not in anchor_map:
            costs += .035 * abs(ct_frames[candidates] - prior)
        if row:
            color_change = np.linalg.norm(cdesc[ci] - cdesc[selected[row - 1]])
            ct_change = np.linalg.norm(tdesc[candidates] - tdesc[np.clip(candidates - 1, 0, len(tdesc) - 1)], axis=1)
            costs += .2 * abs(ct_change - color_change)
        if not states:
            total, parent = costs.copy(), np.full(len(candidates), -1, int)
        else:
            previous_candidates, previous_total = states[-1]
            total, parent = np.full(len(candidates), np.inf), np.full(len(candidates), -1, int)
            expected_step = (color_frames[ci] - color_frames[selected[row - 1]]) * (len(ct_frames) / max(1, len(color_frames)))
            for k, cj in enumerate(candidates):
                valid = np.flatnonzero(previous_candidates < cj)
                if not len(valid): continue
                transition = previous_total[valid] + .012 * abs((ct_frames[cj] - ct_frames[previous_candidates[valid]]) - expected_step)
                winner = int(valid[np.argmin(transition)])
                total[k], parent[k] = costs[k] + previous_total[winner] + .012 * abs((ct_frames[cj] - ct_frames[previous_candidates[winner]]) - expected_step), winner
        if not np.any(np.isfinite(total)): raise ValueError("No monotonic Z path satisfies the anchors and search band")
        states.append((candidates, total)); parents.append(parent); local_costs.append(costs)
    winner = int(np.argmin(states[-1][1])); path = []
    for row in range(len(states) - 1, -1, -1):
        candidates, _ = states[row]; path.append((selected[row], int(candidates[winner])))
        winner = int(parents[row][winner]) if row else -1
    path.reverse()
    knots, suggestions = [], []
    for row, (ci, tj) in enumerate(path):
        order = np.argsort(local_costs[row])
        chosen_index = int(np.flatnonzero(states[row][0] == tj)[0])
        other = [int(index) for index in order if int(index) != chosen_index]
        margin = float(local_costs[row][other[0]] - local_costs[row][chosen_index]) if other else 99.
        confidence = float(1 - math.exp(-max(0., margin)))
        knots.append({"color_depth": float(color_frames[ci] / max(1, color_frames[-1])), "color_frame": int(color_frames[ci]),
                      "ct_frame": float(ct_frames[tj]), "confidence": confidence, "hard_anchor": ci in anchor_map})
        ranked_indices = [chosen_index, *other[:3]]
        ranked = [{"ct_frame": float(ct_frames[int(states[row][0][idx])]), "score": float(local_costs[row][idx]), "recommended": idx == chosen_index} for idx in ranked_indices]
        suggestions.append({"color_frame": int(color_frames[ci]), "color_depth": knots[-1]["color_depth"], "recommended_ct_frame": float(ct_frames[tj]),
                            "score_margin": margin, "confidence": confidence, "alternatives": ranked})
    for rgb, ct in anchors:
        match = min(knots, key=lambda item: abs(item["color_frame"] - rgb))
        match.update({"color_frame": int(rgb), "color_depth": float(rgb / max(1, color_frames[-1])), "ct_frame": float(ct), "confidence": 1., "hard_anchor": True})
    knots.sort(key=lambda item: item["color_depth"])
    if any(knots[i]["ct_frame"] >= knots[i + 1]["ct_frame"] for i in range(len(knots) - 1)):
        raise ValueError("Z solver produced a non-monotonic result")
    return knots, suggestions


def _sampling_matrix(parameters: SpatialParameters, width, height):
    return np.linalg.inv(display_forward(parameters, width, height))[:2].astype(np.float32)


def objective(parameters, fixed, moving, previous=None):
    """35% silhouette, 35% internal gradient, 20% coverage, 10% continuity."""
    x, y, rotation, log_sx, log_sy = map(float, parameters)
    value = SpatialParameters(x, y, rotation, math.exp(log_sx), math.exp(log_sy))
    height, width = fixed[0].shape
    warp = _sampling_matrix(value, width, height)
    fixed_mask, fixed_sdf, fixed_gradient = fixed
    moving_mask, moving_sdf, moving_gradient = moving
    warped_mask = cv2.warpAffine(moving_mask, warp, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    warped_sdf = cv2.warpAffine(moving_sdf, warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    warped_gradient = cv2.warpAffine(moving_gradient, warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    fixed_edge, moving_edge = cv2.Canny(fixed_mask * 255, 20, 80), cv2.Canny(warped_mask * 255, 20, 80)
    silhouette = min(1., registration._symmetric_edge_distance(fixed_edge, moving_edge) / max(1., .04 * math.hypot(width, height)))
    overlap = (fixed_mask > 0) & (warped_mask > 0)
    gradient = float(np.mean(abs(fixed_gradient[overlap] - warped_gradient[overlap]))) if np.any(overlap) else 1.
    intersection, union = np.count_nonzero(overlap), np.count_nonzero((fixed_mask > 0) | (warped_mask > 0))
    iou = intersection / max(1, union)
    body_recall = intersection / max(1, np.count_nonzero(fixed_mask))
    # Penalize both missing source coverage and anatomy shrinkage.
    scale_penalty = max(0., 1 - value.x_scale) + max(0., 1 - value.y_scale)
    coverage = min(1., (1 - iou) * .55 + (1 - body_recall) * .35 + .10 * scale_penalty)
    continuity = 0.
    if previous is not None:
        continuity = min(1., math.sqrt(((x - previous.x_position) / 20) ** 2 + ((y - previous.y_position) / 20) ** 2 +
                                       ((rotation - previous.rotation_deg) / 8) ** 2 + (log_sx - math.log(previous.x_scale)) ** 2 +
                                       (log_sy - math.log(previous.y_scale)) ** 2))
    return .35 * silhouette + .35 * gradient + .20 * coverage + .10 * continuity


def optimize_pair(fixed, moving, initial: SpatialParameters, previous=None):
    base = np.array([initial.x_position, initial.y_position, initial.rotation_deg, math.log(initial.x_scale), math.log(initial.y_scale)])
    global_bounds = [(-160, 160), (-100, 100), (-12, 12), (math.log(.35), math.log(3.5)), (math.log(.25), math.log(3.5))]
    local_radius = [40, 32, 6, math.log(1.6), math.log(1.6)]
    bounds = [(max(low, center - radius), min(high, center + radius))
              for center, radius, (low, high) in zip(base, local_radius, global_bounds)]
    starts = [base, base + [0, 0, 0, math.log(1.1), math.log(1.1)], base + [0, -6, 0, 0, math.log(1.15)]]
    baseline = objective(base, fixed, moving, previous)
    best = (baseline, base)
    for start in starts:
        start = np.array([np.clip(value, *bounds[i]) for i, value in enumerate(start)])
        result = minimize(objective, start, args=(fixed, moving, previous), method="Powell", bounds=bounds,
                          options={"maxiter": 55, "xtol": .12, "ftol": 2e-4})
        if result.success and np.all(np.isfinite(result.x)) and result.fun < best[0]: best = (float(result.fun), result.x.copy())
    score, values = best
    # Rejection is exact: a failed/worse scratch result never replaces input.
    accepted = score < baseline - 1e-4
    if not accepted: values, score = base, baseline
    output = SpatialParameters(float(values[0]), float(values[1]), float(values[2]), math.exp(float(values[3])), math.exp(float(values[4])))
    return output, {"before": float(baseline), "after": float(score), "improvement": float(baseline - score), "accepted": accepted}


def smooth_spatial(solutions, max_frame, breaks):
    """Confidence-weighted splines, independently fit within acquisition segments."""
    output = []
    break_depths = sorted({0., 1., *[max(0., min(1., value / max_frame)) for value in breaks]})
    for lo, hi in zip(break_depths, break_depths[1:]):
        segment = [item for item in solutions if lo <= item["color_depth"] <= hi]
        if not segment: continue
        x = np.asarray([item["color_depth"] for item in segment])
        confidence = np.asarray([max(.05, item["confidence"]) for item in segment])
        for item in segment:
            params = {}
            for field in ("x_position", "y_position", "rotation_deg", "x_scale", "y_scale"):
                y = np.asarray([math.log(row[field]) if "scale" in field else row[field] for row in segment])
                if item.get("reviewed_constraint"):
                    value = math.log(item[field]) if "scale" in field else float(item[field])
                elif len(segment) >= 4 and np.ptp(x) > 1e-8:
                    spline = UnivariateSpline(x, y, w=confidence, s=max(.0001, len(x) * .02), k=min(3, len(x) - 1))
                    value = float(spline(item["color_depth"]))
                else: value = float(item[field])
                params[field] = math.exp(value) if "scale" in field else value
            spatial = SpatialParameters(**params)
            output.append({"color_depth": item["color_depth"], "color_frame": item["color_frame"], **asdict(spatial),
                           "inverse_uv": parameters_to_inverse_uv(spatial), "confidence": item["confidence"], "segment": f"{lo:.6f}-{hi:.6f}"})
    unique = {round(item["color_depth"], 10): item for item in output}
    return [unique[key] for key in sorted(unique)]


def select_review_queue(suggestions, manifest, report, count=10, preferred_frames=(), excluded_frames=()):
    priority = {}
    for item in suggestions:
        priority[item["color_frame"]] = (1 - item["confidence"], ["ambiguous_z"])
    for boundary in manifest.get("boundaries", []):
        if boundary.get("modality") == "rgb":
            frame = int(boundary.get("right", boundary.get("boundary", 0)))
            nearest = min(suggestions, key=lambda item: abs(item["color_frame"] - frame))
            score, reasons = priority[nearest["color_frame"]]
            priority[nearest["color_frame"]] = (score + 2, reasons + ["acquisition_or_fov_boundary"])
    for issue in (report or {}).get("problems", []):
        frame = int(issue.get("color_frame", issue.get("frame", 0)))
        nearest = min(suggestions, key=lambda item: abs(item["color_frame"] - frame))
        score, reasons = priority[nearest["color_frame"]]
        priority[nearest["color_frame"]] = (score + (2 if issue.get("severity") == "high" else .5), reasons + ["problem_region"])
    target = max(8, min(12, max(count, len(preferred_frames)))); excluded = set(excluded_frames)
    available = [item for item in suggestions if item["color_frame"] not in excluded]
    by_frame = {item["color_frame"]: item for item in available}
    chosen = [by_frame[frame] for frame in preferred_frames if frame in by_frame]
    chosen_frames = {item["color_frame"] for item in chosen}
    chosen.extend(item for item in sorted(available, key=lambda item: priority[item["color_frame"]][0], reverse=True) if item["color_frame"] not in chosen_frames)
    chosen = chosen[:target]
    return [{**item, "reasons": list(dict.fromkeys(priority[item["color_frame"]][1])), "status": "pending"} for item in chosen]


def acquisition_break_frames(manifest, depth_knots, max_frame):
    result = []
    z_color = np.asarray([item["color_frame"] for item in depth_knots], float)
    z_ct = np.asarray([item["ct_frame"] for item in depth_knots], float)
    for item in manifest.get("boundaries", []):
        reason = item.get("reason", {})
        if not (item.get("gap") or any(reason.get(key) for key in ("fov_change", "acquisition_change", "metadata_gap", "position_jump"))):
            continue
        location = item.get("right", item.get("boundary"))
        if not isinstance(location, (int, float)): continue
        if item.get("modality") == "density": location = float(np.interp(location, z_ct, z_color))
        result.append(max(0., min(float(max_frame), float(location))))
    return sorted(set(result))


def validation_frame_indices(rgb_frames, max_frame, lo, hi, reviewed_frames=()):
    """Pick three distributed frames farthest from manual review anchors."""
    depths = rgb_frames / max(1, max_frame)
    reviewed = np.asarray(sorted(set(reviewed_frames)), dtype=float)
    result = []
    edges = np.linspace(lo, min(1., hi), 4)
    for index in range(3):
        left, right = edges[index], edges[index + 1]
        candidates = np.flatnonzero((depths >= left) & (depths <= right if index == 2 else depths < right))
        if not len(candidates): continue
        center = (left + right) / 2

        def priority(ci):
            frame = rgb_frames[ci]
            distance = float(np.min(abs(reviewed - frame))) if reviewed.size else max_frame
            return distance, -abs(depths[ci] - center)

        result.append(int(max(candidates, key=priority)))
    return result


def regional_validation_metrics(old, new):
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    median_improvement = float(np.median(old - new))
    baseline_95th, candidate_95th = float(np.percentile(old, 95)), float(np.percentile(new, 95))
    return {"baseline_median": float(np.median(old)), "candidate_median": float(np.median(new)),
            "median_improvement": median_improvement, "baseline_95th": baseline_95th,
            "candidate_95th": candidate_95th,
            "accepted": bool(median_improvement >= 0 and candidate_95th <= 1.05 * baseline_95th)}


def validation_holdouts(cache, manifest, baseline, depth_knots, spatial_knots, reviewed_frames=(), level=512):
    """Three held-out levels per region at final analysis resolution."""
    rgb_layers, ct_layers = manifest["rgb"]["layers"], manifest["density"]["layers"]
    rgb_frames = np.asarray([item["frame"] for item in rgb_layers], float)
    ct_frames = np.asarray([item["frame"] for item in ct_layers], float)
    dx, dz = [item["color_depth"] for item in depth_knots], [item["ct_frame"] for item in depth_knots]
    bx = [item["color_depth"] for item in baseline["spatial_knots"]]
    sx = [item["color_depth"] for item in spatial_knots]
    rows = []
    for region, lo, hi in REGIONS:
        region_rows = []
        for ci in validation_frame_indices(rgb_frames, rgb_frames[-1], lo, hi, reviewed_frames):
            depth = rgb_frames[ci] / max(1, rgb_frames[-1])
            mapped = float(np.interp(depth, dx, dz)); tj = int(np.argmin(abs(ct_frames - mapped)))
            fixed, moving = cache.planes("rgb", ci, level), cache.planes("density", tj, level)
            position_scale = 512 / level
            baseline_uv = [float(np.interp(depth, bx, [item["inverse_uv"][column] for item in baseline["spatial_knots"]])) for column in range(6)]
            try: before = inverse_uv_to_parameters(baseline_uv, baseline.get("orientation", "flip_y"), 512, 304)
            except ValueError: before = SpatialParameters()
            candidate = SpatialParameters(
                float(np.interp(depth, sx, [item["x_position"] for item in spatial_knots])) / position_scale,
                float(np.interp(depth, sx, [item["y_position"] for item in spatial_knots])) / position_scale,
                float(np.interp(depth, sx, [item["rotation_deg"] for item in spatial_knots])),
                float(np.interp(depth, sx, [item["x_scale"] for item in spatial_knots])),
                float(np.interp(depth, sx, [item["y_scale"] for item in spatial_knots])))
            before.x_position /= position_scale; before.y_position /= position_scale
            region_rows.append({"color_frame": int(rgb_frames[ci]), "ct_frame": int(ct_frames[tj]),
                                "baseline": objective([before.x_position, before.y_position, before.rotation_deg, math.log(before.x_scale), math.log(before.y_scale)], fixed, moving),
                                "candidate": objective([candidate.x_position, candidate.y_position, candidate.rotation_deg, math.log(candidate.x_scale), math.log(candidate.y_scale)], fixed, moving)})
        old = [row["baseline"] for row in region_rows]; new = [row["candidate"] for row in region_rows]
        rows.append({"region": region, "samples": region_rows, **regional_validation_metrics(old, new)})
    return rows


def add_residual_knots(cache, manifest, depth_knots, spatial_knots, solution_rows, regional, pass_index, excluded_frames=()):
    """Add local knots beside failed held-outs without fitting the hold-out itself."""
    rgb_layers, ct_layers = manifest["rgb"]["layers"], manifest["density"]["layers"]
    rgb_frames = np.asarray([item["frame"] for item in rgb_layers], float)
    ct_frames = np.asarray([item["frame"] for item in ct_layers], float)
    dx, dz = [item["color_depth"] for item in depth_knots], [item["ct_frame"] for item in depth_knots]
    sx = [item["color_depth"] for item in spatial_knots]
    occupied = {item["color_frame"] for item in solution_rows}; excluded_frames = set(excluded_frames)
    added = 0
    for region in regional:
        if region["accepted"]: continue
        regressions = sorted(region["samples"], key=lambda item: item["candidate"] - item["baseline"], reverse=True)
        for worst in regressions:
            if worst["candidate"] <= worst["baseline"]: continue
            # Once a hold-out exposes a local failure it becomes a training
            # knot; the next validation pass rotates to different unseen data.
            target_frame = int(np.clip(worst["color_frame"], rgb_frames[0], rgb_frames[-1]))
            ci = int(np.argmin(abs(rgb_frames - target_frame))); color_frame = int(rgb_frames[ci])
            if color_frame in occupied or color_frame in excluded_frames: continue
            depth = color_frame / max(1, rgb_frames[-1]); mapped = float(np.interp(depth, dx, dz)); tj = int(np.argmin(abs(ct_frames - mapped)))
            initial = SpatialParameters(float(np.interp(depth, sx, [item["x_position"] for item in spatial_knots])),
                                        float(np.interp(depth, sx, [item["y_position"] for item in spatial_knots])),
                                        float(np.interp(depth, sx, [item["rotation_deg"] for item in spatial_knots])),
                                        float(np.interp(depth, sx, [item["x_scale"] for item in spatial_knots])),
                                        float(np.interp(depth, sx, [item["y_scale"] for item in spatial_knots])))
            fixed, moving = cache.planes("rgb", ci, 512), cache.planes("density", tj, 512)
            fitted, metrics = optimize_pair(fixed, moving, initial)
            solution_rows.append({"color_frame": color_frame, "color_depth": depth, "ct_frame": float(ct_frames[tj]), **asdict(fitted),
                                  "confidence": max(.05, min(1., metrics["improvement"] * 8 + .2)), "objective": metrics,
                                  "fov_limited": bool(np.any(moving[0][0]) or np.any(moving[0][-1]) or np.any(moving[0][:, 0]) or np.any(moving[0][:, -1])),
                                  "added_for_residual": region["region"]})
            occupied.add(color_frame); added += 1
    solution_rows.sort(key=lambda item: item["color_frame"])
    return added


def reviewed_landmark_rms(reviewed):
    values = []
    for anchor in reviewed.get("anchors", []):
        matrix = anchor.get("inverse_uv")
        pairs = anchor.get("landmarks", [])
        if not isinstance(matrix, list) or len(matrix) != 6: continue
        for pair in pairs:
            color, density = pair.get("color"), pair.get("density")
            if not (isinstance(color, list) and isinstance(density, list)): continue
            found = [matrix[0] * color[0] + matrix[1] * color[1] + matrix[2], matrix[3] * color[0] + matrix[4] * color[1] + matrix[5]]
            values.append(math.hypot((found[0] - density[0]) * 512, (found[1] - density[1]) * 304))
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def extend_review_queue_for_failed_regions(queue, regional, depth_knots, rgb_frames, ct_frames, limit=12):
    """Use the remaining adaptive-review budget for the most useful failures."""
    if not queue or any(item.get("status") != "confirmed" for item in queue): return queue
    existing = {item["color_frame"] for item in queue}
    failures = [item for item in regional if not item["accepted"]]
    failures.sort(key=lambda item: item["candidate_95th"] / max(1e-9, item["baseline_95th"]), reverse=True)
    for region in failures:
        if len(queue) >= limit: break
        sample = max(region["samples"], key=lambda item: item["candidate"] - item["baseline"])
        frame = int(sample["color_frame"])
        if frame in existing: continue
        mapped = float(np.interp(frame / max(1, rgb_frames[-1]),
                                 [item["color_depth"] for item in depth_knots], [item["ct_frame"] for item in depth_knots]))
        ordinal = int(np.argmin(abs(np.asarray(ct_frames) - mapped)))
        alternatives = [{"ct_frame": float(ct_frames[index]), "recommended": index == ordinal}
                        for index in range(max(0, ordinal - 2), min(len(ct_frames), ordinal + 3))]
        queue.append({"color_frame": frame, "color_depth": frame / max(1, rgb_frames[-1]),
                      "recommended_ct_frame": float(ct_frames[ordinal]), "score_margin": 0., "confidence": 0.,
                      "alternatives": alternatives, "reasons": [f"{region['region']}_holdout"], "status": "pending"})
        existing.add(frame)
    return queue


def select_control_review_frames(spatial_knots, z_queue, regional, confirmed_frames,
                                 max_frame, limit=10, previous_frames=None):
    """Choose one fixed, diverse manual-review cohort for a candidate.

    A previous cohort is deliberately retained across optimizer rebuilds. This
    prevents each saved point from exposing another pending spline knot and
    turning a ten-point review into a review of the entire curve.
    """
    available = {int(item["color_frame"]): item for item in spatial_knots}
    available.update({int(item["color_frame"]): item for item in z_queue})
    if previous_frames is not None:
        return sorted({int(frame) for frame in previous_frames if int(frame) in available})[:limit]

    confirmed = {int(frame) for frame in confirmed_frames}
    pending = set(available) - confirmed
    selected = []

    def add(frame):
        frame = int(frame)
        if frame in pending and frame not in selected and len(selected) < limit:
            selected.append(frame)

    # Z uncertainty has to be resolved before an in-plane transform is useful.
    for item in sorted(z_queue, key=lambda value: (value.get("status") != "pending",
                                                    value.get("confidence", 0),
                                                    value["color_frame"])):
        if item.get("status") == "pending":
            add(item["color_frame"])

    # Cover held-out locations that caused regional validation to fail. A
    # nearby confirmed point already supplies stronger evidence.
    proximity = max(12, round(max_frame * .015))
    for region in (item for item in regional if not item.get("accepted", False)):
        samples = sorted(region.get("samples", []),
                         key=lambda item: item.get("candidate", 0) - item.get("baseline", 0),
                         reverse=True)
        for sample in samples:
            target = int(sample["color_frame"])
            if confirmed and min(abs(target - frame) for frame in confirmed) <= proximity:
                continue
            choices = pending - set(selected)
            if choices:
                add(min(choices, key=lambda frame: (abs(frame - target), frame)))

    # Spend remaining reviews in the largest unsupported gaps. Confidence is
    # only a tie-breaker so low-confidence clusters do not consume the cohort.
    while len(selected) < limit and pending - set(selected):
        support = confirmed | set(selected)

        def priority(frame):
            distance = min((abs(frame - anchor) for anchor in support), default=max_frame)
            confidence = float(available[frame].get("confidence", 0) or 0)
            return distance, 1 - confidence, -frame

        add(max(pending - set(selected), key=priority))
    return sorted(selected)


def run(directory: Path, subject: str, candidate: Path, cache_dir=None, memory_limit_gib=4., rebuild_cache=False, review_count=10):
    if subject != "male":
        raise ValueError("male establishes this method; rebuild female geometry before using --subject female")
    started = time.time()
    manifest_path, alignment_path, corrections_path = directory / "manifest.json", directory / "alignment.json", directory / "corrections.json"
    manifest, baseline = json.loads(manifest_path.read_text()), json.loads(alignment_path.read_text())
    if not manifest.get("baked"): raise ValueError("a completed baked dataset is required")
    # The cache belongs beside the candidate by default, but may be redirected to a large disk.
    if cache_dir is not None:
        proxy = directory / ".alignment-cache-v2"
        target = Path(cache_dir).resolve() / subject
        if proxy.exists() and not proxy.is_symlink(): raise ValueError("remove the existing local cache before redirecting it")
        target.mkdir(parents=True, exist_ok=True)
        if not proxy.exists(): proxy.symlink_to(target, target_is_directory=True)
    cache = FeatureCache(directory, manifest, subject, rebuild_cache)
    rgb_layers, ct_layers = manifest["rgb"]["layers"], manifest["density"]["layers"]
    rgb_frames = [item["frame"] for item in rgb_layers]; ct_frames = [item["frame"] for item in ct_layers]
    review_file = directory / "reviewed-alignment.json"
    reviewed = json.loads(review_file.read_text()) if review_file.exists() else {"anchors": []}
    anchors = [(0, 7.)]
    for item in reviewed.get("anchors", []):
        if item.get("z_confirmed", True) is not True:
            continue
        try: anchors.append((int(item["color_frame"]), float(item["ct_frame"])))
        except (KeyError, TypeError, ValueError): pass
    anchors = sorted(set(anchors))
    depth_knots, suggestions = solve_monotonic_z(rgb_frames, ct_frames, cache.descriptors["rgb"], cache.descriptors["density"], anchors)
    report_file = directory / "alignment-problems.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else None
    previous_candidate = json.loads(candidate.read_text()) if candidate.exists() else {}
    same_previous_inputs = previous_candidate.get("inputs", {}).get("manifest_sha256") == sha256(manifest_path)
    deleted_knot_frames = set(previous_candidate.get("deleted_knot_frames", [])) if same_previous_inputs else set()
    previous_queue = previous_candidate.get("z_review_queue", []) if same_previous_inputs else []
    previous_control_frames = previous_candidate.get("control_review_frames") if same_previous_inputs else None
    rejected_frames = [item["color_frame"] for item in previous_queue if item.get("status") == "rejected"] + list(deleted_knot_frames)
    preferred_frames = [item["color_frame"] for item in previous_queue if item.get("status") != "rejected"]
    queue = select_review_queue(suggestions, manifest, report, review_count, preferred_frames, rejected_frames)
    confirmed_frames = {int(item["color_frame"]): float(item["ct_frame"]) for item in reviewed.get("anchors", []) if item.get("z_confirmed") is True}
    for item in queue:
        if item["color_frame"] in confirmed_frames:
            item.update({"status": "confirmed", "confirmed_ct_frame": confirmed_frames[item["color_frame"]]})
    reviewed_by_frame = {int(item["color_frame"]): item for item in reviewed.get("anchors", [])
                         if isinstance(item.get("parameters"), dict) and
                         (item.get("reviewed_fields") is None or any(field in item.get("reviewed_fields", [])
                          for field in ("x_position", "y_position", "rotation_deg", "x_scale", "y_scale")))}
    spatial_frames = sorted(set(np.linspace(0, len(rgb_layers) - 1, 28).round().astype(int).tolist()) |
                            {int(np.argmin(abs(np.asarray(rgb_frames) - item["color_frame"]))) for item in queue} |
                            {int(np.argmin(abs(np.asarray(rgb_frames) - frame))) for frame in confirmed_frames} |
                            {int(np.argmin(abs(np.asarray(rgb_frames) - frame))) for frame in reviewed_by_frame})
    spatial_frames = [index for index in spatial_frames if rgb_frames[index] not in deleted_knot_frames]
    baseline_spatial_x = [item["color_depth"] for item in baseline["spatial_knots"]]
    solution_rows, previous_small = [], None
    for order, ci in enumerate(spatial_frames):
        color_frame = rgb_frames[ci]
        depth = color_frame / max(1, rgb_frames[-1])
        mapped = float(np.interp(depth, [item["color_depth"] for item in depth_knots], [item["ct_frame"] for item in depth_knots]))
        tj = int(np.argmin(abs(np.asarray(ct_frames) - mapped)))
        baseline_uv = [float(np.interp(depth, baseline_spatial_x, [item["inverse_uv"][column] for item in baseline["spatial_knots"]])) for column in range(6)]
        try: initial = inverse_uv_to_parameters(baseline_uv, baseline.get("orientation", "flip_y"))
        except ValueError: initial = SpatialParameters()
        reviewed_anchor = reviewed_by_frame.get(color_frame)
        manual_initial = None if reviewed_anchor else interpolate_reviewed_parameters(reviewed_by_frame, color_frame)
        if manual_initial is not None: initial = manual_initial
        fixed = cache.planes("rgb", ci, 256); moving = cache.planes("density", tj, 256)
        if reviewed_anchor:
            parameters = reviewed_anchor["parameters"]
            fitted = SpatialParameters(float(parameters["x_position"]), float(parameters["y_position"]), float(parameters["rotation_deg"]), float(parameters["x_scale"]), float(parameters["y_scale"]))
            fitted_small = SpatialParameters(fitted.x_position / 2, fitted.y_position / 2, fitted.rotation_deg, fitted.x_scale, fitted.y_scale)
            score = objective([fitted_small.x_position, fitted_small.y_position, fitted_small.rotation_deg, math.log(fitted_small.x_scale), math.log(fitted_small.y_scale)], fixed, moving, previous_small)
            metrics = {"before": score, "after": score, "improvement": 0., "accepted": True, "reviewed_constraint": True}
        else:
            # Cache density is already in the fixed flip_y orientation.
            scaled_initial = SpatialParameters(initial.x_position / 2, initial.y_position / 2, initial.rotation_deg, initial.x_scale, initial.y_scale)
            fitted_small, metrics = optimize_pair(fixed, moving, scaled_initial, previous_small)
            fitted = SpatialParameters(fitted_small.x_position * 2, fitted_small.y_position * 2,
                                       fitted_small.rotation_deg, fitted_small.x_scale, fitted_small.y_scale)
        previous_small = fitted_small
        confidence = 1. if reviewed_anchor else max(.05, min(1., metrics["improvement"] * 8 + .2))
        row = {"color_frame": color_frame, "color_depth": depth, "ct_frame": float(ct_frames[tj]), **asdict(fitted),
               "confidence": confidence, "objective": metrics, "reviewed_constraint": bool(reviewed_anchor),
               "fov_limited": bool(np.any(moving[0][0]) or np.any(moving[0][-1]) or np.any(moving[0][:, 0]) or np.any(moving[0][:, -1]))}
        solution_rows.append(row)
        print(f"spatial {order + 1}/{len(spatial_frames)} RGB {color_frame} CT {ct_frames[tj]} objective {metrics['before']:.4f}->{metrics['after']:.4f}", file=sys.stderr, flush=True)
    breaks = acquisition_break_frames(manifest, depth_knots, max(rgb_frames))
    spatial_knots = smooth_spatial(solution_rows, max(rgb_frames), breaks)
    validation_controls = {item["color_frame"] for item in solution_rows}
    heldout = validation_holdouts(cache, manifest, baseline, depth_knots, spatial_knots, validation_controls)
    refinement_passes = 1
    for pass_index in range(2):
        added = add_residual_knots(cache, manifest, depth_knots, spatial_knots, solution_rows, heldout, pass_index, deleted_knot_frames)
        if not added: break
        spatial_knots = smooth_spatial(solution_rows, max(rgb_frames), breaks)
        validation_controls = {item["color_frame"] for item in solution_rows}
        heldout = validation_holdouts(cache, manifest, baseline, depth_knots, spatial_knots, validation_controls)
        refinement_passes += 1
    for item in spatial_knots:
        item["z_position"] = float(np.interp(item["color_depth"], [knot["color_depth"] for knot in depth_knots], [knot["ct_frame"] for knot in depth_knots]))
    compiled_depth_knots = [{"color_depth": item["color_depth"], "ct_frame": item["z_position"],
                             "confidence": item["confidence"], "segment": item["segment"]} for item in spatial_knots]
    landmark_rms = reviewed_landmark_rms(reviewed)
    queue = extend_review_queue_for_failed_regions(queue, heldout, depth_knots, rgb_frames, ct_frames)
    control_review_frames = select_control_review_frames(
        spatial_knots, queue, heldout, confirmed_frames, max(rgb_frames),
        limit=review_count, previous_frames=previous_control_frames)
    # Acceptance remains false until the adaptive Z queue and regional holdouts are reviewed.
    confirmed = sum(item["status"] == "confirmed" for item in queue)
    peak_gib = max(cache.cache_build_peak_gib, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2))
    blockers = []
    if confirmed < len(queue): blockers.append("adaptive_z_review_incomplete")
    if any(not item["accepted"] for item in heldout): blockers.append("regional_holdout_validation_failed")
    if landmark_rms is not None and landmark_rms >= 2: blockers.append("reviewed_landmark_rms_above_two_pixels")
    if peak_gib >= memory_limit_gib: blockers.append("memory_limit_exceeded")
    qc = {"accepted": not blockers, "promotion_blockers": blockers,
          "z_confirmed": confirmed, "z_required": len(queue), "hard_anchors_preserved": any(item["color_frame"] == 0 and item["ct_frame"] == 7 for item in depth_knots),
          "strictly_monotonic_z": all(depth_knots[i]["ct_frame"] < depth_knots[i + 1]["ct_frame"] for i in range(len(depth_knots) - 1)),
          "peak_working_set_gib": peak_gib, "memory_limit_gib": memory_limit_gib, "within_memory_limit": peak_gib < memory_limit_gib,
          "spatial_improved": sum(item["objective"]["accepted"] for item in solution_rows), "spatial_total": len(solution_rows),
          "regional_holdouts": heldout, "reviewed_landmark_rms_pixels": landmark_rms, "refinement_passes": refinement_passes}
    value = {"version": 2, "subject": subject, "status": "needs_review", "reference": "color", "absolute": True,
             "model": list(PARAMETER_NAMES), "orientation": "flip_y",
             "inputs": {"manifest_sha256": sha256(manifest_path), "corrections_sha256": sha256(corrections_path), "baseline_alignment_sha256": sha256(alignment_path)},
             "coverage": {"color_depth": [depth_knots[0]["color_depth"], depth_knots[-1]["color_depth"]], "ct_frame": [depth_knots[0]["ct_frame"], depth_knots[-1]["ct_frame"]]},
             "hard_z_anchors": [{"color_frame": rgb, "ct_frame": ct} for rgb, ct in anchors],
             "deleted_knot_frames": sorted(deleted_knot_frames),
             "z_match_knots": depth_knots, "depth_knots": compiled_depth_knots, "parameter_knots": spatial_knots,
             "spatial_knots": [{"color_depth": item["color_depth"], "inverse_uv": item["inverse_uv"], "confidence": item["confidence"], "segment": item["segment"]} for item in spatial_knots],
             "review_boundaries": [value / max(rgb_frames) for value in breaks], "max_interpolation_span": .06, "unsupported_ranges": [], "z_review_queue": queue,
             "control_review_limit": review_count, "control_review_frames": control_review_frames,
             "local_solutions": solution_rows, "qc": qc, "elapsed_seconds": time.time() - started,
             "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json_write(candidate, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=("male", "female"), default="male")
    parser.add_argument("--processed", type=Path, help="Processed subject directory")
    parser.add_argument("--candidate", type=Path, default=Path("alignment-candidate-v2.json"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--memory-limit-gib", type=float, default=4.)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--review-count", type=int, default=10)
    args = parser.parse_args(argv)
    default_root = Path(os.environ.get("VISIBLE_HUMAN_ROOT", Path.home() / "Visible-Human-Project")) / "Processed" / "v1"
    root = Path(os.environ.get("VISIBLE_HUMAN_PROCESSED_ROOT", default_root))
    directory = (args.processed or root / args.subject).resolve()
    candidate = args.candidate if args.candidate.is_absolute() else directory / args.candidate
    if not 0.5 <= args.memory_limit_gib <= 64: parser.error("--memory-limit-gib must be between .5 and 64")
    try:
        result = run(directory, args.subject, candidate.resolve(), args.cache, args.memory_limit_gib, args.rebuild_cache, args.review_count)
        print(json.dumps({"candidate": str(candidate.resolve()), "status": result["status"], "qc": result["qc"]}, allow_nan=False))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}, allow_nan=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
