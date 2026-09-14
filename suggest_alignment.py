#!/usr/bin/env python3
"""Suggest one reviewable outline fit from baked images, without writing data.

The returned transform is absolute destination-color UV -> source-density UV.
Metrics are segmentation/edge proxies, never anatomical correspondence claims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from preprocess import Slice, read_baked_ct, read_baked_rgb
import register_modalities as registration


def interpolate(knots, position, field):
    ordered = sorted(knots, key=lambda item: item["color_depth"])
    if not ordered:
        raise ValueError(f"Missing registration {field}")
    x = [item["color_depth"] for item in ordered]
    values = np.asarray([item[field] for item in ordered], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite registration {field}")
    if values.ndim == 1:
        return float(np.interp(position, x, values))
    return np.array([np.interp(position, x, values[:, i]) for i in range(values.shape[1])])


def choose_ct_layer(layers, requested):
    """Use an actual layer, refusing out-of-range requests and long gaps."""
    if not layers or not math.isfinite(requested):
        raise ValueError("A finite CT frame and available density layers are required")
    ordered = sorted(layers, key=lambda item: item["frame"])
    frames = np.asarray([item["frame"] for item in ordered], dtype=float)
    if requested < frames[0] or requested > frames[-1]:
        raise ValueError("Requested CT frame is outside available source layers")
    right = int(np.searchsorted(frames, requested))
    if right and right < len(frames) and frames[right] != requested:
        steps = np.diff(frames)
        expected_step = float(np.median(steps[steps > 0]))
        if frames[right] - frames[right - 1] > 2 * expected_step:
            raise ValueError("Requested CT frame falls inside a missing density gap")
    return ordered[int(np.argmin(np.abs(frames - requested)))]


def color_mask(image, subject):
    mask = registration.color_anatomy_mask(image)
    warnings = []
    if subject != "female":
        return mask, warnings
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        # Frame cards are detached, small yellow rectangles below the body.
        # Position alone is insufficient: retain low-positioned body parts.
        rectangle = (.65 * height < y and w < .18 * width and h < .18 * height
                     and .65 < w / max(1, h) < 1.8 and area / max(1, w * h) > .65)
        upper_parts = [
            int(stats[other, 4]) for other in range(1, count)
            if other != label and stats[other, 4] > 1.1 * area
            and stats[other, 1] + stats[other, 3] + .12 * height < y
        ]
        separated_body = sum(upper_parts) > 2 * area
        if not rectangle or not separated_body:
            continue
        pixels = image[labels == label].astype(float)
        r, g, b = pixels.T
        yellow = (r > 70) & (g > 70) & (b < np.minimum(r, g) * .82) & (np.abs(r - g) < .35 * np.maximum(r, g))
        if float(yellow.mean()) > .45:
            mask[labels == label] = 0
            warnings.append("Excluded a detached yellow annotation rectangle from the color fitting mask.")
    return mask, list(dict.fromkeys(warnings))


def density_mask(hu, subject, color_depth):
    if subject != "male" or not .52 <= color_depth <= .84:
        return registration.ct_anatomy_mask(hu), []
    # Closing first can join the radiodense scanner wedges to the legs.
    # Cortical bone is a regional seed only; never require it for head/chest.
    body = (hu > -400).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(body, 8)
    mask = np.zeros_like(body)
    excluded = joined = False
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < 150 or y > hu.shape[0] * .82:
            continue
        part = labels == label
        core_y, core_x = np.nonzero(part & (hu > 700))
        if len(core_x) < 20:
            excluded = True
            continue
        mask[part] = 1
        # A bone-seeded component can still contain an attached support.
        if core_y.min() - y > .42 * h or y + h - 1 - core_y.max() > .55 * h:
            joined = True
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    warnings = ["Lower-body fitting uses cortical-bone seeds; soft tissue without a strong bone core may be excluded."]
    if excluded:
        warnings.append("Excluded detached CT components without a strong cortical core; inspect their identity in the source density.")
    if joined:
        warnings.append("A retained CT component may still contain attached support; inspect the outline and use matching landmarks.")
    return mask, warnings


def valid_mask(mask):
    ys, xs = np.nonzero(mask)
    return len(xs) >= 150 and np.ptp(xs) >= 8 and np.ptp(ys) >= 8


def pixel_matrix(inverse_uv, width, height):
    uv = np.eye(3)
    uv[:2] = np.asarray(inverse_uv, dtype=float).reshape(2, 3)
    pixels = np.diag([width - 1., height - 1., 1.])
    return (pixels @ uv @ np.linalg.inv(pixels))[:2].astype(np.float32)


def metrics(color, moving, fixed_mask, moving_mask, matrix):
    size = (fixed_mask.shape[1], fixed_mask.shape[0])
    warped = cv2.warpAffine(moving_mask, matrix, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    fixed, matched = fixed_mask > 0, warped > 0
    intersection = int((fixed & matched).sum())
    fixed_edges = registration._internal_landmark_edges(color, "color", fixed_mask)
    moving_edges = registration._internal_landmark_edges(moving, "density", moving_mask)
    warped_edges = cv2.warpAffine(moving_edges, matrix, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    interior = (registration._symmetric_edge_distance(fixed_edges, warped_edges)
                if min(np.count_nonzero(fixed_edges), np.count_nonzero(warped_edges)) >= 20 else None)
    support = cv2.warpAffine(np.ones_like(moving_mask), matrix, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    return {
        "iou": intersection / max(1, int((fixed | matched).sum())),
        "body_recall": intersection / max(1, int(fixed.sum())),
        "interior_edge_disagreement_px": interior,
        "canvas_uncovered_fraction": float((fixed & (support == 0)).sum() / max(1, fixed.sum())),
    }


def score(value):
    interior = value["interior_edge_disagreement_px"]
    return .8 * value["iou"] + (.2 * math.exp(-interior / 10.) if interior is not None else 0.)


def suggest_pair(color, hu, subject, color_depth, baseline_uv, orientation="identity"):
    baseline_values = np.asarray(baseline_uv, dtype=float)
    if baseline_values.shape != (6,) or not np.all(np.isfinite(baseline_values)):
        raise ValueError("A finite six-value baseline transform is required")
    if color.shape[:2] != hu.shape[:2]:
        raise ValueError("Color and density analysis images must use the same canvas")
    fixed_mask, color_warnings = color_mask(color, subject)
    raw_mask, ct_warnings = density_mask(hu, subject, color_depth)
    if not valid_mask(fixed_mask) or not valid_mask(raw_mask):
        raise ValueError("Too little reliable body tissue for an outline fit; use matching landmarks")
    moving, moving_mask = registration.orient(hu, orientation), registration.orient(raw_mask, orientation)
    height, width = fixed_mask.shape
    orientation_matrix = registration._orientation_pixels(orientation, width, height)
    baseline = np.eye(3)
    baseline[:2] = pixel_matrix(baseline_uv, width, height)
    if abs(np.linalg.det(baseline[:2, :2])) < 1e-6:
        raise ValueError("The existing transform is degenerate; use matching landmarks")
    baseline_oriented = (np.linalg.inv(orientation_matrix) @ baseline)[:2].astype(np.float32)
    before = metrics(color, moving, fixed_mask, moving_mask, baseline_oriented)
    fixed_y, fixed_x = np.nonzero(fixed_mask)
    moving_y, moving_x = np.nonzero(moving_mask)
    bounds_x = (np.ptp(moving_x) + 1) / (np.ptp(fixed_x) + 1)
    bounds_y = (np.ptp(moving_y) + 1) / (np.ptp(fixed_y) + 1)
    if not (registration.SCALE_X_LIMITS[0] <= bounds_x <= registration.SCALE_X_LIMITS[1]
            and registration.SCALE_Y_LIMITS[0] <= bounds_y <= registration.SCALE_Y_LIMITS[1]):
        raise ValueError("Outline scale lies outside supported fitting limits; use matching landmarks")
    initial = registration._bounds_transform(fixed_mask, moving_mask)
    sx, sy = float(initial[0, 0]), float(initial[1, 1])
    if not (registration.SCALE_X_LIMITS[0] <= sx <= registration.SCALE_X_LIMITS[1]
            and registration.SCALE_Y_LIMITS[0] <= sy <= registration.SCALE_Y_LIMITS[1]):
        raise ValueError("Outline scale lies outside supported fitting limits")
    best = initial.copy()
    after = metrics(color, moving, fixed_mask, moving_mask, best)
    fixed_feature, _ = registration.feature(color, fixed_mask)
    moving_feature, _ = registration.feature(moving, moving_mask)
    try:
        _, scratch = cv2.findTransformECC(
            fixed_feature, moving_feature, initial.copy(), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 70, 1e-5),
            inputMask=moving_mask,
        )
        candidate, candidate_x, candidate_y, rotation = registration._project(scratch)
        plausible = (.65 * sx <= candidate_x <= 1.35 * sx and .65 * sy <= candidate_y <= 1.35 * sy
                     and registration.SCALE_X_LIMITS[0] <= candidate_x <= registration.SCALE_X_LIMITS[1]
                     and registration.SCALE_Y_LIMITS[0] <= candidate_y <= registration.SCALE_Y_LIMITS[1]
                     and abs(rotation) <= 10 and np.all(np.isfinite(candidate)))
        if plausible:
            measured = metrics(color, moving, fixed_mask, moving_mask, candidate)
            if measured["iou"] >= after["iou"] - .02 and score(measured) > score(after) + 1e-6:
                best, after = candidate.copy(), measured
    except cv2.error:
        pass
    warnings = ["Outline and interior-edge scores are proxies only; verify corresponding anatomy and the CT plane before saving.", *color_warnings, *ct_warnings]
    method = "component-aware-outline-candidate-v1"
    if score(before) >= score(after):
        best, after = baseline_oriented.copy(), before.copy()
        method = "existing-profile-retained"
        warnings.append("The outline candidate did not improve the existing transform's proxy score; the existing transform is returned.")
    if after["iou"] < .8:
        warnings.append("Outline agreement remains poor; check CT plane, segmentation, support and matching landmarks.")
    interior_before = before["interior_edge_disagreement_px"]
    interior_after = after["interior_edge_disagreement_px"]
    if interior_after is not None and interior_before is not None and interior_after > interior_before + .5:
        warnings.append("The outline candidate increases interior-edge disagreement; inspect matching bones and the CT plane before saving.")
    elif interior_after is not None and interior_after > 12:
        warnings.append("Interior-edge disagreement remains high even if the outlines agree; inspect matching bones and the CT plane.")
    if after["canvas_uncovered_fraction"] > .01:
        warnings.append("Some color tissue lies outside the transformed CT canvas; an outline fit cannot recover absent source pixels.")
    if np.any(raw_mask[0]) or np.any(raw_mask[-1]) or np.any(raw_mask[:, 0]) or np.any(raw_mask[:, -1]):
        warnings.append("The CT fitting mask touches the baked source canvas edge; inspect the original field of view for truncation.")
    solution = registration.AffineSolution(orientation, best.astype(float).tolist(), 1, 1, 0, 0, 0, after["iou"], 0, 0)
    return {
        "inverse_uv": registration.inverse_uv(solution, width, height),
        "metrics": {"before": before, "after": after}, "warnings": warnings, "method": method,
        "source": "baked corrected color and density; fitting masks only, no source images modified",
    }


def layer_slice(directory, entry, kind):
    path = (directory / entry["file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Layer path leaves the processed subject directory")
    return Slice(str(path), path.name, int(entry["frame"]), kind)


def suggest(directory, color_frame, ct_frame=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    profile = json.loads((directory / "alignment.json").read_text())
    if not manifest.get("baked"):
        raise ValueError("Outline suggestions require baked source layers")
    rgb_layers, ct_layers = manifest["rgb"]["layers"], manifest["density"]["layers"]
    color_layer = next((item for item in rgb_layers if item["frame"] == color_frame), None)
    if color_layer is None:
        raise ValueError("Requested color frame does not exist")
    depth = color_frame / max(1, max(item["frame"] for item in rgb_layers))
    coverage = profile.get("coverage", {}).get("color_depth", [0, 1])
    if ct_frame is None and not coverage[0] <= depth <= coverage[1]:
        raise ValueError("Choose a CT frame explicitly outside registered depth coverage")
    requested = float(ct_frame) if ct_frame is not None else interpolate(profile["depth_knots"], depth, "ct_frame")
    density_layer = choose_ct_layer(ct_layers, requested)
    color = read_baked_rgb(layer_slice(directory, color_layer, "rgb"))
    hu = read_baked_ct(layer_slice(directory, density_layer, "density"),
                       int(manifest["density"]["width"]), int(manifest["density"]["height"]))
    baseline = interpolate(profile["spatial_knots"], depth, "inverse_uv")
    result = suggest_pair(color, hu, manifest.get("subject", directory.name), depth, baseline, profile.get("orientation", "identity"))
    result.update({"ct_frame": density_layer["frame"], "color_frame": color_frame})
    if abs(requested - density_layer["frame"]) > 1e-9:
        result["warnings"].append(f"Requested CT {requested:g}; fitted actual available CT {density_layer['frame']} without density interpolation.")
    if not coverage[0] <= depth <= coverage[1]:
        result["warnings"].append("This color frame is outside registered depth coverage; the baseline transform uses its nearest spatial endpoint.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True, help="Processed subject directory")
    parser.add_argument("--color-frame", type=int, required=True)
    parser.add_argument("--ct-frame", type=float)
    args = parser.parse_args(argv)
    try:
        result = suggest(args.processed, args.color_frame, args.ct_frame)
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, KeyError, cv2.error) as error:
        print(json.dumps({"error": str(error)}, allow_nan=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
