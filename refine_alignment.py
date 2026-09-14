#!/usr/bin/env python3
"""Read-only local RGB/CT refinement; every result still needs human review.

Optimize the current six-parameter editor state, never a global profile guess.
Only one RGB image and its small CT neighborhood are loaded. RGB intensities
are not subtracted from HU: tissue-only mutual information supplies the actual
cross-modality intensity term alongside internal gradients and surface distance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from optimize_alignment import (SpatialParameters, inverse_uv_to_parameters,
                                parameters_to_inverse_uv, sha256)
from preprocess import read_baked_ct, read_baked_rgb
import register_modalities as registration
from suggest_alignment import color_mask, density_mask, layer_slice, pixel_matrix

METHOD = "local-six-parameter-rgb-hu-mutual-information-v1"
ANALYSIS_SIZE = (512, 304)
MAX_SECONDS = 45.


def _finite(value, name):
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError(f"{name} must be finite")
    return answer


def local_z_bounds(manifest, requested, color_frame, reviewed=(), supplied=None,
                   landmarks=()):
    """Stay inside one acquisition, available layers, and neighboring anchors."""
    frames = np.asarray(sorted({float(item["frame"]) for item in manifest["density"]["layers"]}))
    if not len(frames) or requested < frames[0] or requested > frames[-1]:
        raise ValueError("The current CT frame is outside the available density layers")
    low, high = max(float(frames[0]), requested - 8), min(float(frames[-1]), requested + 8)
    cuts = []
    steps = np.diff(frames)
    if len(steps):
        for index in np.flatnonzero(steps > 2 * float(np.median(steps))):
            cuts.append((frames[index], frames[index + 1]))
    for boundary in manifest.get("boundaries", []):
        reason = boundary.get("reason") or {}
        if boundary.get("modality") != "density" or not (boundary.get("gap") or any(
                reason.get(key) for key in ("fov_change", "acquisition_change", "metadata_gap", "position_jump"))):
            continue
        right = boundary.get("right", boundary.get("boundary"))
        if right is None:
            continue
        right = float(right)
        previous = frames[frames < right]
        left = float(boundary.get("left", previous[-1] if len(previous) else right))
        cuts.append((left, right))
    for left, right in cuts:
        if left < requested < right:
            raise ValueError("The current CT plane crosses an acquisition boundary or missing gap; choose an actual source plane first")
        if requested <= left:
            high = min(high, float(left))
        elif requested >= right:
            low = max(low, float(right))
    anchors = [(int(a["color_frame"]), _finite(a["ct_frame"], "anchor CT"))
               for a in reviewed if a.get("z_confirmed", True) is True]
    if manifest.get("subject") == "male":
        anchors.append((0, 7.))
    for frame, z in anchors:
        if frame < color_frame:
            low = max(low, z + 1e-4)
        elif frame > color_frame:
            high = min(high, z - 1e-4)
    if supplied is not None:
        if not isinstance(supplied, (list, tuple)) or len(supplied) != 2:
            raise ValueError("z_bounds must contain two finite ordered CT frames")
        lo, hi = [_finite(value, "z_bounds") for value in supplied]
        if lo > hi:
            raise ValueError("z_bounds are reversed")
        low, high = max(low, lo), min(high, hi)
    if manifest.get("subject") == "male" and color_frame == 0:
        if abs(requested - 7.) > 1e-8:
            raise ValueError("The hard RGB 0 to CT 7 anchor must be restored before refinement")
        low, high = max(low, 7.), min(high, 7.)
    if landmarks:
        low, high = max(low, requested), min(high, requested)
    if low > high or not low - 1e-8 <= requested <= high + 1e-8:
        raise ValueError("The current CT frame conflicts with neighboring confirmed Z anchors")
    return float(low), float(high)


def _feature(image, mask, kind):
    mask = mask.astype(np.uint8)
    interior = cv2.erode(mask, np.ones((9, 9), np.uint8))
    outline = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if kind == "rgb":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = registration._internal_landmark_edges(image, "color", mask)
    else:
        gray = np.clip((image + 250) * (255 / 1500), 0, 255).astype(np.uint8)
        edges = registration._internal_landmark_edges(image, "density", mask)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    scale = float(np.percentile(gradient[interior > 0], 95)) if np.any(interior) else 1.
    gradient = np.minimum(gradient / max(1., scale), 1.5) * interior
    outline_distance = cv2.distanceTransform((outline == 0).astype(np.uint8), cv2.DIST_L2, 3)
    edge_distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return {"mask": mask, "interior": interior, "outline": outline.astype(np.float32),
            "edges": (edges > 0).astype(np.float32), "gradient": gradient,
            "outline_distance": np.minimum(outline_distance, 30), "edge_distance": np.minimum(edge_distance, 30),
            "planes": np.stack([mask, np.minimum(outline_distance, 30), np.minimum(edge_distance, 30), gradient], axis=-1).astype(np.float32)}


def normalized_mutual_information(rgb_bins, hu, valid):
    """Mean normalized MI of RGB luminance/chroma versus HU, tissue only.

    Histogram smoothing limits quantization artifacts during subpixel motion.
    Fixed intensity bins keep changes in overlap from renormalizing the data.
    """
    if np.count_nonzero(valid) < 150:
        return 0., False
    density = hu[valid]
    if float(np.std(density)) < 10:
        return 0., False
    density_bin = np.minimum(23, np.maximum(0, ((density + 300) * (24 / 1900)).astype(np.int32)))
    values = []
    informative = 0
    for channel in rgb_bins:
        fixed = channel[valid]
        if float(np.std(fixed)) < .35:
            continue
        joint = np.bincount(fixed * 24 + density_bin, minlength=24 * 24).reshape(24, 24).astype(float)
        joint = gaussian_filter(joint, .6)
        joint /= max(1., joint.sum())
        px, py = joint.sum(1), joint.sum(0)
        entropy_x = -float(np.sum(px * np.log(px + 1e-12)))
        entropy_y = -float(np.sum(py * np.log(py + 1e-12)))
        independent = px[:, None] * py[None, :]
        mi = float(np.sum(joint * np.log((joint + 1e-12) / (independent + 1e-12))))
        values.append(max(0., mi / max(1e-8, math.sqrt(entropy_x * entropy_y))))
        informative += entropy_x > .5 and entropy_y > .5
    return float(np.mean(values)) if values else 0., bool(informative)


class PairObjective:
    def __init__(self, color, ct_frames, ct_images, baseline, orientation, subject, depth,
                 landmarks=(), width=256):
        self.width, self.height = width, round(width * 304 / 512)
        size = (self.width, self.height)
        self.ratio = (self.width - 1) / 511.
        self.frames = np.asarray(ct_frames, dtype=float)
        self.baseline, self.orientation = baseline, orientation
        self.landmarks = landmarks
        fixed_mask, self.warnings = color_mask(color, subject)
        color = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
        fixed_mask = cv2.resize(fixed_mask, size, interpolation=cv2.INTER_NEAREST)
        self.fixed = _feature(color, fixed_mask, "rgb")
        lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)
        # Chromatic channels distinguish tissues with similar brightness.
        self.rgb_bins = [np.minimum(23, (channel.astype(np.int32) * 24 // 256)) for channel in cv2.split(lab)]
        self.moving = []
        self.hu = []
        self.source_clipped = False
        for image in ct_images:
            raw_mask, warnings = density_mask(image, subject, depth)
            self.warnings.extend(warnings)
            self.source_clipped |= bool(np.any(raw_mask[0]) or np.any(raw_mask[-1]) or np.any(raw_mask[:, 0]) or np.any(raw_mask[:, -1]))
            mask = cv2.resize(raw_mask, size, interpolation=cv2.INTER_NEAREST)
            hu = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            self.hu.append(hu)
            self.moving.append(_feature(hu, mask, "density"))
        self.last_z, self.last_moving = None, None

    def blend(self, z):
        if z == self.last_z:
            return self.last_moving
        right = int(np.searchsorted(self.frames, z))
        if right == len(self.frames):
            right -= 1
        left = max(0, right - 1)
        if self.frames[right] == z or left == right:
            value = self.moving[right], self.hu[right]
        else:
            amount = float((z - self.frames[left]) / (self.frames[right] - self.frames[left]))
            value = ({key: self.moving[left][key] * (1 - amount) + self.moving[right][key] * amount
                      for key in ("planes", "outline", "edges", "interior")},
                     self.hu[left] * (1 - amount) + self.hu[right] * amount)
        self.last_z, self.last_moving = z, value
        return value

    def parameters(self, vector):
        return SpatialParameters(self.baseline.x_position + vector[1], self.baseline.y_position + vector[2],
                                 self.baseline.rotation_deg + vector[3], self.baseline.x_scale * math.exp(vector[4]),
                                 self.baseline.y_scale * math.exp(vector[5]))

    def measure(self, vector):
        parameters = self.parameters(vector)
        inverse = parameters_to_inverse_uv(parameters, self.orientation, width=513, height=305)
        matrix = pixel_matrix(inverse, self.width, self.height)
        moving, hu = self.blend(float(vector[0]))
        size, flags = (self.width, self.height), cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        warped = cv2.warpAffine(moving["planes"], matrix, size, flags=flags, borderValue=(0, 30, 30, 0))
        moving_edges = cv2.warpAffine(moving["edges"], matrix, size, flags=flags)
        moving_outline = cv2.warpAffine(moving["outline"], matrix, size, flags=flags)
        moving_interior = cv2.warpAffine(moving["interior"], matrix, size, flags=flags)
        warped_hu = cv2.warpAffine(hu, matrix, size, flags=flags, borderValue=-1024)
        support = cv2.warpAffine(np.ones_like(hu), matrix, size, flags=flags) > .99
        fixed = self.fixed
        fixed_body, body = fixed["mask"] > 0, warped[..., 0] > .5
        overlap = fixed_body & body & support
        interior = (fixed["interior"] > 0) & (moving_interior > .9) & support
        intersection = float(np.count_nonzero(overlap))
        body_recall = intersection / max(1, np.count_nonzero(fixed_body))
        precision = intersection / max(1, np.count_nonzero(body))
        iou = intersection / max(1, np.count_nonzero(fixed_body | body))
        uncovered = float(np.count_nonzero(fixed_body & ~support) / max(1, np.count_nonzero(fixed_body)))
        # Penalize anatomy pushed off canvas, even if all remaining tissue agrees.
        original_area = float(np.sum(moving["planes"][..., 0]))
        expected_area = original_area / abs(float(np.linalg.det(matrix[:, :2])))
        lost = max(0., 1 - float(np.sum(warped[..., 0])) / max(1., expected_area))
        silhouette_px = .5 * (float(np.sum(fixed["outline"] * warped[..., 1])) / max(1., float(np.sum(fixed["outline"]))) +
                              float(np.sum(moving_outline * fixed["outline_distance"])) / max(1., float(np.sum(moving_outline)))) / self.ratio
        fixed_edges = fixed["edges"] * support
        edge_count = min(float(fixed_edges.sum()), float(moving_edges.sum()))
        internal_px = .5 * (float(np.sum(fixed_edges * warped[..., 2])) / max(1., float(fixed_edges.sum())) +
                            float(np.sum(moving_edges * fixed["edge_distance"])) / max(1., float(moving_edges.sum()))) / self.ratio
        a, b = fixed["gradient"][interior], warped[..., 3][interior]
        if len(a) > 150 and float(np.std(a)) > .02 and float(np.std(b)) > .02:
            a, b = a - a.mean(), b - b.mean()
            gradient = max(0., float(np.dot(a, b) / max(1e-6, float(np.linalg.norm(a) * np.linalg.norm(b)))))
        else:
            gradient = 0.
        nmi, informative = normalized_mutual_information(self.rgb_bins, warped_hu, interior)
        internal = .65 * math.exp(-internal_px / 5.) + .35 * gradient if edge_count >= 20 else 0.
        coverage = .5 * (body_recall + precision) - uncovered - lost
        objective = .30 * (1 - math.exp(-silhouette_px / 5.)) + .30 * (1 - internal) + .25 * (1 - nmi) + .15 * (1 - coverage)
        rms = None
        if self.landmarks:
            errors = []
            sample = np.asarray(inverse).reshape(2, 3)
            for pair in self.landmarks:
                predicted = sample @ np.array([*pair["color"], 1.])
                errors.append(float(np.sum(((predicted - pair["density"]) * [512, 304]) ** 2)))
            rms = math.sqrt(float(np.mean(errors)))
            objective += .1 * min(rms ** 2, 400) / 4
        return {"objective": float(objective), "iou": iou, "body_recall": body_recall,
                "body_precision": precision, "canvas_uncovered_fraction": uncovered,
                "source_body_outside_fraction": lost, "silhouette_distance_px": silhouette_px,
                "internal_edge_distance_px": internal_px, "internal_gradient": gradient,
                "internal_agreement": internal, "rgb_hu_nmi": nmi,
                "informative": bool(informative and edge_count >= 20 and np.count_nonzero(interior) >= 150),
                "landmark_rms_px": rms}


class _BudgetReached(Exception):
    pass


def refine_pair(color, ct_frames, ct_images, request, subject="male", depth=.3,
                z_bounds=None, max_seconds=MAX_SECONDS):
    start = time.monotonic()
    cv2.setNumThreads(1)
    original_uv = list(request["inverse_uv"])
    original_z = _finite(request["ct_frame"], "CT frame")
    orientation = request.get("orientation", "flip_y")
    if orientation not in ("identity", "flip_x", "flip_y", "rotate_180"):
        raise ValueError("Unsupported fixed CT orientation")
    # Viewer coordinates use a 512x304 continuous canvas (including its outer
    # edge); optimizer helpers use width-1 sample coordinates, hence +1 here.
    baseline = inverse_uv_to_parameters(original_uv, orientation, width=513, height=305)
    reconstructed = parameters_to_inverse_uv(baseline, orientation, width=513, height=305)
    if not np.allclose(reconstructed, original_uv, atol=1e-5, rtol=1e-5):
        raise ValueError("The current transform contains shear; local refinement supports rotation and independent scales only")
    landmarks = request.get("landmarks") or []
    for pair in landmarks:
        for name in ("color", "density"):
            point = np.asarray(pair[name], dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)) or np.any(point < 0) or np.any(point > 1):
                raise ValueError("Landmark coordinates must be two finite values inside source UV")
    low, high = z_bounds or (max(float(ct_frames[0]), original_z - 8), min(float(ct_frames[-1]), original_z + 8))
    if landmarks:
        low = high = original_z
    if not low <= original_z <= high:
        raise ValueError("Current CT frame lies outside refinement bounds")
    origin = np.array([original_z, 0, 0, 0, 0, 0], dtype=float)
    bounds = [(low, high), (max(-12, -512 - baseline.x_position), min(12, 512 - baseline.x_position)),
              (max(-12, -304 - baseline.y_position), min(12, 304 - baseline.y_position)),
              (max(-4, -20 - baseline.rotation_deg), min(4, 20 - baseline.rotation_deg)),
              (max(math.log(.92), math.log(.25 / baseline.x_scale)), min(math.log(1.08), math.log(4 / baseline.x_scale))),
              (max(math.log(.92), math.log(.25 / baseline.y_scale)), min(math.log(1.08), math.log(4 / baseline.y_scale)))]
    if any(a > b or not a <= value <= b for (a, b), value in zip(bounds, origin)):
        raise ValueError("The current transform lies outside supported local scale limits")
    warnings = ["Automatic RGB/CT correspondence is a registration proxy; inspect the CT plane, skin boundary and internal anatomy before saving."]
    if landmarks:
        warnings.append("CT depth is fixed because the current landmark pairs refer to this exact CT plane.")
    coarse = PairObjective(color, ct_frames, ct_images, baseline, orientation, subject, depth, landmarks)
    full = PairObjective(color, ct_frames, ct_images, baseline, orientation, subject, depth, landmarks, width=512)
    before = full.measure(origin)
    answer = {"accepted": False, "ct_frame": original_z, "inverse_uv": original_uv,
              "parameters": {"z_position": original_z, **asdict(baseline)},
              "metrics": {"before": before, "after": dict(before)}, "warnings": warnings,
              "method": METHOD, "quality": "needs-human-review", "bounds": {"ct_frame": [low, high],
              "translation_px": 12, "rotation_deg": 4, "scale_fraction": .08}}
    warnings.extend(dict.fromkeys(coarse.warnings))
    if full.source_clipped:
        warnings.append("CT tissue reaches the source field-of-view edge; alignment cannot recover missing anatomy.")
    if not before["informative"]:
        warnings.append("Insufficient shared internal texture or RGB/HU variation for a reliable fine adjustment; current values are retained.")
        answer["elapsed_seconds"] = time.monotonic() - start
        return answer
    evaluations = 0
    candidates = [(before["objective"], origin.copy())]
    best_coarse = [float("inf"), origin.copy()]

    def objective(vector):
        nonlocal evaluations
        if time.monotonic() - start > max_seconds:
            raise _BudgetReached()
        vector = np.array(vector, dtype=float, copy=True)
        if not all(lo - 1e-7 <= x <= hi + 1e-7 for (lo, hi), x in zip(bounds, vector)):
            return 1000.
        measured = coarse.measure(vector)
        evaluations += 1
        displacement = (vector - origin) / [8, 12, 12, 4, .08, .08]
        score = measured["objective"] + .005 * float(np.mean(displacement ** 2))
        if score < best_coarse[0] and measured["informative"]:
            best_coarse[:] = [score, vector.copy()]
        return score

    try:
        seeds = []
        z_candidates = sorted({original_z, low, high, *[float(z) for z in ct_frames if low <= z <= high]})
        for z in z_candidates:
            vector = origin.copy(); vector[0] = z
            seeds.append((objective(vector), vector))
        seeds.sort(key=lambda value: value[0])
        starts = [origin.copy()]
        for _, vector in seeds:
            if all(abs(vector[0] - existing[0]) > .75 for existing in starts):
                starts.append(vector.copy())
            if len(starts) >= 3:
                break
        # Powell does not mutate an accepted starting transform. Each run has
        # its own vector and bounded budget; a failed run can never overwrite it.
        for vector in starts:
            best_coarse[:] = [objective(vector), vector.copy()]
            try:
                result = minimize(objective, vector.copy(), method="Powell", bounds=bounds,
                                  options={"maxiter": 8, "maxfev": 450, "xtol": .025, "ftol": 1e-5})
                if np.all(np.isfinite(result.x)) and result.status in (0, 1, 2):
                    objective(result.x.copy())
            except (ValueError, FloatingPointError, cv2.error):
                warnings.append("An optimization start failed; its unchecked result was discarded.")
                continue
            candidate = best_coarse[1].copy()
            measured = full.measure(candidate)
            candidates.append((measured["objective"], candidate))
    except _BudgetReached:
        warnings.append("The local optimization time budget was reached; only completed, validated candidates were considered.")
    # Final acceptance is at 512x304 with independent anatomy and coverage guards.
    for _, vector in sorted(candidates, key=lambda value: value[0]):
        after = full.measure(vector)
        internal_improvement = (after["internal_agreement"] > before["internal_agreement"] + .001 or
                                after["rgb_hu_nmi"] > before["rgb_hu_nmi"] + .001)
        acceptable = (after["informative"] and after["objective"] < before["objective"] - .0005
                      and internal_improvement
                      and after["internal_agreement"] >= before["internal_agreement"] - .015
                      and after["rgb_hu_nmi"] >= before["rgb_hu_nmi"] - max(.002, .05 * before["rgb_hu_nmi"])
                      and after["body_recall"] >= before["body_recall"] - .015
                      and after["body_precision"] >= before["body_precision"] - .015
                      and after["canvas_uncovered_fraction"] <= before["canvas_uncovered_fraction"] + .005
                      and after["source_body_outside_fraction"] <= before["source_body_outside_fraction"] + .005
                      and (after["landmark_rms_px"] is None or after["landmark_rms_px"] <= max(2., before["landmark_rms_px"])))
        if not acceptable:
            continue
        parameters = full.parameters(vector)
        answer.update(accepted=True, ct_frame=float(vector[0]),
                      inverse_uv=parameters_to_inverse_uv(parameters, orientation, width=513, height=305),
                      parameters={"z_position": float(vector[0]), **asdict(parameters)},
                      metrics={"before": before, "after": after})
        if abs(vector[0] - low) < .1 or abs(vector[0] - high) < .1:
            if high > low:
                warnings.append("Best CT correspondence is near the local Z search limit; check neighboring planes manually.")
        break
    if not answer["accepted"]:
        warnings.append("No candidate improved internal correspondence and total fit without sacrificing coverage; current values are retained.")
    answer.update(elapsed_seconds=time.monotonic() - start, evaluations=evaluations)
    return answer


def refine(directory, color_frame, request):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("baked"):
        raise ValueError("Fine alignment requires baked color and density source images")
    rgb_layers = manifest["rgb"]["layers"]
    color_layer = next((row for row in rgb_layers if row["frame"] == color_frame), None)
    if color_layer is None:
        raise ValueError("The requested RGB frame does not exist")
    current_inputs = {"manifest_sha256": sha256(directory / "manifest.json"),
                      "alignment_sha256": sha256(directory / "alignment.json")}
    if request.get("inputs") is not None and request["inputs"] != current_inputs:
        raise ValueError("Registration inputs changed; reload before Auto Align")
    reviewed_path = directory / "reviewed-alignment.json"
    reviewed = json.loads(reviewed_path.read_text()).get("anchors", []) if reviewed_path.exists() else []
    reviewed = [row for row in reviewed if all((row.get("inputs") or {}).get(key) == value
                                              for key, value in current_inputs.items())]
    z = _finite(request["ct_frame"], "CT frame")
    bounds = local_z_bounds(manifest, z, color_frame, reviewed, request.get("z_bounds"), request.get("landmarks"))
    ct_layers = sorted(manifest["density"]["layers"], key=lambda row: row["frame"])
    frames = np.asarray([row["frame"] for row in ct_layers], dtype=float)
    first = max(0, int(np.searchsorted(frames, bounds[0], side="right")) - 1)
    last = min(len(frames), int(np.searchsorted(frames, bounds[1], side="left")) + 1)
    selected = ct_layers[first:last]
    color = read_baked_rgb(layer_slice(directory, color_layer, "rgb"))
    density = manifest["density"]
    images = [read_baked_ct(layer_slice(directory, row, "density"), int(density["width"]), int(density["height"])) for row in selected]
    result = refine_pair(color, [row["frame"] for row in selected], images, request,
                         manifest.get("subject", directory.name), color_frame / max(1, max(row["frame"] for row in rgb_layers)), bounds)
    result.update(color_frame=color_frame, source="baked RGB and native-HU analysis pairs; no profiles or source files modified")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--color-frame", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        request = json.load(sys.stdin)
        print(json.dumps(refine(args.processed, args.color_frame, request), allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        print(json.dumps({"error": str(error)}, allow_nan=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
