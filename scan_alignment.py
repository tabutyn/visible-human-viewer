#!/usr/bin/env python3
"""Read-only QC of baked color/CT pairs; write only the explicit report target.

These measurements are image proxies, not anatomical correspondence labels.
The audit evaluates alignment.json without applying manual residual interpolation.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from register_modalities import _internal_landmark_edges, _symmetric_edge_distance

ANALYSIS_SIZE = (512, 304)
THRESHOLDS = {
    "minimum_mask_pixels": 32,
    "minimum_component_pixels": 8,
    "annotation_rectangular_fill_min": .88,
    "annotation_yellow_fraction_min": .75,
    "mask_span_percentile_trim": 2.5,
    "body_recall_min": .88,
    "iou_min": .70,
    "canvas_uncovered_fraction_max": .02,
    "mask_size_ratio_min": .85,
    "mask_size_ratio_max": 1.18,
    "extent_ambiguity_iou_min": .85,
    "extent_ambiguity_log_span_min": .35,
    "interior_edge_disagreement_max_px": 12.,
    "minimum_interior_edge_pixels": 20,
    "possible_plane_mismatch_iou_min": .78,
    "source_border_contact_fraction_max": .01,
    "detached_component_pixels_min": 64,
    "detached_component_overlap_max": .10,
    "neighbor_scale_change_per_12_frames_max": .10,
    "neighbor_center_shift_per_12_frames_max_px": 6.,
    "density_gap_multiple_of_median_step": 2.,
}
CAVEATS = [
    "QC proxies only: silhouettes and interior edges do not establish anatomical correspondence.",
    "Color tissue and CT threshold masks can miss tissue or include resin/support; components of at least 8 analysis pixels are retained, including paired limbs and small detached parts.",
    "Nearly filled rectangular yellow color components are excluded as possible printed slice labels; excluded pixel counts are reported, and this heuristic is not an anatomical segmentation.",
    "Only thin horizontal CT components near the bottom are excluded as possible support; removed pixels are reported.",
    "Detached CT components are retained and flagged when they miss color tissue; support wedges can occur above the legs, and their identity cannot be inferred from threshold masks.",
    "Width and height ratios use the central 95% of mask coordinates to reduce distant fleck sensitivity; all retained mask pixels still contribute to coverage and overlap.",
    "Source-border contact suggests inspecting field of view; it does not establish source clipping.",
    "Potential plane mismatch is a heuristic: good silhouette overlap with disagreeing interior edges can also reflect contrast or segmentation differences.",
    "CT is sampled at the nearest available layer to the registered depth; no synthetic density is supplied outside registered coverage or across a large source gap.",
    "Base automatic profile only. Manual anchors add scan locations but their transforms are not applied.",
    "Issue spans group sampled warnings; unscanned frames inside a span are not individually verified.",
    "Input SHA-256 hashes identify manifest and profile metadata, not the contents of every baked image file.",
]


def retained_components(candidate, *, ct=False):
    """Keep paired/small anatomy without a largest-component-relative cutoff."""
    mask = np.asarray(candidate, np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep, ignored_small, ignored_support = [], 0, 0
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < THRESHOLDS["minimum_component_pixels"]:
            ignored_small += area
        elif ct and y > height * .65 and w / max(1, h) > 6 and h < height * .12:
            ignored_support += area
        else:
            keep.append(label)
    return np.isin(labels, keep).astype(np.uint8), {
        "retained_components": len(keep), "ignored_small_pixels": ignored_small,
        "ignored_possible_support_pixels": ignored_support,
    }


def anatomy_mask(image, kind):
    if kind == "density":
        candidate = cv2.morphologyEx((image > -400).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return retained_components(candidate, ct=True)
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    tissue = (r > 25) & (r > b * 1.08 + 4) & (g > b * .72)
    if np.count_nonzero(tissue) < 20:
        lum = .2126 * r + .7152 * g + .0722 * b
        blue = (b > r * 1.08 + 5) & (b > g * 1.04 + 5)
        tissue = (lum > 18) & (lum < 248) & ~blue
    candidate = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    annotation_pixels = 0
    yellow = (g > r * .85) & (b < g * .70)
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if w >= 10 and h >= 10 and area / (w * h) >= THRESHOLDS["annotation_rectangular_fill_min"]:
            component = labels == label
            if np.count_nonzero(yellow & component) / area >= THRESHOLDS["annotation_yellow_fraction_min"]:
                candidate[component] = 0
                annotation_pixels += area
    mask, metadata = retained_components(candidate)
    metadata["ignored_possible_annotation_pixels"] = annotation_pixels
    return mask, metadata


def interpolate(knots, depth, field):
    ordered = sorted(knots, key=lambda item: item["color_depth"])
    if not ordered:
        raise ValueError(f"alignment requires {field} knots")
    x = [item["color_depth"] for item in ordered]
    y = np.asarray([item[field] for item in ordered], float)
    if y.ndim == 1:
        return float(np.interp(depth, x, y))
    return np.asarray([np.interp(depth, x, y[:, index]) for index in range(y.shape[1])])


def pixel_inverse(inverse_uv, width=512, height=304):
    matrix = np.eye(3)
    matrix[:2] = np.asarray(inverse_uv, float).reshape(2, 3)
    to_pixels = np.diag([width - 1., height - 1., 1.])
    return to_pixels @ matrix @ np.linalg.inv(to_pixels)


def _bounds(mask):
    ys, xs = np.nonzero(mask)
    return (xs, ys, _robust_span(xs), _robust_span(ys)) if len(xs) else (xs, ys, 0., 0.)


def _robust_span(coordinates):
    trim = THRESHOLDS["mask_span_percentile_trim"]
    lo, hi = np.percentile(coordinates, [trim, 100 - trim])
    return float(hi - lo + 1)


def measure_pair(color, ct, inverse_uv):
    """Separate body overlap from UV canvas exclusion, using destination→source UV."""
    color_mask, color_segmentation = anatomy_mask(color, "color")
    ct_mask, ct_segmentation = anatomy_mask(ct, "density")
    height, width = color_mask.shape
    matrix = pixel_inverse(inverse_uv, width, height)
    forward = np.linalg.inv(matrix)
    warped_mask = cv2.warpAffine(ct_mask, matrix[:2], (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    ys, xs = np.indices((height, width))
    sample_x = matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]
    sample_y = matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]
    valid = (sample_x >= 0) & (sample_x <= width - 1) & (sample_y >= 0) & (sample_y <= height - 1)
    fixed, moving = color_mask > 0, warped_mask > 0
    color_area, ct_area = int(fixed.sum()), int(ct_mask.sum())
    intersection, union = int((fixed & moving).sum()), int((fixed | moving).sum())
    _, _, cw, ch = _bounds(color_mask)
    source_x, source_y, _, _ = _bounds(ct_mask)
    projected_width = projected_height = None
    if len(source_x):
        px = forward[0, 0] * source_x + forward[0, 1] * source_y + forward[0, 2]
        py = forward[1, 0] * source_x + forward[1, 1] * source_y + forward[1, 2]
        projected_width, projected_height = _robust_span(px), _robust_span(py)
    color_edges = _internal_landmark_edges(color, "color", color_mask)
    source_edges = _internal_landmark_edges(ct, "density", ct_mask)
    ct_edges = cv2.warpAffine(source_edges, matrix[:2], (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    edge_counts = [int(np.count_nonzero(color_edges)), int(np.count_nonzero(ct_edges))]
    edge_distance = _symmetric_edge_distance(color_edges, ct_edges) if min(edge_counts) >= THRESHOLDS["minimum_interior_edge_pixels"] else None
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(warped_mask, 8)
    unmatched_components = 0
    if component_count > 2:
        for label in range(1, component_count):
            area = int(component_stats[label, cv2.CC_STAT_AREA])
            if area >= THRESHOLDS["detached_component_pixels_min"] and np.count_nonzero(fixed & (component_labels == label)) / area < THRESHOLDS["detached_component_overlap_max"]:
                unmatched_components += 1
    perimeter = np.concatenate([ct_mask[0], ct_mask[-1], ct_mask[1:-1, 0], ct_mask[1:-1, -1]])
    center = forward @ np.array([(width - 1) / 2, (height - 1) / 2, 1.])
    valid_masks = min(color_area, ct_area) >= THRESHOLDS["minimum_mask_pixels"]
    return {
        "color_mask_pixels": color_area, "ct_mask_pixels": ct_area,
        "body_recall": intersection / color_area if color_area and valid_masks else None,
        "body_uncovered_fraction": 1 - intersection / color_area if color_area and valid_masks else None,
        "canvas_uncovered_fraction": float((fixed & ~valid).sum() / color_area) if color_area else None,
        "iou": intersection / union if union and valid_masks else None,
        "mask_width_ratio": projected_width / cw if cw and projected_width and valid_masks else None,
        "mask_height_ratio": projected_height / ch if ch and projected_height and valid_masks else None,
        "interior_edge_disagreement_px": edge_distance,
        "color_interior_edge_pixels": edge_counts[0], "ct_interior_edge_pixels": edge_counts[1],
        "source_border_contact_fraction": float(np.count_nonzero(perimeter) / len(perimeter)),
        "detached_unmatched_ct_components": unmatched_components,
        "display_scale_x": float(np.linalg.norm(forward[:2, 0])),
        "display_scale_y": float(np.linalg.norm(forward[:2, 1])),
        "display_center_x": float(center[0]), "display_center_y": float(center[1]),
        "color_segmentation": color_segmentation, "ct_segmentation": ct_segmentation,
    }


def classify(metrics):
    warnings = []

    def add(code, reason, severity):
        warnings.append((code, reason, min(99, round(severity))))

    if min(metrics["color_mask_pixels"], metrics["ct_mask_pixels"]) < THRESHOLDS["minimum_mask_pixels"]:
        add("mask_unreliable", "Too little segmented tissue for reliable overlap measurements", 65)
    recall = metrics["body_recall"]
    if recall is not None and recall < THRESHOLDS["body_recall_min"]:
        add("body_uncovered", f"CT tissue mask leaves {1 - recall:.0%} of color tissue uncovered", 40 + 65 * (1 - recall))
    canvas = metrics["canvas_uncovered_fraction"]
    if canvas is not None and canvas > THRESHOLDS["canvas_uncovered_fraction_max"]:
        add("canvas_clipping", f"{canvas:.0%} of color tissue samples outside the CT canvas", 70 + 29 * canvas)
    iou = metrics["iou"]
    if iou is not None and iou < THRESHOLDS["iou_min"]:
        add("outline_disagreement", f"Body outline overlap is {iou:.0%}", 40 + 50 * (1 - iou))
    for axis in ("width", "height"):
        ratio = metrics[f"mask_{axis}_ratio"]
        if ratio is not None and not THRESHOLDS["mask_size_ratio_min"] <= ratio <= THRESHOLDS["mask_size_ratio_max"]:
            # A remote component can dominate extents even while most tissue
            # coincides. Do not rank that inconsistent proxy as a severe fit.
            if iou is not None and iou > THRESHOLDS["extent_ambiguity_iou_min"] and abs(math.log(max(.01, ratio))) > THRESHOLDS["extent_ambiguity_log_span_min"]:
                add("mask_extent_ambiguous", f"Mask {axis} differs despite high overlap; inspect detached tissue, labels or support", 35)
            else:
                add("size_mismatch", f"Transformed CT tissue {axis} is {ratio:.0%} of color tissue {axis}", 45 + 40 * abs(math.log(max(.01, ratio))))
    distance = metrics["interior_edge_disagreement_px"]
    if distance is not None and distance > THRESHOLDS["interior_edge_disagreement_max_px"]:
        add("interior_edge_disagreement", f"Interior edges disagree by {distance:.1f} analysis pixels", 45 + min(35, distance))
        if iou is not None and iou >= THRESHOLDS["possible_plane_mismatch_iou_min"]:
            add("possible_plane_mismatch", "Possible depth-plane mismatch: outlines agree better than interior edges; inspect matching landmarks", 60)
    if metrics["source_border_contact_fraction"] > THRESHOLDS["source_border_contact_fraction_max"]:
        add("source_canvas_contact", "CT mask touches source canvas edge; inspect source field of view", 60)
    if metrics["detached_unmatched_ct_components"]:
        add("mask_support_ambiguous", "Detached CT components miss color tissue; inspect support, positioning and segmentation", 66)
    return warnings


def _warnings(sample, warnings):
    sample["issue_codes"] = list(dict.fromkeys(sample.get("issue_codes", []) + [item[0] for item in warnings]))
    sample["reasons"] = list(dict.fromkeys(sample.get("reasons", []) + [item[1] for item in warnings]))
    sample["severity"] = max([sample.get("severity", 0), *[item[2] for item in warnings]])


def add_neighbor_warnings(samples):
    for previous, current in zip(samples, samples[1:]):
        if previous["status"] != "evaluated" or current["status"] != "evaluated":
            continue
        a, b = previous["metrics"], current["metrics"]
        factor = 12 / max(1, current["color_frame"] - previous["color_frame"])
        scale_change = max(abs(math.log(b[f"display_scale_{axis}"] / a[f"display_scale_{axis}"])) for axis in ("x", "y")) * factor
        shift = math.hypot(b["display_center_x"] - a["display_center_x"], b["display_center_y"] - a["display_center_y"]) * factor
        b["neighbor_scale_change_per_12_frames"] = scale_change
        b["neighbor_center_shift_per_12_frames_px"] = shift
        if scale_change > THRESHOLDS["neighbor_scale_change_per_12_frames_max"] or shift > THRESHOLDS["neighbor_center_shift_per_12_frames_max_px"]:
            _warnings(current, [("transform_change", "Alignment scale or position changes rapidly between nearby sampled frames", 72)])


def select_sample_indices(layers, profile, anchors, stride):
    frames = np.asarray([item["frame"] for item in layers], float)
    last = max(1., float(frames.max()))
    indices = set(range(0, len(layers), stride)) | {len(layers) - 1}
    depths = [item["color_depth"] for item in profile.get("spatial_knots", [])]
    depths += list(profile.get("coverage", {}).get("color_depth", [0., 1.]))
    for depth in depths:
        index = int(np.argmin(np.abs(frames - depth * last)))
        indices.update(i for i in (index - 1, index, index + 1) if 0 <= i < len(layers))
    for anchor in anchors:
        frame = anchor.get("color_frame", anchor.get("color_depth", 0) * last)
        indices.add(int(np.argmin(np.abs(frames - frame))))
    return sorted(indices)


def group_issues(samples, subject, max_gap):
    groups, current = [], []

    def category(sample):
        return sample["status"] if sample["status"] != "evaluated" else "alignment"

    for sample in samples:
        if not sample["severity"]:
            if current:
                groups.append(current)
                current = []
            continue
        if current and (sample["color_frame"] - current[-1]["color_frame"] > max_gap or sample["color_frame"] - current[0]["color_frame"] > max_gap * 4 or category(sample) != category(current[-1])):
            groups.append(current)
            current = []
        current.append(sample)
    if current:
        groups.append(current)
    issues = []
    for group in groups:
        peak = max(group, key=lambda sample: sample["severity"])
        start, end = group[0]["color_frame"], group[-1]["color_frame"]
        issues.append({**peak, "id": f"{subject}-{start}-{end}", "start_frame": start,
                       "end_frame": end, "sample_count": len(group)})
    return sorted(issues, key=lambda issue: (-issue["severity"], issue["color_frame"]))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def scan(processed, subject, stride=12, progress=print):
    processed = Path(processed).resolve()
    manifest_path, alignment_path = processed / "manifest.json", processed / "alignment.json"
    manual_path = processed / "manual-alignment.json"
    inputs = {"manifest_sha256": _digest(manifest_path), "alignment_sha256": _digest(alignment_path),
              "manual_alignment_sha256": _digest(manual_path), "corrections_sha256": _digest(processed / "corrections.json")}
    manifest, profile = json.loads(manifest_path.read_text()), json.loads(alignment_path.read_text())
    if not manifest.get("baked"):
        raise ValueError("scan requires a completed baked manifest")
    if manifest.get("subject", subject) != subject or profile.get("subject", subject) != subject:
        raise ValueError("subject does not match processed inputs")
    for key in ("manifest_sha256", "corrections_sha256"):
        if profile.get("inputs", {}).get(key) and profile["inputs"][key] != inputs[key]:
            raise ValueError(f"alignment is stale: {key} differs")
    rgb = sorted(manifest["rgb"]["layers"], key=lambda item: item["frame"])
    density = sorted(manifest["density"]["layers"], key=lambda item: item["frame"])
    if not rgb or not density or stride < 1:
        raise ValueError("both modalities and a positive stride are required")
    anchors = json.loads(manual_path.read_text()).get("anchors", []) if manual_path.is_file() else []
    known_review_frames = {"male": (80, 117, 121, 160, 240, 1211, 1300, 1370), "female": (400,)}[subject]
    anchors = [*anchors, *({"color_frame": frame} for frame in known_review_frames if rgb[0]["frame"] <= frame <= rgb[-1]["frame"])]
    selected = select_sample_indices(rgb, profile, anchors, stride)
    density_frames = [item["frame"] for item in density]
    expected_step = float(np.median(np.diff(density_frames))) if len(density_frames) > 1 else 1.
    coverage = profile.get("coverage", {}).get("color_depth", [0., 1.])
    maximum_frame = max(1, rgb[-1]["frame"])
    density_width = int(manifest["density"]["width"])
    density_height = int(manifest["density"]["height"])
    samples = []
    for ordinal, index in enumerate(selected):
        item = rgb[index]
        frame, depth = int(item["frame"]), item["frame"] / maximum_frame
        sample = {"color_frame": frame, "color_depth": depth, "ct_frame": None,
                  "status": "evaluated", "severity": 0, "issue_codes": [], "reasons": [], "metrics": {}}
        samples.append(sample)
        color_path = processed / item["file"]
        if not color_path.is_file():
            sample["status"] = "missing_color_source"
            _warnings(sample, [("missing_color_source", "Baked color source file is missing", 100)])
        elif not coverage[0] <= depth <= coverage[1]:
            sample["status"] = "outside_registered_coverage"
            _warnings(sample, [("outside_registered_coverage", "Outside registered depth coverage; no CT correspondence is established", 90)])
        else:
            mapped = interpolate(profile["depth_knots"], depth, "ct_frame")
            sample["mapped_ct_frame"] = mapped
            right = bisect_left(density_frames, mapped)
            nearest = min(range(max(0, right - 1), min(len(density), right + 1)), key=lambda i: abs(density_frames[i] - mapped))
            layer = density[nearest]
            source_path = processed / layer["file"]
            large_gap = (0 < right < len(density) and density_frames[right - 1] < mapped < density_frames[right]
                         and density_frames[right] - density_frames[right - 1] > expected_step * THRESHOLDS["density_gap_multiple_of_median_step"])
            if mapped < density_frames[0] or mapped > density_frames[-1] or large_gap:
                sample["status"] = "missing_density_correspondence"
                _warnings(sample, [("density_source_gap", "Registered CT depth falls outside available layers or in a large source gap", 100)])
            elif not source_path.is_file():
                sample["status"] = "missing_density_source"
                _warnings(sample, [("missing_density_source", "Mapped baked CT source file is missing", 100)])
            else:
                sample["ct_frame"] = int(layer["frame"])
                try:
                    with Image.open(color_path) as image:
                        color = cv2.resize(np.asarray(image.convert("RGB")), ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
                    values = np.fromfile(source_path, dtype=">u2")
                    if values.size != density_width * density_height:
                        raise ValueError("invalid baked density dimensions")
                    ct = values.reshape(density_height, density_width).astype(np.float32) - 1024.
                    ct = cv2.resize(ct, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
                    matrix = interpolate(profile["spatial_knots"], depth, "inverse_uv")
                    metrics = measure_pair(color, ct, matrix)
                    sample["metrics"] = metrics
                    _warnings(sample, classify(metrics))
                except (OSError, ValueError, np.linalg.LinAlgError, cv2.error) as error:
                    sample["status"] = "unreadable_source_or_transform"
                    _warnings(sample, [("unreadable_source_or_transform", f"Cannot measure this pair: {error}", 100)])
        if (ordinal + 1) % 20 == 0 or ordinal + 1 == len(selected):
            progress(f"alignment scan {subject}: {ordinal + 1}/{len(selected)} samples; color frame {frame}")
    add_neighbor_warnings(samples)
    frame_step = float(np.median(np.diff([item["frame"] for item in rgb]))) if len(rgb) > 1 else 1.
    issues = group_issues(samples, subject, max_gap=max(2., stride * frame_step * 1.6))
    evaluated = [sample for sample in samples if sample["status"] == "evaluated"]
    summary = {
        "total_color_frames": len(rgb), "total_density_frames": len(density),
        "scanned_frames": len(samples), "evaluated_frames": len(evaluated),
        "flagged_frames": sum(sample["severity"] > 0 for sample in samples), "issue_count": len(issues),
        "outside_coverage_frames": sum(not coverage[0] <= item["frame"] / maximum_frame <= coverage[1] for item in rgb),
        "outside_coverage_samples": sum(sample["status"] == "outside_registered_coverage" for sample in samples),
        "missing_source_samples": sum(sample["status"].startswith("missing_") for sample in samples),
        "qc_proxies_only": True, "profile": "base_alignment", "stride": stride,
        "analysis_size": list(ANALYSIS_SIZE), "coverage_color_depth": coverage,
        "max_severity": max((sample["severity"] for sample in samples), default=0),
    }
    for name in ("body_recall", "iou", "interior_edge_disagreement_px", "canvas_uncovered_fraction"):
        values = [sample["metrics"][name] for sample in evaluated if sample["metrics"][name] is not None]
        summary[f"mean_{name}"] = float(np.mean(values)) if values else None
    return {"version": 1, "subject": subject, "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
            "inputs": inputs, "thresholds": THRESHOLDS, "caveats": CAVEATS,
            "summary": summary, "issues": issues, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True, help="Processed subject directory containing manifest.json")
    parser.add_argument("--subject", choices=("male", "female"), required=True)
    parser.add_argument("--stride", type=int, default=12, help="Scan every N color layers plus knots, anchors and coverage endpoints")
    parser.add_argument("--report", type=Path, required=True, help="Explicit output JSON path; no profiles are activated")
    args = parser.parse_args()
    report_path, processed = args.report.resolve(), args.processed.resolve()
    if report_path.suffix != ".json" or report_path.name in {"manifest.json", "alignment.json", "manual-alignment.json", "corrections.json", "overrides.json", "qc.json"} or any(report_path.is_relative_to(processed / folder) for folder in ("rgb", "density")):
        parser.error("--report must name a separate JSON report, never a source/profile file")
    report = scan(processed, args.subject, args.stride, progress=lambda message: print(message, flush=True))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(report_path), **report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
