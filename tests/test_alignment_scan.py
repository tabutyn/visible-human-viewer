import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
import scan_alignment as audit


def paired_images():
    color = np.zeros((304, 512, 3), np.uint8)
    ct = np.full((304, 512), -1024., np.float32)
    for x in (170, 330):
        cv2.ellipse(color, (x, 150), (45, 80), 0, 0, 360, (150, 100, 70), -1)
        cv2.ellipse(ct, (x, 150), (45, 80), 0, 0, 360, 60., -1)
        cv2.circle(color, (x, 140), 15, (245, 230, 190), -1)
        cv2.circle(ct, (x, 140), 15, 700., -1)
    return color, ct


def test_scan_masks_keep_paired_limbs_and_small_detached_parts():
    color, ct = paired_images()
    cv2.circle(color, (260, 245), 5, (150, 100, 70), -1)
    cv2.circle(ct, (260, 245), 5, 60., -1)
    for image, kind in ((color, "color"), (ct, "density")):
        mask, metadata = audit.anatomy_mask(image, kind)
        assert mask[150, 170] and mask[150, 330] and mask[245, 260]
        assert metadata["retained_components"] == 3


def test_overlap_and_geometric_canvas_exclusion_are_distinct():
    color, ct = paired_images()
    exact = audit.measure_pair(color, ct, [1, 0, 0, 0, 1, 0])
    assert exact["iou"] > .99 and exact["body_recall"] > .99
    assert exact["canvas_uncovered_fraction"] == 0
    # Display CT at 60% width. The source canvas still includes most of the
    # color mask, but body recall is low: tissue coverage is not canvas coverage.
    narrow = audit.measure_pair(color, ct, [1 / .6, 0, (.6 - 1) / (2 * .6), 0, 1, 0])
    assert narrow["body_recall"] < .6
    assert narrow["canvas_uncovered_fraction"] < .1
    assert abs(narrow["mask_width_ratio"] - .6) < .01
    shifted = audit.measure_pair(color, ct, [1, 0, .8, 0, 1, 0])
    assert shifted["canvas_uncovered_fraction"] > .7
    codes = [entry[0] for entry in audit.classify(shifted)]
    assert "canvas_clipping" in codes and "body_uncovered" in codes


def test_support_above_limbs_is_retained_and_flagged_as_ambiguous():
    color, ct = paired_images()
    cv2.rectangle(ct, (245, 15), (270, 45), 30., -1)
    metrics = audit.measure_pair(color, ct, [1, 0, 0, 0, 1, 0])
    assert metrics["ct_segmentation"]["retained_components"] == 3
    assert metrics["detached_unmatched_ct_components"] == 1
    assert "mask_support_ambiguous" in [entry[0] for entry in audit.classify(metrics)]


def test_tiny_distant_color_component_does_not_dominate_size_ratio():
    color, ct = paired_images()
    cv2.circle(color, (15, 15), 4, (150, 100, 70), -1)
    metrics = audit.measure_pair(color, ct, [1, 0, 0, 0, 1, 0])
    assert metrics["color_segmentation"]["retained_components"] == 3
    assert .95 < metrics["mask_width_ratio"] < 1.05
    assert .95 < metrics["mask_height_ratio"] < 1.05
    assert metrics["body_recall"] < 1  # The component still counts in coverage.


def test_rectangular_yellow_printed_slice_label_is_excluded_with_audit_count():
    color, ct = paired_images()
    cv2.rectangle(color, (350, 260), (390, 290), (195, 208, 120), -1)
    metrics = audit.measure_pair(color, ct, [1, 0, 0, 0, 1, 0])
    assert metrics["color_segmentation"]["ignored_possible_annotation_pixels"] > 1000
    assert metrics["color_segmentation"]["retained_components"] == 2
    assert metrics["iou"] > .99


def test_sample_selection_includes_exact_anchors_and_coverage_neighbors():
    layers = [{"frame": i} for i in range(101)]
    profile = {"coverage": {"color_depth": [.11, .91]}, "spatial_knots": [{"color_depth": .47}]}
    selected = audit.select_sample_indices(layers, profile, [{"color_frame": 33}], stride=20)
    assert {0, 100, 10, 11, 12, 46, 47, 48, 90, 91, 92, 33}.issubset(selected)


def test_scan_distinguishes_unregistered_coverage_and_missing_files(tmp_path):
    color, ct = paired_images()
    Image.fromarray(color).save(tmp_path / "color.png")
    (ct + 1024).astype(">u2").tofile(tmp_path / "ct.u16be")
    manifest = {
        "baked": True, "subject": "male",
        "rgb": {"layers": [{"frame": frame, "file": "color.png"} for frame in range(5)]},
        "density": {"width": 512, "height": 304, "layers": [{"frame": frame, "file": "missing.u16be" if frame == 2 else "ct.u16be"} for frame in range(5)]},
    }
    profile = {"subject": "male", "coverage": {"color_depth": [.25, .75]},
               "depth_knots": [{"color_depth": 0, "ct_frame": 0}, {"color_depth": 1, "ct_frame": 4}],
               "spatial_knots": [{"color_depth": 0, "inverse_uv": [1, 0, 0, 0, 1, 0]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "alignment.json").write_text(json.dumps(profile))
    # Bad saved transforms must not influence a base-profile audit.
    (tmp_path / "manual-alignment.json").write_text(json.dumps({"anchors": [{"color_frame": 1, "scale_x": 4}]}))
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    report = audit.scan(tmp_path, "male", stride=1, progress=lambda message: None)
    assert report["samples"][0]["status"] == "outside_registered_coverage"
    assert report["samples"][0]["ct_frame"] is None
    assert report["samples"][1]["metrics"]["iou"] > .99
    assert report["samples"][2]["status"] == "missing_density_source"
    assert report["summary"]["outside_coverage_frames"] == 2
    assert report["summary"]["missing_source_samples"] == 1
    assert report["summary"]["qc_proxies_only"] is True
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    json.dumps(report, allow_nan=False)


def test_nearby_problem_samples_deduplicate_and_ranking_uses_peak():
    def sample(frame, severity, status="evaluated"):
        return {"color_frame": frame, "color_depth": frame / 100, "ct_frame": frame,
                "severity": severity, "status": status, "reasons": ["test"], "metrics": {}, "issue_codes": []}
    issues = audit.group_issues([sample(0, 40), sample(10, 60), sample(20, 0), sample(30, 80), sample(40, 90, "outside_registered_coverage")], "male", 15)
    assert len(issues) == 3
    assert [issue["severity"] for issue in issues] == [90, 80, 60]
    assert issues[-1]["start_frame"] == 0 and issues[-1]["end_frame"] == 10
    assert issues[-1]["color_frame"] == 10


def test_neighbor_transform_change_is_reported():
    color, ct = paired_images()
    a = audit.measure_pair(color, ct, [1, 0, 0, 0, 1, 0])
    b = audit.measure_pair(color, ct, [1.3, 0, 0, 0, 1, 0])
    samples = [{"color_frame": frame, "status": "evaluated", "metrics": metrics} for frame, metrics in ((0, a), (12, b))]
    audit.add_neighbor_warnings(samples)
    assert "transform_change" in samples[1]["issue_codes"]
