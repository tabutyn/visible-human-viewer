import sys
from pathlib import Path

import numpy as np
import pytest
import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))
import optimize_alignment as optimizer


def test_six_parameter_uv_round_trip():
    truth = optimizer.SpatialParameters(17.5, -9.25, 4.5, 1.23, .82)
    matrix = optimizer.parameters_to_inverse_uv(truth, "flip_y")
    found = optimizer.inverse_uv_to_parameters(matrix, "flip_y")
    assert found.x_position == pytest.approx(truth.x_position)
    assert found.y_position == pytest.approx(truth.y_position)
    assert found.rotation_deg == pytest.approx(truth.rotation_deg)
    assert found.x_scale == pytest.approx(truth.x_scale)
    assert found.y_scale == pytest.approx(truth.y_scale)


def test_manual_parameter_curve_initializes_between_reviewed_anchors():
    reviewed = {
        100: {"parameters": {"x_position": -5, "y_position": -25, "rotation_deg": -1,
                              "x_scale": .7, "y_scale": 1.1}},
        200: {"parameters": {"x_position": 5, "y_position": -15, "rotation_deg": 1,
                              "x_scale": 1.4, "y_scale": 1.21}},
    }
    found = optimizer.interpolate_reviewed_parameters(reviewed, 150)
    assert found.x_position == pytest.approx(0)
    assert found.y_position == pytest.approx(-20)
    assert found.rotation_deg == pytest.approx(0)
    assert found.x_scale == pytest.approx(np.sqrt(.7 * 1.4))
    assert found.y_scale == pytest.approx(np.sqrt(1.1 * 1.21))
    assert optimizer.interpolate_reviewed_parameters(reviewed, 50) is None


def test_local_optimizer_recovers_translation_rotation_and_anisotropic_scale():
    height, width = 76, 128
    moving_mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(moving_mask, (58, 39), (26, 18), 13, 0, 360, 1, -1)
    cv2.circle(moving_mask, (48, 34), 5, 0, -1)
    cv2.rectangle(moving_mask, (70, 29), (78, 34), 1, -1)
    truth = optimizer.SpatialParameters(5, -3, 4, 1.13, .86)
    fixed_mask = cv2.warpAffine(moving_mask, optimizer.display_forward(truth, width, height)[:2].astype(np.float32), (width, height))

    def planes(mask):
        edges = cv2.Canny(mask * 255, 20, 80)
        distance = np.exp(-cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3) / 4).astype(np.float32)
        return mask, distance, distance

    found, metrics = optimizer.optimize_pair(planes(fixed_mask), planes(moving_mask), optimizer.SpatialParameters())
    assert metrics["accepted"] and metrics["after"] < metrics["before"] * .2
    assert found.x_position == pytest.approx(truth.x_position, abs=.3)
    assert found.y_position == pytest.approx(truth.y_position, abs=.3)
    assert found.rotation_deg == pytest.approx(truth.rotation_deg, abs=.4)
    assert found.x_scale == pytest.approx(truth.x_scale, abs=.03)
    assert found.y_scale == pytest.approx(truth.y_scale, abs=.03)


def test_local_optimizer_cannot_run_away_from_valid_initial_transform():
    height, width = 76, 128
    fixed_mask = np.zeros((height, width), np.uint8)
    moving_mask = np.zeros((height, width), np.uint8)
    cv2.circle(fixed_mask, (110, 38), 8, 1, -1)
    cv2.circle(moving_mask, (10, 38), 8, 1, -1)

    def planes(mask):
        edges = cv2.Canny(mask * 255, 20, 80)
        distance = np.exp(-cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3) / 4).astype(np.float32)
        return mask, distance, distance

    found, _ = optimizer.optimize_pair(planes(fixed_mask), planes(moving_mask), optimizer.SpatialParameters())
    assert abs(found.x_position) <= 40
    assert abs(found.y_position) <= 32
    assert abs(found.rotation_deg) <= 6


def test_singular_and_sheared_transforms_are_rejected():
    with pytest.raises(ValueError, match="singular"):
        optimizer.inverse_uv_to_parameters([1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="shear"):
        optimizer.inverse_uv_to_parameters([1, .8, 0, 0, -1, 1])


def test_monotonic_z_preserves_hard_origin_anchor():
    frames = np.arange(30)
    ct_frames = np.arange(5, 35)
    # A unique, smoothly changing synthetic anatomy descriptor.
    values = np.stack([np.sin(frames / 6 + column) for column in np.linspace(0, 1, 18)], axis=1).astype(np.float32)
    ct_values = np.stack([np.sin((ct_frames - 7) / 6 + column) for column in np.linspace(0, 1, 18)], axis=1).astype(np.float32)
    knots, _ = optimizer.solve_monotonic_z(frames, ct_frames, values, ct_values, [(0, 7)], sample_count=20, band_frames=8)
    assert knots[0]["color_frame"] == 0
    assert knots[0]["ct_frame"] == 7
    assert all(knots[index]["ct_frame"] < knots[index + 1]["ct_frame"] for index in range(len(knots) - 1))


def test_reversed_z_anchors_are_rejected():
    values = np.zeros((10, 18), np.float32)
    with pytest.raises(ValueError, match="strictly increasing"):
        optimizer.solve_monotonic_z(np.arange(10), np.arange(10), values, values, [(0, 7), (8, 3)])


def test_spatial_smoothing_does_not_bridge_boundaries():
    rows = []
    for frame, x in ((0, 0), (10, 1), (20, 2), (30, 40), (40, 41), (50, 42)):
        rows.append({"color_frame": frame, "color_depth": frame / 50, "confidence": 1,
                     "x_position": x, "y_position": 0, "rotation_deg": 0, "x_scale": 1, "y_scale": 1})
    result = optimizer.smooth_spatial(rows, 50, [30])
    left = next(item for item in result if item["color_frame"] == 20)
    right = next(item for item in result if item["color_frame"] == 30)
    assert left["segment"] != right["segment"]
    assert left["x_position"] < 10 and right["x_position"] > 30


def test_spatial_smoothing_preserves_reviewed_parameters_exactly():
    rows = []
    for frame, x in ((0, 0), (10, 1), (20, 22.5), (30, 3), (40, 4)):
        rows.append({"color_frame": frame, "color_depth": frame / 40, "confidence": 1,
                     "x_position": x, "y_position": -7 if frame == 20 else 0,
                     "rotation_deg": 2 if frame == 20 else 0, "x_scale": 1.2 if frame == 20 else 1,
                     "y_scale": .8 if frame == 20 else 1, "reviewed_constraint": frame == 20})
    reviewed = next(item for item in optimizer.smooth_spatial(rows, 40, []) if item["color_frame"] == 20)
    assert reviewed["x_position"] == 22.5
    assert reviewed["y_position"] == -7
    assert reviewed["rotation_deg"] == 2
    assert reviewed["x_scale"] == 1.2
    assert reviewed["y_scale"] == .8


def test_failed_regions_extend_completed_queue_to_at_most_twelve():
    queue = [{"color_frame": frame, "status": "confirmed"} for frame in range(10)]
    regional = [
        {"region": "chest", "accepted": False, "baseline_95th": .2, "candidate_95th": .21,
         "samples": [{"color_frame": 40, "baseline": .1, "candidate": .2}]},
        {"region": "abdomen", "accepted": False, "baseline_95th": .1, "candidate_95th": .13,
         "samples": [{"color_frame": 60, "baseline": .1, "candidate": .18}]},
        {"region": "knee", "accepted": False, "baseline_95th": .3, "candidate_95th": .31,
         "samples": [{"color_frame": 80, "baseline": .1, "candidate": .2}]},
    ]
    result = optimizer.extend_review_queue_for_failed_regions(queue, regional,
        [{"color_depth": 0, "ct_frame": 7}, {"color_depth": 1, "ct_frame": 107}], list(range(101)), list(range(5, 108)))
    assert len(result) == 12
    assert [item["color_frame"] for item in result[-2:]] == [60, 40]
    assert all(item["status"] == "pending" for item in result[-2:])


def test_control_review_cohort_is_capped_prioritized_and_diverse():
    knots = [{"color_frame": frame, "confidence": .5} for frame in range(0, 1001, 100)]
    z_queue = [{"color_frame": 450, "status": "pending", "confidence": 0}]
    regional = [{"region": "knee", "accepted": False,
                 "samples": [{"color_frame": 700, "baseline": .1, "candidate": .3}]}]
    result = optimizer.select_control_review_frames(
        knots, z_queue, regional, {0, 100}, 1000, limit=4)
    assert len(result) == 4
    assert 450 in result  # unresolved Z is first priority
    assert 700 in result  # failed held-out anatomy is represented
    assert max(result) == 1000  # remaining reviews cover large unsupported gaps


def test_control_review_cohort_stays_fixed_across_rebuilds():
    knots = [{"color_frame": frame, "confidence": .5} for frame in range(0, 501, 100)]
    result = optimizer.select_control_review_frames(
        knots, [], [], {0, 100, 200}, 500, limit=10,
        previous_frames=[100, 300, 500])
    assert result == [100, 300, 500]


def test_validation_frames_avoid_reviewed_anchors_and_cover_region():
    frames = np.arange(101)
    found = optimizer.validation_frame_indices(frames, 100, 0, 1, {20, 50, 80})
    assert len(found) == 3
    assert not set(found) & {20, 50, 80}
    assert found[0] < found[1] < found[2]


def test_regional_acceptance_uses_paired_median_improvement():
    # Two of three paired samples improve, even though the median absolute
    # candidate score is slightly higher than the baseline median.
    metrics = optimizer.regional_validation_metrics(
        [.1498, .1330, .4470], [.1619, .1282, .4389])
    assert metrics["candidate_median"] > metrics["baseline_median"]
    assert metrics["median_improvement"] > 0
    assert metrics["accepted"]
