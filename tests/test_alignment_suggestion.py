import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import suggest_alignment as suggestion


def test_density_selection_never_bridges_missing_gap():
    layers = [{"frame": x} for x in (0, 3, 6, 18, 21)]
    assert suggestion.choose_ct_layer(layers, 4.7)["frame"] == 6
    assert suggestion.choose_ct_layer(layers, 18)["frame"] == 18
    with pytest.raises(ValueError, match="gap"):
        suggestion.choose_ct_layer(layers, 10)
    with pytest.raises(ValueError, match="outside"):
        suggestion.choose_ct_layer(layers, -1)


@pytest.mark.parametrize("radii", [(33, 40), (26, 29)])
def test_annotation_filter_preserves_paired_legs_and_removes_detached_note(radii):
    color = np.full((304, 512, 3), [20, 50, 100], np.uint8)
    cv2.ellipse(color, (205, 90), radii, 0, 0, 360, (160, 90, 50), -1)
    cv2.ellipse(color, (300, 98), radii, 0, 0, 360, (160, 90, 50), -1)
    cv2.rectangle(color, (301, 221), (340, 251), (205, 205, 100), -1)
    mask, warnings = suggestion.color_mask(color, "female")
    assert mask[90, 205] and mask[98, 300]
    assert not mask[235, 320]
    assert any("annotation" in item for item in warnings)
    # A similarly placed nonyellow component must not be removed by this rule.
    color[221:252, 301:341] = [160, 90, 50]
    mask, _ = suggestion.color_mask(color, "female")
    assert mask[235, 320]


def test_lower_body_wedge_is_removed_before_morphology_can_attach_it():
    hu = np.full((304, 512), -1024., np.float32)
    cv2.ellipse(hu, (150, 165), (40, 50), 0, 0, 360, 30, -1)
    cv2.ellipse(hu, (355, 165), (40, 50), 0, 0, 360, 30, -1)
    cv2.circle(hu, (150, 165), 10, 1000, -1)
    cv2.circle(hu, (355, 165), 10, 1000, -1)
    cv2.rectangle(hu, (120, 60), (190, 111), 320, -1)
    mask, warnings = suggestion.density_mask(hu, "male", .66)
    assert mask[165, 150] and mask[165, 355]
    assert not mask[80, 155]
    assert any("Excluded detached" in item for item in warnings)
    # Cortical thresholds must not remove ordinary soft tissue in the chest.
    hu[hu > 700] = 30
    chest_mask, _ = suggestion.density_mask(hu, "male", .25)
    assert chest_mask[165, 150]


def test_attached_support_is_reported_not_claimed_removed():
    hu = np.full((304, 512), -1024., np.float32)
    cv2.rectangle(hu, (160, 30), (240, 200), 40, -1)
    cv2.circle(hu, (200, 165), 10, 1000, -1)
    _, warnings = suggestion.density_mask(hu, "male", .66)
    assert any("attached support" in item for item in warnings)


@pytest.mark.parametrize("throws", [True, False])
def test_ecc_mutation_on_failure_or_rejection_preserves_outline_fit(monkeypatch, throws):
    color = np.zeros((304, 512, 3), np.uint8)
    cv2.ellipse(color, (250, 150), (70, 90), 0, 0, 360, (150, 90, 50), -1)
    hu = np.full((304, 512), -1024., np.float32)
    cv2.ellipse(hu, (275, 135), (90, 70), 0, 0, 360, 40, -1)

    def mutate_then_fail(template, image, scratch, *args, **kwargs):
        scratch[:] = 10000
        if throws:
            raise cv2.error("deliberate optimizer failure")
        return 1., scratch

    monkeypatch.setattr(cv2, "findTransformECC", mutate_then_fail)
    result = suggestion.suggest_pair(color, hu, "male", .3, [1, 0, 0, 0, 1, 0])
    assert result["metrics"]["after"]["iou"] > .93
    assert result["metrics"]["after"]["iou"] > result["metrics"]["before"]["iou"]
    assert max(abs(x) for x in result["inverse_uv"]) < 3
    assert "proxies" in result["warnings"][0]
    json.dumps(result, allow_nan=False)


def test_empty_mask_rejected():
    with pytest.raises(ValueError, match="Too little"):
        suggestion.suggest_pair(np.zeros((304, 512, 3), np.uint8),
                                np.full((304, 512), -1024, np.float32),
                                "male", .3, [1, 0, 0, 0, 1, 0])


def test_layer_cannot_escape_processed_directory(tmp_path):
    with pytest.raises(ValueError, match="leaves"):
        suggestion.layer_slice(tmp_path, {"file": "../outside.png", "frame": 1}, "rgb")
