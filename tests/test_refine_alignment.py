import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import refine_alignment as refine
from optimize_alignment import SpatialParameters, parameters_to_inverse_uv


def inverse(parameters=SpatialParameters(), orientation="identity"):
    return parameters_to_inverse_uv(parameters, orientation, width=513, height=305)


def phantom():
    ys, xs = np.mgrid[:304, :512].astype(np.float32)
    mask = ((xs - 256) / 105) ** 2 + ((ys - 152) / 112) ** 2 < 1
    images = []
    for z in range(9):
        hu = 40 + 75 * np.sin(xs / 13 + ys / 27) + 40 * np.sin(ys / 12)
        for x, y, radius, drift in [(217, 108, 19, 1.5), (278, 169, 25, -1.4), (248, 207, 13, .8), (301, 115, 13, -.6)]:
            distance = ((xs - x - drift * z) ** 2 + (ys - y + drift * z) ** 2) ** .5
            hu += 900 / (1 + np.exp(np.clip((distance - radius) / 1.4, -30, 30)))
        hu[~mask] = -1024
        images.append(hu.astype(np.float32))
    return images


def make_pair(z=4.35, parameters=SpatialParameters(5, -4, 1.3, 1.035, .97)):
    images = phantom()
    left = int(np.floor(z)); right = min(8, left + 1)
    hu = images[left] * (right - z) + images[right] * (z - left) if left != right else images[left]
    matrix = refine.pixel_matrix(inverse(parameters), 512, 304)
    hu = cv2.warpAffine(hu, matrix, (512, 304), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=-1024)
    tissue = np.clip((hu + 150) / 1200, 0, 1)
    rgb = np.stack([90 + tissue * 160, 45 + tissue * 175, 25 + tissue * 180], axis=-1).astype(np.uint8)
    rgb[hu < -400] = [10, 35, 75]
    return rgb, images


def manifest(frames=tuple(range(20)), boundaries=(), subject="male"):
    return {"subject": subject, "density": {"layers": [{"frame": z} for z in frames]}, "boundaries": boundaries}


def test_z_neighborhood_preserves_hard_anchor_and_confirmed_neighbors():
    assert refine.local_z_bounds(manifest(), 7, 0) == (7, 7)
    with pytest.raises(ValueError, match="hard"):
        refine.local_z_bounds(manifest(), 8, 0)
    anchors = [{"color_frame": 10, "ct_frame": 9, "z_confirmed": True},
               {"color_frame": 30, "ct_frame": 14, "z_confirmed": True}]
    low, high = refine.local_z_bounds(manifest(), 12, 20, anchors)
    assert 9 < low < 9.001 and 13.999 < high < 14
    with pytest.raises(ValueError, match="conflicts"):
        refine.local_z_bounds(manifest(), 8, 20, anchors)
    with pytest.raises(ValueError, match="reversed"):
        refine.local_z_bounds(manifest(), 12, 20, supplied=[14, 10])


def test_cannot_interpolate_over_gap_or_fov_boundary():
    gaps = manifest((0, 1, 2, 8, 9, 10), subject="female")
    assert refine.local_z_bounds(gaps, 2, 20) == (0, 2)
    assert refine.local_z_bounds(gaps, 8, 20) == (8, 10)
    with pytest.raises(ValueError, match="boundary|gap"):
        refine.local_z_bounds(gaps, 3.5, 20)
    fov = manifest(boundaries=[{"modality": "density", "left": 8, "right": 9, "reason": {"fov_change": True}}])
    assert refine.local_z_bounds(fov, 8, 20) == (7.0001, 8)
    with pytest.raises(ValueError, match="boundary"):
        refine.local_z_bounds(fov, 8.5, 20)


def test_flat_tissue_is_rejected_without_changing_exact_editor_state():
    rgb = np.zeros((304, 512, 3), np.uint8)
    cv2.ellipse(rgb, (256, 152), (90, 100), 0, 0, 360, (160, 90, 50), -1)
    hu = np.full((304, 512), -1024, np.float32)
    cv2.ellipse(hu, (265, 142), (90, 100), 0, 0, 360, 50, -1)
    original = inverse(SpatialParameters(1.23, -2.41, .47, 1.03, 1.02))
    request = {"ct_frame": 4.25, "inverse_uv": original[:], "orientation": "identity"}
    result = refine.refine_pair(rgb, [4, 5], [hu, hu], request)
    assert result["accepted"] is False
    assert result["inverse_uv"] == original and result["ct_frame"] == 4.25
    assert result["metrics"]["after"] == result["metrics"]["before"]
    assert request["inverse_uv"] == original
    json.dumps(result, allow_nan=False)


def test_singular_and_shear_transforms_cannot_enter_optimizer():
    rgb, images = make_pair()
    for values in ([0, 0, 0, 0, 0, 0], [1, .2, 0, 0, 1, 0]):
        with pytest.raises(ValueError, match="singular|shear"):
            refine.refine_pair(rgb, range(9), images, {"ct_frame": 4, "inverse_uv": values, "orientation": "identity"})


def test_fractional_z_and_spatial_parameters_recover_jointly():
    expected = SpatialParameters(5, -4, 1.3, 1.035, .97)
    rgb, images = make_pair(parameters=expected)
    result = refine.refine_pair(rgb, range(9), images, {"ct_frame": 3.1, "inverse_uv": inverse(), "orientation": "identity"})
    assert result["accepted"], result
    parameters = result["parameters"]
    assert abs(result["ct_frame"] - 4.35) < .8, parameters
    assert abs(parameters["x_position"] - expected.x_position) < 1.5, parameters
    assert abs(parameters["y_position"] - expected.y_position) < 1.5, parameters
    assert abs(parameters["rotation_deg"] - expected.rotation_deg) < 1, parameters
    assert abs(parameters["x_scale"] - expected.x_scale) < .02, parameters
    assert abs(parameters["y_scale"] - expected.y_scale) < .02, parameters
    assert result["metrics"]["after"]["objective"] < result["metrics"]["before"]["objective"]
    assert result["metrics"]["after"]["rgb_hu_nmi"] > result["metrics"]["before"]["rgb_hu_nmi"]


def test_failed_optimizer_cannot_mutate_input_or_return_a_worse_result(monkeypatch):
    rgb, images = make_pair()
    values = inverse()

    def explode(fun, vector, **kwargs):
        vector[:] = 1e6
        raise ValueError("mutated failing temporary")

    monkeypatch.setattr(refine, "minimize", explode)
    result = refine.refine_pair(rgb, range(9), images, {"ct_frame": 3.1, "inverse_uv": values, "orientation": "identity"})
    assert not result["accepted"]
    assert result["inverse_uv"] == values and result["ct_frame"] == 3.1
    assert result["metrics"]["after"] == result["metrics"]["before"]


def test_landmarks_freeze_ct_plane_and_are_validated(monkeypatch):
    rgb, images = make_pair()
    request = {"ct_frame": 4.25, "inverse_uv": inverse(), "orientation": "identity",
               "landmarks": [{"color": [.5, .5], "density": [.5, .5]}]}
    assert refine.local_z_bounds(manifest(), 8, 20, landmarks=request["landmarks"]) == (8, 8)
    monkeypatch.setattr(refine, "minimize", lambda *a, **k: (_ for _ in ()).throw(ValueError("skip")))
    result = refine.refine_pair(rgb, range(9), images, request)
    assert result["ct_frame"] == 4.25
    assert result["bounds"]["ct_frame"] == [4.25, 4.25]
    assert any("landmark" in warning for warning in result["warnings"])
    request["landmarks"][0]["density"] = [float("nan"), .5]
    with pytest.raises(ValueError, match="Landmark"):
        refine.refine_pair(rgb, range(9), images, request)


def test_stale_reviewed_anchor_cannot_restrict_new_geometry(tmp_path, monkeypatch):
    data = manifest(subject="female")
    data.update(baked=True, rgb={"layers": [{"frame": 20, "file": "color.png"}]})
    data["density"].update(width=512, height=304)
    for row in data["density"]["layers"]:
        row["file"] = f"ct-{row['frame']}.u16be"
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    (tmp_path / "alignment.json").write_text("{}")
    current_inputs = {"manifest_sha256": refine.sha256(tmp_path / "manifest.json"),
                      "alignment_sha256": refine.sha256(tmp_path / "alignment.json")}
    (tmp_path / "reviewed-alignment.json").write_text(json.dumps({"anchors": [
        {"color_frame": 10, "ct_frame": 17, "inputs": {**current_inputs, "manifest_sha256": "old-geometry"}},
        {"color_frame": 30, "ct_frame": 15, "inputs": current_inputs}]}))
    monkeypatch.setattr(refine, "read_baked_rgb", lambda *args: np.zeros((304, 512, 3), np.uint8))
    monkeypatch.setattr(refine, "read_baked_ct", lambda *args: np.zeros((304, 512), np.float32))
    monkeypatch.setattr(refine, "refine_pair", lambda *args: {"bounds": args[-1]})
    result = refine.refine(tmp_path, 20, {"ct_frame": 12, "inputs": current_inputs})
    assert result["bounds"] == (4, 14.9999)
    with pytest.raises(ValueError, match="inputs changed"):
        refine.refine(tmp_path, 20, {"ct_frame": 12, "inputs": {**current_inputs, "alignment_sha256": "old"}})
