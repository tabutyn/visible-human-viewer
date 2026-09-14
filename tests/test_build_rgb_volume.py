import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import build_rgb_volume


def test_builds_registered_planar_rgb_on_ct_grid(tmp_path):
    subject = tmp_path / "male"
    rgb = subject / "rgb"
    rgb.mkdir(parents=True)
    first = np.zeros((4, 4, 3), dtype=np.uint8)
    first[..., 2] = 255
    second = np.zeros((4, 4, 3), dtype=np.uint8)
    second[..., 1] = 255
    assert cv2.imwrite(str(rgb / "000000.png"), first)
    assert cv2.imwrite(str(rgb / "000001.png"), second)
    (subject / "manifest.json").write_text(json.dumps({"rgb": {"layers": [{"frame": 0, "file": "rgb/000000.png"}, {"frame": 1, "file": "rgb/000001.png"}]}}))
    (subject / "alignment.json").write_text(json.dumps({"orientation": "flip_y"}))
    identity = [1, 0, 0, 0, 1, 0]
    (subject / "alignment-candidate-v2.json").write_text(json.dumps({"parameter_knots": [
        {"color_frame": 0, "color_depth": 0, "z_position": 7, "inverse_uv": identity},
        {"color_frame": 1, "color_depth": 1, "z_position": 8, "inverse_uv": identity},
    ]}))
    ct = subject / "volume-v1"
    ct.mkdir()
    (ct / "manifest.json").write_text(json.dumps({"subject": "male", "dimensions": [4, 4, 2], "spacing_mm": [1, 1, 1], "slice_frames": [7, 8], "brick_size": 32,
        "levels": [
            {"level": 0, "factor": 1, "dimensions": [4, 4, 2]},
            {"level": 1, "factor": 2, "dimensions": [2, 2, 1]},
        ]}))

    manifest = build_rgb_volume.build_rgb_volume(subject)
    assert manifest["format"] == "rgb8-planar"
    assert manifest["dimensions"] == [4, 4, 2]
    assert manifest["levels"][0]["bytes"] == 4 * 4 * 2 * 3
    brick = manifest["levels"][0]["bricks"][0]
    payload = np.frombuffer((subject / "rgb-volume-v1" / brick["file"]).read_bytes(), dtype=np.uint8)
    channel_size = 32 * 32 * 2
    red, green, blue = payload[:channel_size], payload[channel_size:2 * channel_size], payload[2 * channel_size:]
    assert red.reshape(2, 32, 32)[0, :4, :4].min() == 255
    assert green.reshape(2, 32, 32)[1, :4, :4].min() == 255
    assert blue.max() == 0
    assert [level["dimensions"] for level in manifest["levels"]] == [[4, 4, 2], [2, 2, 1]]
    coarse = manifest["levels"][1]["bricks"][0]
    coarse_payload = np.frombuffer((subject / "rgb-volume-v1" / coarse["file"]).read_bytes(), dtype=np.uint8)
    coarse_channel_size = 32 * 32
    coarse_red = coarse_payload[:coarse_channel_size].reshape(32, 32)[:2, :2]
    coarse_green = coarse_payload[coarse_channel_size : 2 * coarse_channel_size].reshape(32, 32)[:2, :2]
    assert np.all(coarse_red == 128)
    assert np.all(coarse_green == 128)
