import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import build_volume


def _male_slice(root: Path, number: int, value: int) -> None:
    directory = root / "Male" / "Radiological" / "frozenCT" / "png"
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((512, 512), value, dtype=np.uint16)
    assert cv2.imwrite(str(directory / f"cvm{number:04d}f.png"), image)


def test_volume_size_and_analysis_to_square_affine():
    assert build_volume.volume_bytes((512, 512, 16384)) == 8 * 1024**3
    matrix = build_volume.square_affine(build_volume.Similarity(tx=4, ty=3))
    assert matrix[0, 2] == 4
    assert matrix[1, 2] == 3 * 512 / 304


def test_builds_little_endian_bricked_pyramid(tmp_path):
    root = tmp_path / "Visible-Human-Project"
    for index, value in enumerate((1024, 1124, 1224, 1324, 1424), start=1006):
        _male_slice(root, index, value)
    subject_output = root / "Processed" / "v1" / "male"
    subject_output.mkdir(parents=True)
    (subject_output / "corrections.json").write_text(json.dumps({
        "version": 1,
        "transforms": {"density": {str(index): {"tx": 0, "ty": 0, "scale": 1, "rotation_deg": 0} for index in range(5, 10)}},
    }))

    manifest = build_volume.build_volume(root, subject_output, "male", brick_size=32, levels=2)
    assert manifest["dimensions"] == [512, 512, 5]
    assert manifest["format"] == "u16le-hu-plus-1024"
    assert manifest["levels"][1]["dimensions"] == [256, 256, 3]
    assert sum(manifest["histogram"]["bins"]) == 512 * 512 * 5
    assert manifest["slice_frames"] == [5, 6, 7, 8, 9]

    brick = manifest["levels"][0]["bricks"][0]
    payload = (subject_output / "volume-v1" / brick["file"]).read_bytes()
    values = np.frombuffer(payload, dtype="<u2").reshape(5, 32, 32)
    assert values[:, 0, 0].tolist() == [1024, 1124, 1224, 1324, 1424]
    assert brick["min_hu"] == 0 and brick["max_hu"] == 400
    on_disk = json.loads((subject_output / "volume-v1" / "manifest.json").read_text())
    assert on_disk["inputs"]["ct_sha256"] == manifest["inputs"]["ct_sha256"]
