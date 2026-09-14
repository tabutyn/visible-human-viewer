import sys
from pathlib import Path
import json
import struct

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import preprocess


def test_binary_ge_header_geometry_fallback(tmp_path):
    data = bytearray(preprocess.GE_HEADER_BYTES)
    image_header = 2394
    struct.pack_into(">I", data, 148, image_header)
    struct.pack_into(">hh", data, image_header + 30, 512, 512)
    struct.pack_into(">ffff", data, image_header + 42, 440.0, 440.0, .859375, .859375)
    source = tmp_path / "c_vf1001.fre"
    source.write_bytes(data)
    parsed = preprocess.parse_ge_header(source)
    assert parsed["dimension_x"] == parsed["dimension_y"] == 512
    assert parsed["pixel_size_x"] == parsed["pixel_size_y"] == pytest.approx(.859375)
    assert parsed["fov_x"] == parsed["fov_y"] == pytest.approx(440.0)
import register_modalities as modalities


def test_filename_depths():
    assert preprocess.parse_rgb_filename("a_vm1001.raw") == ("m", 0)
    assert preprocess.parse_rgb_filename("avf1002c.raw") == ("f", 5)
    assert preprocess.parse_ct_filename("c_vf1011.fre", "female") == 30
    assert preprocess.parse_ct_filename("c_vm1011.fre", "male") == 10


def test_gap_detection_uses_depth():
    slices = [preprocess.Slice("", "", depth, "rgb") for depth in (0, 1, 2, 10)]
    assert any(item["gap"] for item in preprocess.robust_boundaries(slices, [1, 1, 1, 1], 1))


def test_synthetic_alignment():
    assert preprocess.synthetic_transform_test()["pass"]


def test_frozen_ct_png_and_separate_header(tmp_path):
    root = tmp_path / "Visible-Human-Project"
    png = root / "Male" / "Radiological" / "frozenCT" / "png"
    headers = root / "Male" / "Radiological" / "frozenCT" / "headers"
    png.mkdir(parents=True); headers.mkdir()
    image = preprocess.np.full((512, 512), 1200, dtype=preprocess.np.uint16)
    preprocess.cv2.imwrite(str(png / "cvm1006f.png"), image)
    # Real NLM text dumps put these fields after the 3,416-byte binary-header
    # length used by .fre files.  Keep padding here to prevent regression.
    (headers / "cvm1006f.txt").write_text("x" * 4000 + """\nExam number for this image...............: 32
Series number for this image.............: 4
Image dimension - X......................: 270
Image dimension - Y......................: 270
Image pixel size - X.....................: 0.527344
Image pixel size - Y.....................: 0.527344
Center R coord of plane image............: 0
Center A coord of plane image............: 30
Image location...........................: -20
""")
    found = preprocess.discover_ct(root, "male")
    assert len(found) == 1 and found[0].depth == 5
    assert found[0].exam == "32" and found[0].series == "4"
    assert found[0].pixel_size_x == .527344 and found[0].position_mm == -20
    assert found[0].center_r == 0 and found[0].center_a == 30
    assert preprocess.full_ct(found[0]).dtype == preprocess.np.uint16


def test_centered_similarity_round_trip_and_composition():
    center = (256.0, 152.0)
    first = preprocess.Similarity(3.0, -4.0, 1.02, 1.5)
    second = preprocess.Similarity(-2.0, 1.0, .99, -.5)
    matrix = preprocess.np.eye(3)
    matrix[:2] = preprocess.affine_from_similarity(first, center)
    recovered = preprocess.similarity_from_affine(matrix[:2], center)
    assert abs(recovered.tx - first.tx) < 1e-5
    assert abs(recovered.ty - first.ty) < 1e-5
    composed = preprocess.compose_similarity(first, second, center)
    actual = preprocess.np.eye(3)
    actual[:2] = preprocess.affine_from_similarity(composed, center)
    expected_a = preprocess.np.eye(3); expected_b = preprocess.np.eye(3)
    expected_a[:2] = preprocess.affine_from_similarity(first, center)
    expected_b[:2] = preprocess.affine_from_similarity(second, center)
    assert preprocess.np.allclose(actual, expected_a @ expected_b, atol=1e-5)
    inverse = preprocess.inverse_similarity(first, center)
    identity = preprocess.compose_similarity(first, inverse, center)
    assert abs(identity.tx) < 1e-5 and abs(identity.ty) < 1e-5
    assert abs(identity.scale - 1) < 1e-5 and abs(identity.rotation_deg) < 1e-5


def test_ct_metadata_uses_raster_axis_and_analysis_aspect():
    narrow = preprocess.Slice(
        "", "narrow", 0, "density", pixel_size_x=.527344,
        pixel_size_y=.527344, center_r=0, center_a=30,
    )
    wide = preprocess.Slice(
        "", "wide", 1, "density", pixel_size_x=.9375,
        pixel_size_y=.9375, center_r=0, center_a=0,
    )
    transform = preprocess.ct_metadata_transforms([narrow, wide])["0"]
    assert abs(transform.scale - (.9375 / .527344)) < 1e-6
    expected_ty = (30 / .527344) * (304 / 512)
    assert abs(transform.ty - expected_ty) < 1e-6


def test_planar_rgb_decodes_to_image(tmp_path):
    pixels = preprocess.RGB_SIZE[0] * preprocess.RGB_SIZE[1]
    path = tmp_path / "a_vm1001.raw"
    preprocess.np.concatenate([
        preprocess.np.full(pixels, 10, dtype=preprocess.np.uint8),
        preprocess.np.full(pixels, 20, dtype=preprocess.np.uint8),
        preprocess.np.full(pixels, 30, dtype=preprocess.np.uint8),
    ]).tofile(path)
    item = preprocess.Slice(str(path), path.name, 0, "rgb")
    image = preprocess.full_rgb(item)
    assert image.shape == (1216, 2048, 3)
    assert image[0, 0].tolist() == [10, 20, 30]
    assert preprocess.read_rgb(item).shape == (304, 512, 3)


def test_binary_ge_recon_annotation(tmp_path):
    path = tmp_path / "c_vf1314.fre"
    header = b"IMGF\x00CT Recon HSUC/370/2/318 @ loc -351.000 mm\x00"
    path.write_bytes(header + bytes(preprocess.GE_HEADER_BYTES - len(header)))
    parsed = preprocess.parse_ge_header(path)
    assert parsed["exam"] == "370" and parsed["series"] == "2"
    assert parsed["section"] == 318 and parsed["position_mm"] == -351


def test_full_ge_ct_converts_to_native_endian(tmp_path):
    path = tmp_path / "c_vf1001.fre"
    payload = preprocess.np.arange(512 * 512, dtype=preprocess.np.uint16) % 4096
    with path.open("wb") as stream:
        stream.write(bytes(preprocess.GE_HEADER_BYTES))
        stream.write(payload.astype(">u2").tobytes())
    item = preprocess.Slice(str(path), path.name, 0, "density")
    decoded = preprocess.full_ct(item)
    assert decoded.dtype.isnative
    assert int(decoded[0, 1]) == 1 and int(decoded.max()) == 4095


def test_json_output_is_strict_for_non_finite_qc_values():
    encoded = preprocess.json_text({"finite": 1.0, "missing": float("inf")})
    assert "Infinity" not in encoded
    assert json.loads(encoded) == {"finite": 1.0, "missing": None}


def test_existing_baked_layers_only_returns_completed_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"baked": False, "rgb": {"layers": []}, "density": {"layers": []}}))
    assert preprocess.existing_baked_layers(tmp_path) is None
    manifest.write_text(json.dumps({
        "baked": True,
        "rgb": {"layers": [{"frame": 0, "file": "rgb/000000.png"}]},
        "density": {"layers": [{"frame": 0, "file": "density/000000.u16be"}]},
    }))
    rgb, density = preprocess.existing_baked_layers(tmp_path)
    assert rgb[0]["file"].endswith(".png")
    assert density[0]["file"].endswith(".u16be")


def _synthetic_modalities_image():
    image = preprocess.np.full((304, 512), -1000, dtype=preprocess.np.float32)
    preprocess.cv2.ellipse(image, (245, 158), (118, 130), 0, 0, 360, 90, -1)
    preprocess.cv2.circle(image, (195, 122), 28, 600, -1)
    preprocess.cv2.rectangle(image, (285, 195), (330, 230), 400, -1)
    color = preprocess.np.repeat(preprocess.np.clip((image + 1000) / 8, 0, 220).astype(preprocess.np.uint8)[..., None], 3, axis=2)
    # Deliberate blue resin/label areas outside anatomy must not become mask.
    color[:35, :90] = [10, 30, 240]
    return color, image


def test_modality_orientation_and_anatomy_masks():
    color, ct = _synthetic_modalities_image()
    flipped = preprocess.cv2.flip(ct, 1)
    solution = modalities.choose_orientation(color, flipped)
    assert solution.orientation == "flip_x"
    assert modalities.color_anatomy_mask(color)[10, 10] == 0
    assert modalities.ct_anatomy_mask(ct)[158, 245] == 1


def test_modality_affine_recovers_anisotropic_scale_and_translation():
    color, ct = _synthetic_modalities_image()
    forward = preprocess.np.array([[1.08, 0.0, 13.0], [0.0, .92, -7.0]], dtype=preprocess.np.float32)
    moving = preprocess.cv2.warpAffine(ct, forward, (512, 304), borderValue=-1000)
    solution = modalities.solve_affine(color, moving)
    # ECC's WARP_INVERSE_MAP convention stores the sampled CT transform in
    # the same forward coefficient convention supplied to warpAffine.
    expected = forward
    recovered = preprocess.np.asarray(solution.matrix)
    assert preprocess.np.allclose(recovered[:, :2], expected[:, :2], atol=.08)
    assert preprocess.np.allclose(recovered[:, 2], expected[:, 2], atol=2.0)
    assert abs(solution.scale_x - solution.scale_y) > .08


def test_modality_affine_rejection_preserves_initial_transform(monkeypatch):
    color, ct = _synthetic_modalities_image()
    moving = preprocess.cv2.warpAffine(
        ct,
        preprocess.np.array([[1.08, 0.0, 13.0], [0.0, .92, -7.0]], dtype=preprocess.np.float32),
        (512, 304),
        borderValue=-1000,
    )
    fixed_mask = modalities.color_anatomy_mask(color)
    moving_mask = modalities.ct_anatomy_mask(moving)
    initial = modalities._bounds_transform(fixed_mask, moving_mask)

    def corrupt_in_place(_fixed, _moving, candidate, *_args, **_kwargs):
        candidate[:] = [[9.0, 0.0, -2000.0], [0.0, 1.0, 0.0]]
        return 1.0, candidate

    monkeypatch.setattr(modalities.cv2, "findTransformECC", corrupt_in_place)
    solution = modalities.solve_affine(color, moving)
    assert preprocess.np.allclose(solution.matrix, initial)


def test_modality_affine_rejects_plausible_but_worse_candidate(monkeypatch):
    color, ct = _synthetic_modalities_image()
    moving = preprocess.cv2.warpAffine(
        ct,
        preprocess.np.array([[1.08, 0.0, 13.0], [0.0, .92, -7.0]], dtype=preprocess.np.float32),
        (512, 304),
        borderValue=-1000,
    )
    fixed_mask = modalities.color_anatomy_mask(color)
    moving_mask = modalities.ct_anatomy_mask(moving)
    initial = modalities._bounds_transform(fixed_mask, moving_mask)

    def move_off_anatomy(_fixed, _moving, candidate, *_args, **_kwargs):
        candidate[:, :] = initial
        candidate[:, 2] += (180.0, 90.0)
        return 1.0, candidate

    monkeypatch.setattr(modalities.cv2, "findTransformECC", move_off_anatomy)
    solution = modalities.solve_affine(color, moving)
    assert preprocess.np.allclose(solution.matrix, initial)


def test_internal_landmark_distance_improves_with_recovered_transform():
    color, ct = _synthetic_modalities_image()
    moving = preprocess.cv2.warpAffine(
        ct,
        preprocess.np.array([[1.08, 0.0, 13.0], [0.0, .92, -7.0]], dtype=preprocess.np.float32),
        (512, 304),
        borderValue=-1000,
    )
    solution = modalities.solve_affine(color, moving)
    before, after = modalities._internal_landmark_distance(color, moving, solution)
    assert after < before


def test_regional_qc_rejects_internal_landmark_mismatch():
    held = [{
        "color_ordinal": 5,
        "silhouette_before": 20.,
        "silhouette_after": 2.,
        "coverage": .95,
        "landmark_before": 18.,
        "landmark_after": 12.,
    }]
    qc = modalities._regional_qc(held, list(range(101)), 100)
    assert not qc["head"]["accepted"]
    assert qc["head"]["failures"] == ["internal_landmark_distance"]
    assert qc["head"]["worst_internal_landmark_distance"] == 12.


def test_regional_qc_rejects_a_bad_holdout_hidden_by_the_mean():
    held = [
        {"color_ordinal": ordinal, "silhouette_before": 20., "silhouette_after": 2., "coverage": .95, "landmark_before": 18., "landmark_after": after}
        for ordinal, after in ((5, 1.), (6, 13.))
    ]
    qc = modalities._regional_qc(held, list(range(101)), 100)
    assert qc["head"]["internal_landmark_after_distance"] == 7.
    assert qc["head"]["failures"] == ["internal_landmark_outlier"]


def test_monotonic_depth_mapping_retains_color_gap_and_endpoints():
    color_depths = [0, 1, 2, 4, 5, 6]  # Female-style missing plane at 3.
    ct_depths = [10, 11, 12, 13]
    knots = modalities.monotonic_depth_knots(color_depths, ct_depths, [(0, 0, .9), (1, 2, .9), (2, 4, .8), (3, 6, .8)])
    assert knots[0]["color_depth"] == 0 and knots[-1]["color_depth"] == 1
    assert [item["color_depth"] for item in knots] == sorted(item["color_depth"] for item in knots)
    assert [item["ct_frame"] for item in knots] == sorted(item["ct_frame"] for item in knots)
    assert any(abs(item["color_depth"] - 4 / 6) < 1e-6 for item in knots)


def test_constrained_dtw_recovers_nonlinear_depth_offset():
    count = 60
    color = [preprocess.np.array([index / (count - 1)], dtype=preprocess.np.float32) for index in range(count)]
    # CT anatomy descriptors lead the equal-ordinal estimate by up to 4%.
    ct = [preprocess.np.array([index / (count - 1) + .04 * preprocess.np.sin(preprocess.np.pi * index / (count - 1))], dtype=preprocess.np.float32) for index in range(count)]
    path, _ = modalities.constrained_dtw(color, ct, band=.05)
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(path, path[1:]))
    assert any(abs(i - j) >= 2 for i, j in path[8:-8])
    warped_error = preprocess.np.mean([abs(float(color[i][0] - ct[j][0])) for i, j in path])
    ordinal_error = preprocess.np.mean([abs(float(color[i][0] - ct[i][0])) for i in range(count)])
    assert warped_error < ordinal_error * .5


def test_spatial_sampling_is_denser_across_head_and_neck():
    uniform = modalities.representative_indices(1000, 32)
    spatial = modalities.spatial_sample_indices(1000, 32)
    assert len(spatial) > len(uniform)
    assert sum(index < 200 for index in spatial) >= 13
    assert spatial[-1] == 999


def test_bounds_transform_permits_large_but_anatomical_head_scale():
    fixed = preprocess.np.zeros((304, 512), dtype=preprocess.np.uint8)
    moving = fixed.copy()
    fixed[75:225, 195:325] = 1
    moving[30:280, 80:430] = 1
    transform = modalities._bounds_transform(fixed, moving)
    assert preprocess.np.allclose(transform.diagonal(), [350 / 130, 250 / 150], atol=1e-6)


def test_depth_profile_rejects_uncorroborated_repeated_anatomy_offset():
    color_depths = list(range(1900))
    ct_depths = list(range(1900))
    color_indices = list(range(0, 1861, 60))
    matches = []
    for ordinal, color_index in enumerate(color_indices):
        if ordinal < 5 or ordinal >= len(color_indices) - 5:
            ct_index = min(1899, color_index + 5)
        else:
            ct_index = color_index - 54
        matches.append((color_index, ct_index))
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(matches, matches[1:]))

    confidence = [1. if ordinal < 5 or ordinal >= len(matches) - 5 else .1 for ordinal in range(len(matches))]
    knots, _, _ = modalities._depth_profile(matches, color_depths, ct_depths, confidence)
    frames = [item["ct_frame"] for item in knots]
    assert all(a < b for a, b in zip(frames, frames[1:]))
    assert max(abs(item["ct_frame"] - (item["color_depth"] * 1899 + 5)) for item in knots) <= 9.5


def test_depth_profile_bounds_even_confident_repeated_anatomy_warp():
    color_depths = list(range(1900))
    ct_depths = list(range(1900))
    color_indices = list(range(0, 1861, 60))
    matches = [
        (color_index, min(1899, color_index + 5 if ordinal < 5 or ordinal >= len(color_indices) - 5 else color_index - 54))
        for ordinal, color_index in enumerate(color_indices)
    ]
    knots, _, _ = modalities._depth_profile(matches, color_depths, ct_depths, [1.] * len(matches))
    expected = [item["color_depth"] * 1899 + 5 for item in knots]
    assert max(abs(item["ct_frame"] - baseline) for item, baseline in zip(knots, expected)) <= 19.01


def test_inverse_uv_incorporates_orientation():
    solution = modalities.AffineSolution("flip_x", [[1, 0, 0], [0, 1, 0]], 1, 1, 0, 0, 0, 1, 0, 1)
    assert preprocess.np.allclose(modalities.inverse_uv(solution, 512, 304), [-1, 0, 1, 0, 1, 0], atol=1e-6)


def test_viewer_alignment_schema_is_color_referenced_and_sorted(monkeypatch, tmp_path):
    rgb = [preprocess.Slice("missing-a", "a", 0, "rgb"), preprocess.Slice("missing-b", "b", 6, "rgb")]
    ct = [preprocess.Slice("missing-c", "c", 10, "density"), preprocess.Slice("missing-d", "d", 13, "density")]
    captured = {}
    def mocked_register(baked_rgb, baked_ct, *_args, **_kwargs):
        captured["rgb"] = [item.path for item in baked_rgb]
        captured["ct"] = [item.path for item in baked_ct]
        return {
        "orientation": "flip_x", "depth_knots": [{"color_depth": 1., "ct_frame": 13., "confidence": .7}, {"color_depth": 0., "ct_frame": 10., "confidence": .8}],
        "spatial_knots": [{"color_depth": 1., "inverse_uv": [1, 0, 0, 0, 1, 0], "confidence": .7}, {"color_depth": 0., "inverse_uv": [-1, 0, 1, 0, 1, 0], "confidence": .8}],
        "qc": {"representative_count": 2}, "coverage_color": [.2, .8], "coverage_ct": [10, 13], "orientation_scores": {"flip_x": .75}, "representatives": [],
        }
    monkeypatch.setattr(preprocess, "register_modalities", mocked_register)
    manifest = tmp_path / "manifest.json"; corrections = tmp_path / "corrections.json"
    manifest_bytes = b'{"baked":true,"rgb":{"layers":[{"frame":0,"file":"rgb/a.png"},{"frame":6,"file":"rgb/b.png"}]},"density":{"layers":[{"frame":10,"file":"density/a.u16be"},{"frame":13,"file":"density/b.u16be"}]}}\n'
    manifest.write_bytes(manifest_bytes); corrections.write_bytes(b'{"transforms":{}}\n')
    alignment = preprocess.modality_alignment("female", rgb, ct, tmp_path)
    assert alignment["version"] == 1 and alignment["reference"] == "color"
    assert alignment["orientation"] == "flip_x"
    assert alignment["coverage"] == {"color_depth": [.2, .8], "ct_frame": [10, 13]}
    assert [item["color_depth"] for item in alignment["depth_knots"]] == [0., 1.]
    assert [item["color_depth"] for item in alignment["spatial_knots"]] == [0., 1.]
    assert captured["rgb"] == [str(tmp_path / "rgb/a.png"), str(tmp_path / "rgb/b.png")]
    assert captured["ct"] == [str(tmp_path / "density/a.u16be"), str(tmp_path / "density/b.u16be")]
    assert alignment["inputs"] == {
        "manifest_sha256": preprocess.hashlib.sha256(manifest_bytes).hexdigest(),
        "corrections_sha256": preprocess.hashlib.sha256(b'{"transforms":{}}\n').hexdigest(),
    }
