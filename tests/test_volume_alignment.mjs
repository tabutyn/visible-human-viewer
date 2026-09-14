import assert from "node:assert/strict";
import test from "node:test";
import { applyDisplayAdjustment } from "../viewer-math.mjs";
import { buildVolumeAlignmentLut, ctFrameToIndex, inverseAffine6, multiplyAffine6 } from "../volume-alignment.mjs";

const IDENTITY = [1, 0, 0, 0, 1, 0], FLIP_Y = [1, 0, 0, 0, -1, 1];
const near = (actual, expected, tolerance = 2e-6) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < tolerance, `component ${index}: ${value} != ${expected[index]}`));
};
const knot = (colorFrame, ctFrame, matrix = FLIP_Y) => ({ color_frame: colorFrame, color_depth: colorFrame / 1000, ct_frame: ctFrame, inverse_uv: [...matrix] });
function fixture(knots = [knot(100, 10), knot(120, 30), knot(140, 50)]) {
  return { volumeManifest: { subject: "male", dimensions: [16, 16, 61], slice_frames: Array.from({ length: 61 }, (_, index) => index + 5) },
    sourceManifest: { colourPlanes: 1001, rgb: { layers: Array.from({ length: 1001 }, (_, frame) => ({ frame })) } },
    rgbManifest: { subject: "male", dimensions: [16, 16, 61], alignment_knots: structuredClone(knots) },
    knots: structuredClone(knots), profile: { orientation: "flip_y" } };
}
const row = (lut, nativeIndex) => [...lut.data.slice(nativeIndex * 8, nativeIndex * 8 + 8)];
const matrixRow = (lut, nativeIndex) => { const [a, b, c, , d, e, f] = row(lut, nativeIndex); return [a, b, c, d, e, f]; };
const transform = (matrix, uv) => [matrix[0] * uv[0] + matrix[1] * uv[1] + matrix[2], matrix[3] * uv[0] + matrix[4] * uv[1] + matrix[5]];

test("affine inversion and irregular source CT frame indexing", () => {
  const matrix = applyDisplayAdjustment(FLIP_Y, { tx: 12, ty: -7, scale_x: 1.3, scale_y: .8, rotation_deg: 18 });
  near(multiplyAffine6(matrix, inverseAffine6(matrix)), IDENTITY);
  assert.equal(inverseAffine6([0, 0, 0, 0, 0, 0]), null);
  assert.equal(ctFrameToIndex([5, 6, 8, 9], 7), 1.5);
  assert.equal(ctFrameToIndex([5, 6, 8, 9], 9), 3);
  assert.equal(ctFrameToIndex([5, 6, 8, 9], 4), null);
});

test("unchanged baked data stays identity, including flip, varying rotation, anisotropic scale and legacy coefficient interpolation", () => {
  const input = fixture([
    knot(100, 10, applyDisplayAdjustment(FLIP_Y, { tx: 11, ty: -8, scale_x: .8, scale_y: 1.2, rotation_deg: -26 })),
    knot(120, 30, applyDisplayAdjustment(FLIP_Y, { tx: -3, ty: 8, scale_x: 1.3, scale_y: .9, rotation_deg: 17 })),
    knot(140, 50, applyDisplayAdjustment(FLIP_Y, { tx: 3, scale_x: 1.1, rotation_deg: -8 })),
  ]);
  const lut = buildVolumeAlignmentLut(input);
  assert.equal(lut.enabled, true);
  assert.equal(lut.validCount, 41);
  for (let index = 5; index <= 45; index++) {
    near(matrixRow(lut, index), IDENTITY);
    near([row(lut, index)[3]], [index / 60]);
    assert.equal(row(lut, index)[7], 1);
  }
  assert.equal(row(lut, 0)[7], 0);
  assert.equal(row(lut, 60)[7], 0);
});

test("translations, anisotropic scale and rotation agree with the source RGB projection at the edited knot", () => {
  const input = fixture();
  const adjusted = applyDisplayAdjustment(FLIP_Y, { tx: 24, ty: -13, scale_x: 1.24, scale_y: .83, rotation_deg: 21 });
  input.draft = { ...input.knots[1], inverse_uv: adjusted };
  const lut = buildVolumeAlignmentLut(input);
  assert.equal(lut.selected.valid, true);
  assert.equal(lut.selected.index, 25);
  near([lut.selected.z], [25 / 60]);
  const rgbPoint = [.38, .61], newCtPoint = transform(adjusted, rgbPoint), originalBakedPoint = transform(FLIP_Y, rgbPoint);
  near(transform(matrixRow(lut, 25), newCtPoint), originalBakedPoint);
  near(transform(lut.selected.inverseUv, newCtPoint), rgbPoint);
  near(matrixRow(lut, 5), IDENTITY);
  near(matrixRow(lut, 45), IDENTITY);
  const halfway = matrixRow(lut, 15);
  assert.notDeepEqual(halfway, IDENTITY);
  assert.ok(Math.abs(halfway[2]) < Math.abs(matrixRow(lut, 25)[2]));
});

test("Z edits move the selected source RGB plane on the native CT grid and fade toward neighbor correspondences", () => {
  const input = fixture();
  input.draft = { ...input.knots[1], ct_frame: 34 };
  const lut = buildVolumeAlignmentLut(input);
  near([lut.selected.z], [29 / 60]);
  near([row(lut, 29)[3]], [25 / 60]); // CT 34 now samples the baked CT 30 plane.
  near([row(lut, 5)[3], row(lut, 45)[3]], [5 / 60, 45 / 60]);
  near(matrixRow(lut, 29), IDENTITY);
  assert.ok(row(lut, 17)[3] < 17 / 60);
});

test("an edit stays local to its neighboring knots", () => {
  const input = fixture([knot(80, 5), knot(100, 15), knot(120, 30), knot(140, 45), knot(160, 60)]);
  input.draft = { ...input.knots[2], inverse_uv: applyDisplayAdjustment(FLIP_Y, { tx: 32 }) };
  const lut = buildVolumeAlignmentLut(input);
  near(matrixRow(lut, 5), IDENTITY);
  near(matrixRow(lut, 10), IDENTITY);
  near(matrixRow(lut, 40), IDENTITY);
  near(matrixRow(lut, 50), IDENTITY);
  assert.ok(Math.abs(matrixRow(lut, 25)[2]) > .01);
  near([matrixRow(lut, 17)[2]], [matrixRow(lut, 25)[2] * 7 / 15]);
});

test("interpolated rotations take the shortest arc and anisotropic scales interpolate geometrically", () => {
  const input = fixture([knot(100, 10), knot(120, 30)]);
  input.knots[0].inverse_uv = applyDisplayAdjustment(FLIP_Y, { rotation_deg: 170, scale_x: .5, scale_y: .81 });
  input.knots[1].inverse_uv = applyDisplayAdjustment(FLIP_Y, { rotation_deg: -170, scale_x: 2, scale_y: 1.21 });
  const lut = buildVolumeAlignmentLut(input);
  const midpoint = applyDisplayAdjustment(FLIP_Y, { rotation_deg: 180, scale_x: 1, scale_y: .99 });
  near(matrixRow(lut, 15), multiplyAffine6(FLIP_Y, inverseAffine6(midpoint)));
});

test("adjacent valid native rows on either side of a missing CT range use different continuity groups", () => {
  const input = fixture([knot(100, 5), knot(102, 6), knot(104, 10), knot(106, 11)]);
  input.volumeManifest.dimensions[2] = 4;
  input.volumeManifest.slice_frames = [5, 6, 10, 11];
  input.rgbManifest.dimensions[2] = 4;
  const lut = buildVolumeAlignmentLut(input);
  assert.equal(lut.validCount, 4);
  assert.equal(row(lut, 0)[7], row(lut, 1)[7]);
  assert.notEqual(row(lut, 1)[7], row(lut, 2)[7]);
  assert.equal(row(lut, 2)[7], row(lut, 3)[7]);
});

test("deleting a bad knot removes its baked deformation instead of retaining it as a hidden interpolation support", () => {
  const input = fixture([knot(100, 10), knot(120, 30, applyDisplayAdjustment(FLIP_Y, { tx: 64 })), knot(140, 50)]);
  input.profile.deleted_knot_frames = [120];
  const lut = buildVolumeAlignmentLut(input);
  // The newly straight curve must sample the offset original RGB volume to undo the deleted middle knot.
  near(matrixRow(lut, 25), multiplyAffine6(input.rgbManifest.alignment_knots[1].inverse_uv, inverseAffine6(FLIP_Y)));
  assert.ok(Math.abs(matrixRow(lut, 25)[2]) > .1);
  assert.equal(lut.validCount, 41);
});

test("an unchanged long interval preserves the existing baked view, but a correction cannot interpolate over it", () => {
  const input = fixture([knot(100, 10), knot(200, 50)]);
  const unchanged = buildVolumeAlignmentLut(input);
  assert.equal(unchanged.validCount, 41);
  assert.equal(unchanged.preservedCount, 39);
  near(matrixRow(unchanged, 25), IDENTITY);
  input.draft = { ...input.knots[0], inverse_uv: applyDisplayAdjustment(FLIP_Y, { tx: 8 }) };
  const edited = buildVolumeAlignmentLut(input);
  assert.equal(row(edited, 25)[7], 0);
  assert.equal(edited.validCount, 2);
});

test("corrections cannot cross acquisition/FOV boundaries, unsupported regions or source gaps", () => {
  for (const modify of [
    (input) => { input.profile.review_boundaries = [.11]; },
    (input) => { input.profile.unsupported_ranges = [[.105, .115]]; },
    (input) => { input.sourceManifest.boundaries = [{ modality: "rgb", right: 110 }]; },
    (input) => { input.sourceManifest.boundaries = [{ modality: "density", right: 20 }]; },
    (input) => { input.sourceManifest.gaps = [{ modality: "rgb", left: 108, right: 112 }]; },
    (input) => { input.volumeManifest.acquisition_segments = [{ start_index: 0, end_index: 14 }, { start_index: 15, end_index: 60 }]; },
    (input) => { input.knots[0].segment = "head"; input.knots[1].segment = "neck"; },
    (input) => { input.sourceManifest.rgb.layers = input.sourceManifest.rgb.layers.filter(({ frame }) => frame <= 108 || frame >= 112); },
  ]) {
    const input = fixture(); modify(input);
    input.draft = { ...input.knots[0], inverse_uv: applyDisplayAdjustment(FLIP_Y, { tx: 8 }) };
    const lut = buildVolumeAlignmentLut(input);
    assert.equal(row(lut, 15)[7], 0, String(modify));
    assert.equal(row(lut, 5)[7], 1, "exact supported endpoint remains available");
    assert.ok(row(lut, 35)[7] > 0, "unaffected next interval remains available");
    assert.ok(lut.blockedCorrectionCount > 0);
    assert.equal(lut.reason, "some_corrections_unsupported");
  }
});

test("existing baked images remain unchanged on entering a seam region, with explicit continuity groups", () => {
  const input = fixture();
  input.sourceManifest.boundaries = [{ modality: "density", right: 20 }];
  const lut = buildVolumeAlignmentLut(input);
  assert.equal(lut.validCount, 41);
  assert.ok(lut.preservedCount > 0);
  assert.equal(lut.reason, null);
  near(matrixRow(lut, 14), IDENTITY);
  near(matrixRow(lut, 15), IDENTITY);
  assert.ok(row(lut, 14)[7] > 0);
  assert.ok(row(lut, 15)[7] > 0);
  assert.notEqual(row(lut, 14)[7], row(lut, 15)[7], "the GPU must not blend across the acquisition seam");
});

test("metric-only rejected boundary suggestions do not prevent a local correction", () => {
  const input = fixture();
  input.sourceManifest.boundaries = [{ modality: "rgb", right: 110,
    reason: { metadata_gap: false, position_jump: false, acquisition_change: false, fov_change: false, metric_z: 7 },
    applied_transform: { tx: 0, ty: 0, scale: 1, rotation_deg: 0 }, accepted: false }];
  input.draft = { ...input.knots[0], inverse_uv: applyDisplayAdjustment(FLIP_Y, { tx: 8 }) };
  assert.equal(buildVolumeAlignmentLut(input).validCount, 41);
  input.sourceManifest.boundaries[0].applied_transform.tx = 3;
  assert.equal(row(buildVolumeAlignmentLut(input), 15)[7], 0);
});

test("a Z edit cannot pull source RGB across a CT acquisition boundary", () => {
  const input = fixture();
  input.sourceManifest.boundaries = [{ modality: "density", right: 32 }];
  input.draft = { ...input.knots[1], ct_frame: 34 };
  const lut = buildVolumeAlignmentLut(input);
  assert.equal(row(lut, 29)[7], 0);
  assert.equal(lut.selected.valid, true, "exact raw source plane remains reviewable without interpolating the baked acquisition");
});

test("missing snapshots and changed CT grids never silently advertise a live correction", () => {
  const input = fixture(); input.draft = input.knots[1];
  delete input.rgbManifest.alignment_knots;
  const missing = buildVolumeAlignmentLut(input);
  assert.equal(missing.enabled, false);
  assert.equal(missing.reason, "missing_alignment_snapshot");
  assert.equal(missing.selected.valid, true, "source PNG review remains possible before rebuilding a volume");
  assert.ok(missing.data.every((value) => value === 0));
  const mismatch = fixture(); mismatch.rgbManifest.dimensions = [8, 8, 31];
  assert.equal(buildVolumeAlignmentLut(mismatch).reason, "incompatible_rgb_volume");
  const stale = fixture(); stale.rgbManifest.status = "stale";
  assert.equal(buildVolumeAlignmentLut(stale).reason, "stale_source_geometry");
});

test("reversed Z, duplicate knots, stale anchors and singular transforms cannot be used for interpolation", () => {
  const reversed = fixture(); reversed.draft = { ...reversed.knots[1], ct_frame: 55 };
  assert.equal(buildVolumeAlignmentLut(reversed).reason, "non_monotonic_z");
  assert.equal(buildVolumeAlignmentLut(reversed).selected.valid, false);
  const duplicate = fixture(); duplicate.knots.push(duplicate.knots[0]);
  assert.equal(buildVolumeAlignmentLut(duplicate).reason, "non_monotonic_z");
  for (const changes of [{ stale: true }, { inverse_uv: [0, 0, 0, 0, 0, 0] }]) {
    const input = fixture(); input.knots[1] = { ...input.knots[1], ...changes };
    const lut = buildVolumeAlignmentLut(input);
    assert.equal(row(lut, 25)[7], 0);
    assert.equal(row(lut, 15)[7], 0);
    assert.equal(row(lut, 35)[7], 0);
  }
});
