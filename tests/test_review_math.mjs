import assert from "node:assert/strict";
import test from "node:test";
import { fitLandmarkTransform, reviewAlignmentAt, validateReviewMatrix } from "../review-math.mjs";

const IDENTITY = [1, 0, 0, 0, 1, 0];
const FLIPPED = [1, 0, 0, 0, -1, 1];
const POINTS = [[.2, .2], [.7, .2], [.7, .7], [.2, .7], [.45, .5]];
const apply = (matrix, [u, v]) => [matrix[0] * u + matrix[1] * v + matrix[2], matrix[3] * u + matrix[4] * v + matrix[5]];
const pairsFor = (matrix) => POINTS.map((color) => ({ color, density: apply(matrix, color) }));
const close = (actual, expected, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < tolerance, `${index}: ${value} versus ${expected[index]}`));
};
const anchor = (depth, frame, matrix = FLIPPED) => ({ color_depth: depth, ct_frame: frame, inverse_uv: [...matrix] });

test("landmarks fit the inverse UV direction including independent scales and shear", () => {
  const expected = [.85, .08, .04, -.12, -.9, .96];
  const result = fitLandmarkTransform(pairsFor(expected), { referenceMatrix: FLIPPED });
  close(result.inverse_uv, expected);
  assert.ok(result.rms_pixels < 1e-10);
  assert.ok(result.max_error_pixels < 1e-10);
  assert.equal(result.count, 5);
  close(apply(result.inverse_uv, POINTS[0]), pairsFor(expected)[0].density);
});

test("least-squares residuals are measured in CT analysis pixels", () => {
  const pairs = pairsFor(IDENTITY);
  pairs[4].density[0] += 1 / 512;
  const result = fitLandmarkTransform(pairs);
  const errors = pairs.map((pair) => {
    const prediction = apply(result.inverse_uv, pair.color);
    return Math.hypot((prediction[0] - pair.density[0]) * 512, (prediction[1] - pair.density[1]) * 304);
  });
  assert.ok(result.rms_pixels > .1 && result.rms_pixels < 1);
  assert.ok(Math.abs(result.rms_pixels - Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length)) < 1e-10);
  assert.ok(Math.abs(result.max_error_pixels - Math.max(...errors)) < 1e-10);
});

test("fit rejects too few, collinear, and nearly collinear points on either side", () => {
  assert.throws(() => fitLandmarkTransform(pairsFor(IDENTITY).slice(0, 2)), /three/);
  const line = [[.1, .1], [.3, .3], [.6, .6 + 1e-9]];
  assert.throws(() => fitLandmarkTransform(line.map((color, index) => ({ color, density: POINTS[index] }))), /collinear/);
  assert.throws(() => fitLandmarkTransform(line.map((density, index) => ({ density, color: POINTS[index] }))), /collinear/);
  assert.throws(() => fitLandmarkTransform(Array.from({ length: 3 }, () => ({ color: [.4, .4], density: [.5, .5] }))), /distinct/);
});

test("fit permits CT flips and detects mismatches against an optional reference", () => {
  close(fitLandmarkTransform(pairsFor(FLIPPED)).inverse_uv, FLIPPED);
  close(fitLandmarkTransform(pairsFor(FLIPPED), { referenceMatrix: FLIPPED }).inverse_uv, FLIPPED);
  assert.throws(() => fitLandmarkTransform(pairsFor(FLIPPED), { referenceMatrix: IDENTITY }), /orientation/);
  assert.throws(() => fitLandmarkTransform(pairsFor(IDENTITY), { referenceMatrix: [.2, 0, 0, 0, .2, 0] }), /fourfold/);
});

test("fit rejects nonfinite, out-of-image and implausible inputs", () => {
  for (const bad of [NaN, Infinity, -.01, 1.01, "0.5"]) {
    const pairs = pairsFor(IDENTITY);
    pairs[0].color = [bad, .1];
    assert.throws(() => fitLandmarkTransform(pairs), /finite UV/);
  }
  assert.throws(() => fitLandmarkTransform(pairsFor([.01, 0, .5, 0, 1, 0])), /implausible/);
  assert.throws(() => fitLandmarkTransform(pairsFor(IDENTITY), { width: 0 }), /dimensions/);
  assert.throws(() => fitLandmarkTransform(pairsFor(IDENTITY), { referenceMatrix: [1, 2] }), /six finite/);
});

test("shared validation rejects determinant-valid extreme stretches and unsupported offsets", () => {
  assert.equal(validateReviewMatrix(FLIPPED), true);
  assert.equal(validateReviewMatrix([1.4, 0, -.2, 0, -.85, .84], { referenceMatrix: FLIPPED }), true);
  assert.throws(() => validateReviewMatrix([20, 0, 0, 0, .05, 0]), /implausible/);
  assert.throws(() => validateReviewMatrix([5.6, 0, 0, 0, -.2125, 0]), /implausible/);
  assert.throws(() => validateReviewMatrix([1, 0, 21, 0, 1, 0]), /supported range/);
  assert.throws(() => validateReviewMatrix(FLIPPED, { referenceMatrix: IDENTITY }), /orientation/);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100, [1, 0, 21, 0, 1, 0])], .2), null);
});

test("reviewed anchors are absolute, copied exactly, and limited to the same color frame", () => {
  const saved = anchor(.2, 123, [.9, .02, .04, .01, -1.1, 1.02]);
  const profile = { spatial_knots: [anchor(.2, 30, IDENTITY)], coverage: { color_depth: [.3, .8] } };
  const result = reviewAlignmentAt(profile, [saved], .2);
  assert.deepEqual(result.inverse_uv, saved.inverse_uv);
  assert.equal(result.ct_frame, 123);
  assert.equal(result.interpolated, false);
  result.inverse_uv[0] = 8;
  assert.equal(saved.inverse_uv[0], .9);
  assert.equal(reviewAlignmentAt(profile, [saved], .201), null);
  assert.equal(reviewAlignmentAt(profile, [saved], .20049, { maxFrame: 1000 }).ct_frame, 123);
  assert.equal(reviewAlignmentAt(profile, [saved], .20051, { maxFrame: 1000 }), null);
});

test("interpolation requires opt-in and never extrapolates or bridges a long span", () => {
  const anchors = [anchor(.2, 100), anchor(.23, 130)];
  assert.equal(reviewAlignmentAt(null, anchors, .215), null);
  const middle = reviewAlignmentAt(null, anchors, .215, { interpolate: true });
  assert.ok(Math.abs(middle.ct_frame - 115) < 1e-9);
  close(middle.inverse_uv, FLIPPED);
  assert.equal(middle.interpolated, true);
  assert.equal(reviewAlignmentAt(null, anchors, .19, { interpolate: true }), null);
  assert.equal(reviewAlignmentAt(null, anchors, .24, { interpolate: true }), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100), anchor(.3, 200)], .25, { interpolate: true }), null);
});

test("review interpolation blocks boundaries, flipped orientation, and reversed CT correspondence", () => {
  const anchors = [anchor(.2, 100), anchor(.23, 130)];
  assert.equal(reviewAlignmentAt(null, anchors, .215, { interpolate: true, boundaries: [.21] }), null);
  assert.equal(reviewAlignmentAt(null, anchors, .215, { interpolate: true, boundaries: [.23] }), null);
  assert.ok(reviewAlignmentAt(null, anchors, .215, { interpolate: true, boundaries: [.2] }));
  assert.equal(reviewAlignmentAt({ review_boundaries: [.21] }, anchors, .215, { interpolate: true }), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100), anchor(.23, 130, IDENTITY)], .215, { interpolate: true }), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 130), anchor(.23, 100)], .215, { interpolate: true }), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100), anchor(.23, 100)], .215, { interpolate: true }), null);
});

test("CT seams block interpolation using reviewed frame correspondence independently of automatic depth", () => {
  const profile = { depth_knots: [{ color_depth: .1, ct_frame: 100 }, { color_depth: .4, ct_frame: 130 }] };
  const anchors = [anchor(.2, 100), anchor(.23, 130)];
  // Automatic mapping puts CT frame 120 at depth .3, outside this reviewed
  // interval. It must still block interpolation through reviewed frames 100–130.
  assert.equal(reviewAlignmentAt(profile, anchors, .215, { interpolate: true, boundaries: [.3], ctBoundaries: [120] }), null);
  assert.equal(reviewAlignmentAt(profile, anchors, .215, { interpolate: true, ctBoundaries: [130] }), null);
  assert.ok(reviewAlignmentAt(profile, anchors, .215, { interpolate: true, ctBoundaries: [100] }));
  assert.ok(reviewAlignmentAt(profile, anchors, .215, { interpolate: true, ctBoundaries: [90, 140, NaN] }));
  assert.equal(reviewAlignmentAt(profile, anchors, .23, { interpolate: true, ctBoundaries: [130] }).ct_frame, 130);
});

test("rotation and scale interpolation stays nonsingular and uses geometric stretch", () => {
  const inverted = [-1, 0, 1, 0, -1, 1];
  const rotated = reviewAlignmentAt(null, [anchor(.2, 100, IDENTITY), anchor(.23, 130, inverted)], .215, { interpolate: true });
  assert.ok(rotated);
  const [a, b, , d, e] = rotated.inverse_uv;
  assert.ok(Math.abs(a * e - b * d - 1) < 1e-9);
  close(apply(rotated.inverse_uv, [.5, .5]), [.5, .5]);
  const scaled = reviewAlignmentAt(null, [anchor(.2, 100, IDENTITY), anchor(.23, 130, [4, 0, -1.5, 0, 4, -1.5])], .215, { interpolate: true });
  close(scaled.inverse_uv, [2, 0, -.5, 0, 2, -.5]);
});

test("malformed or conflicting anchors block interpolation and legacy deltas never become absolute fits", () => {
  const legacy = { color_depth: .2, scale_x: 1, scale_y: 1, tx: 0, ty: 0 };
  assert.equal(reviewAlignmentAt(null, [legacy], .2), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.19, 90), legacy, anchor(.23, 130)], .215, { interpolate: true }), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100), anchor(.2, 110)], .2), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100), anchor(.2, 110), anchor(.23, 130)], .215, { interpolate: true }), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, -1)], .2), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100, [NaN, 0, 0, 0, 1, 0])], .2), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100)], NaN), null);
  assert.equal(reviewAlignmentAt(null, [anchor(.2, 100), anchor(.23, 130)], .215, { interpolate: true, maxGap: NaN }), null);
});
