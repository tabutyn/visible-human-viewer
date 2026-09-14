import assert from "node:assert/strict";
import test from "node:test";
import { alignedDensityFrame, alignmentAt, applyDisplayAdjustment, blendColour, bracketKnots, bracketLayers, colourDepthForDensity, glMatrixFromInverseUv, interpolateKnots, inverseUvToDisplayParameters, localizedCalibrationAdjustment } from "../viewer-math.mjs";

test("depth interpolation uses exact endpoints and linear intermediate", () => {
  const layers = [{ frame: 0 }, { frame: 10 }];
  assert.deepEqual(bracketLayers(layers, 0), { a: layers[0], b: layers[0], mix: 0, missingGap: false });
  assert.equal(bracketLayers(layers, 5).mix, .5);
  assert.deepEqual(bracketLayers(layers, 10), { a: layers[1], b: layers[1], mix: 0, missingGap: false });
});
test("density refuses a long missing gap and blend has correct endpoints", () => {
  assert.equal(bracketLayers([{ frame: 0 }, { frame: 9 }], 4, 3, true).missingGap, true);
  assert.deepEqual(blendColour([1, .2, 0], [.1, .8, .5], 0), [1, .2, 0]);
  assert.deepEqual(blendColour([1, .2, 0], [.1, .8, .5], 1), [.1, .8, .5]);
  assert.deepEqual(blendColour([1, 0], [0, 1], .5), [.5, .5]);
});
test("a two-step density interval is permitted but a larger one is not", () => {
  assert.equal(bracketLayers([{ frame: 0 }, { frame: 2 }], 1, 1, true).missingGap, false);
  assert.equal(bracketLayers([{ frame: 0 }, { frame: 3 }], 1, 1, true).missingGap, true);
});

test("registration profile maps colour depth to density and back", () => {
  const profile = {
    coverage: { color_depth: [.1, .9] },
    depth_knots: [
      { color_depth: .1, ct_frame: 10 },
      { color_depth: .5, ct_frame: 50 },
      { color_depth: .9, ct_frame: 110 },
    ],
  };
  assert.equal(alignedDensityFrame(profile, 0), null);
  assert.ok(Math.abs(alignedDensityFrame(profile, .3) - 30) < 1e-9);
  assert.ok(Math.abs(colourDepthForDensity(profile, 80) - .7) < 1e-9);
});

test("registration profile interpolates UV matrices and WebGL column order", () => {
  const profile = { spatial_knots: [
    { color_depth: 0, inverse_uv: [1, 0, 0, 0, 1, 0], confidence: .5 },
    { color_depth: 1, inverse_uv: [2, 0, .2, 0, 3, .4], confidence: .9 },
  ] };
  assert.deepEqual(interpolateKnots(profile.spatial_knots, .5, "color_depth", "inverse_uv"), [1.5, 0, .1, 0, 2, .2]);
  assert.deepEqual(alignmentAt(profile, .5), { inverseUv: [1.5, 0, .1, 0, 2, .2], confidence: .7 });
  assert.deepEqual([...glMatrixFromInverseUv([1, 2, 3, 4, 5, 6])], [1, 4, 0, 2, 5, 0, 3, 6, 1]);
});

test("knot bracketing reports the exact interpolation sources", () => {
  const knots = [{ color_depth: 0 }, { color_depth: .4 }, { color_depth: 1 }];
  const between = bracketKnots(knots, .25);
  assert.equal(between.left, knots[0]); assert.equal(between.right, knots[1]);
  assert.equal(between.mix, .625);
  const exact = bracketKnots(knots, .4);
  assert.equal(exact.left, knots[1]); assert.equal(exact.right, knots[1]);
});

test("manual display adjustment expands around center and translates intuitively", () => {
  assert.deepEqual(applyDisplayAdjustment([1, 0, 0, 0, 1, 0], { scale_x: 2, scale_y: 2 }), [.5, 0, .25, 0, .5, .25]);
  const translated = applyDisplayAdjustment([1, 0, 0, 0, 1, 0], { tx: 51.2, ty: 30.4 });
  assert.ok(translated.every((value, index) => Math.abs(value - [1, 0, -.1, 0, 1, -.1][index]) < 1e-12));
  assert.deepEqual(applyDisplayAdjustment([2, 0, .1, 0, 3, .2], { scale_x: 2, scale_y: 1, tx: 0, ty: 0 }), [1, 0, .6, 0, 3, .2]);
});

test("rotation adjustment and six-parameter decomposition are finite", () => {
  const matrix = applyDisplayAdjustment([1,0,0,0,-1,1], { scale_x:1.2, scale_y:.8, tx:12, ty:-7, rotation_deg:5 });
  const parameters = inverseUvToDisplayParameters(matrix, "flip_y");
  assert.ok(parameters);
  assert.ok(Math.abs(parameters.x_position - 12) < .2);
  assert.ok(Math.abs(parameters.y_position + 7) < .2);
  assert.ok(Math.abs(parameters.rotation_deg - 5) < .05);
  assert.ok(Math.abs(parameters.x_scale - 1.2) < .01);
  assert.ok(Math.abs(parameters.y_scale - .8) < .01);
  const rebuilt = applyDisplayAdjustment([1,0,0,0,-1,1], { scale_x:parameters.x_scale, scale_y:parameters.y_scale,
    tx:parameters.x_position, ty:parameters.y_position, rotation_deg:parameters.rotation_deg });
  rebuilt.forEach((value,index)=>assert.ok(Math.abs(value-matrix[index])<1e-9));
});

test("manual calibration remains exact at anchors and fades at automatic knots", () => {
  const spatialKnots = [0, .25, .5, .75, 1].map((color_depth) => ({ color_depth }));
  const anchors = [
    { color_depth: .4, scale_x: 2, scale_y: 1, tx: 10, ty: -4 },
    { color_depth: .8, scale_x: 1, scale_y: 1.5, tx: 0, ty: 8 },
  ];
  assert.deepEqual(localizedCalibrationAdjustment(spatialKnots, anchors, .4), { scale_x: 2, scale_y: 1, tx: 10, ty: -4 });
  assert.deepEqual(localizedCalibrationAdjustment(spatialKnots, anchors, .25), { scale_x: 1, scale_y: 1, tx: 0, ty: 0 });
  const halfway = localizedCalibrationAdjustment(spatialKnots, anchors, .325);
  assert.ok(Math.abs(halfway.scale_x - 1.5) < 1e-12);
  assert.ok(Math.abs(halfway.tx - 5) < 1e-12);
  assert.deepEqual(localizedCalibrationAdjustment(spatialKnots, anchors, .6), { scale_x: 1, scale_y: 1, tx: 0, ty: 0 });
});
