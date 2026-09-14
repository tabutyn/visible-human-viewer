export function bracketLayers(layers, frame, expectedStep = 1, forbidLongGap = false) {
  if (!layers.length) return null;
  let a = layers[0], b = layers.at(-1);
  for (const item of layers) { if (item.frame <= frame) a = item; if (item.frame >= frame) { b = item; break; } }
  const span = b.frame - a.frame;
  if (forbidLongGap && span > expectedStep * 2) return { a, b: a, mix: 0, missingGap: true };
  return { a, b, mix: span ? (frame - a.frame) / span : 0, missingGap: false };
}

export function blendColour(rgb, density, blend) {
  return rgb.map((value, index) => value * (1 - blend) + density[index] * blend);
}

export const IDENTITY_UV = [1, 0, 0, 0, 1, 0];

export function bracketKnots(knots, position, positionKey = "color_depth") {
  if (!Array.isArray(knots) || !knots.length || !Number.isFinite(position)) return null;
  const ordered = knots.filter((item) => Number.isFinite(item?.[positionKey])).sort((a, b) => a[positionKey] - b[positionKey]);
  if (!ordered.length) return null;
  const exact = ordered.find((item) => Math.abs(item[positionKey] - position) <= 1e-10);
  if (exact) return { left: exact, right: exact, mix: 0, ordered };
  if (position < ordered[0][positionKey]) return { left: ordered[0], right: ordered[0], mix: 0, ordered };
  if (position > ordered.at(-1)[positionKey]) return { left: ordered.at(-1), right: ordered.at(-1), mix: 0, ordered };
  const rightIndex = ordered.findIndex((item) => item[positionKey] > position);
  const left = ordered[rightIndex - 1], right = ordered[rightIndex];
  return { left, right, mix: (position - left[positionKey]) / (right[positionKey] - left[positionKey]), ordered };
}

export function interpolateKnots(knots, position, positionKey, valueKey) {
  const bracket = bracketKnots(knots, position, positionKey); if (!bracket) return null;
  const { left, right, mix } = bracket;
  if (left === right) return structuredClone(left[valueKey]);
  const a = left[valueKey], b = right[valueKey];
  if (Array.isArray(a) && Array.isArray(b)) return a.map((value, index) => value + (b[index] - value) * mix);
  return a + (b - a) * mix;
}

export function alignedDensityFrame(profile, colourDepth) {
  if (!profile?.depth_knots?.length) return null;
  const coverage = profile.coverage?.color_depth || [0, 1];
  if (colourDepth < coverage[0] || colourDepth > coverage[1]) return null;
  if (blockedAt(profile.depth_knots, colourDepth, profile)) return null;
  return interpolateKnots(profile.depth_knots, colourDepth, "color_depth", "ct_frame");
}

export function colourDepthForDensity(profile, frame) {
  return interpolateKnots(profile?.depth_knots, frame, "ct_frame", "color_depth");
}

export function alignmentAt(profile, colourDepth) {
  if (blockedAt(profile?.spatial_knots, colourDepth, profile)) return { inverseUv: IDENTITY_UV, confidence: null, unsupported: true };
  const matrix = interpolateKnots(profile?.spatial_knots, colourDepth, "color_depth", "inverse_uv");
  const confidence = interpolateKnots(profile?.spatial_knots, colourDepth, "color_depth", "confidence");
  return {
    inverseUv: Array.isArray(matrix) && matrix.length === 6 && matrix.every(Number.isFinite) ? matrix : IDENTITY_UV,
    confidence: Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : null,
  };
}

function blockedAt(knots, position, profile) {
  if (!Array.isArray(knots) || knots.length < 2) return false;
  if ((profile?.unsupported_ranges || []).some(([lo, hi]) => position > lo && position < hi)) return true;
  const ordered=knots.filter(item=>Number.isFinite(item?.color_depth)).sort((a,b)=>a.color_depth-b.color_depth);
  const right=ordered.findIndex(item=>item.color_depth>=position); if(right<=0)return false;
  const left=ordered[right-1], next=ordered[right];
  if(left.segment&&next.segment&&left.segment!==next.segment&&position>left.color_depth&&position<next.color_depth)return true;
  return (profile?.review_boundaries || []).some(boundary=>boundary>left.color_depth&&boundary<=next.color_depth&&position>left.color_depth&&position<next.color_depth);
}

export function applyDisplayAdjustment(matrix = IDENTITY_UV, adjustment = {}, width = 512, height = 304) {
  const scaleX = Number(adjustment.scale_x ?? 1);
  const scaleY = Number(adjustment.scale_y ?? 1);
  const tx = Number(adjustment.tx ?? 0);
  const ty = Number(adjustment.ty ?? 0);
  const rotation = Number(adjustment.rotation_deg ?? 0) * Math.PI / 180;
  if (![...matrix, scaleX, scaleY, tx, ty, rotation].every(Number.isFinite) || scaleX <= 0 || scaleY <= 0) return matrix;
  const cosine = Math.cos(rotation), sine = Math.sin(rotation);
  const inverseX = cosine / scaleX, inverseY = cosine / scaleY;
  const crossX = sine * height / (scaleX * width), crossY = -sine * width / (scaleY * height);
  const offsetX = .5 - inverseX * (.5 + tx / width) - crossX * (.5 + ty / height);
  const offsetY = .5 - crossY * (.5 + tx / width) - inverseY * (.5 + ty / height);
  const [a, b, c, d, e, f] = matrix;
  return [
    a * inverseX + b * crossY,
    a * crossX + b * inverseY,
    a * offsetX + b * offsetY + c,
    d * inverseX + e * crossY,
    d * crossX + e * inverseY,
    d * offsetX + e * offsetY + f,
  ];
}

export function inverseUvToDisplayParameters(matrix, orientation = "flip_y", width = 512, height = 304) {
  if (!Array.isArray(matrix) || matrix.length !== 6 || !matrix.every(Number.isFinite)) return null;
  const orient = { identity:[1,0,0,0,1,0], flip_y:[1,0,0,0,-1,1], flip_x:[-1,0,1,0,1,0], rotate_180:[-1,0,1,0,-1,1] }[orientation];
  if (!orient) return null;
  const multiply = (left, right) => [left[0]*right[0]+left[1]*right[3],left[0]*right[1]+left[1]*right[4],left[0]*right[2]+left[1]*right[5]+left[2],left[3]*right[0]+left[4]*right[3],left[3]*right[1]+left[4]*right[4],left[3]*right[2]+left[4]*right[5]+left[5]];
  const inverse = (value) => { const det=value[0]*value[4]-value[1]*value[3]; return Math.abs(det)<1e-10?null:[value[4]/det,-value[1]/det,(value[1]*value[5]-value[4]*value[2])/det,-value[3]/det,value[0]/det,(value[3]*value[2]-value[0]*value[5])/det]; };
  const forward = inverse(multiply(orient, matrix)); if (!forward) return null;
  const pixel = [forward[0],forward[1]*width/height,forward[2]*width,forward[3]*height/width,forward[4],forward[5]*height];
  const x_scale=Math.hypot(pixel[0],pixel[3]), y_scale=Math.hypot(pixel[1],pixel[4]);
  if (!(x_scale>0&&y_scale>0)) return null;
  const rotation_deg=Math.atan2(pixel[3],pixel[0])*180/Math.PI, cx=width/2, cy=height/2;
  return { x_position:pixel[0]*cx+pixel[1]*cy+pixel[2]-cx, y_position:pixel[3]*cx+pixel[4]*cy+pixel[5]-cy, rotation_deg, x_scale, y_scale };
}

const CALIBRATION_FIELDS = ["scale_x", "scale_y", "tx", "ty"];

export function localizedCalibrationAdjustment(spatialKnots, anchors, position) {
  const identity = { scale_x: 1, scale_y: 1, tx: 0, ty: 0 };
  const validAnchors = (Array.isArray(anchors) ? anchors : []).filter((item) =>
    Number.isFinite(item?.color_depth) && CALIBRATION_FIELDS.every((field) => Number.isFinite(item[field]))
  );
  if (!validAnchors.length || !Number.isFinite(position)) return identity;

  // A saved anchor is authoritative at its selected slice. Automatic spatial
  // knots are identity supports, so its residual correction fades out locally
  // instead of pulling the CT all the way to the next manual anchor.
  const exact = validAnchors.find((item) => Math.abs(item.color_depth - position) <= 1e-10);
  if (exact) return Object.fromEntries(CALIBRATION_FIELDS.map((field) => [field, exact[field]]));

  const supports = new Map();
  for (const item of Array.isArray(spatialKnots) ? spatialKnots : []) {
    if (Number.isFinite(item?.color_depth)) supports.set(item.color_depth.toFixed(12), { color_depth: item.color_depth, adjustment: [1, 1, 0, 0] });
  }
  for (const item of validAnchors) {
    supports.set(item.color_depth.toFixed(12), { color_depth: item.color_depth, adjustment: CALIBRATION_FIELDS.map((field) => item[field]) });
  }
  const knots = [...supports.values()].sort((a, b) => a.color_depth - b.color_depth);
  const values = interpolateKnots(knots, position, "color_depth", "adjustment");
  if (!Array.isArray(values)) return identity;
  return Object.fromEntries(CALIBRATION_FIELDS.map((field, index) => [field, values[index]]));
}

export function glMatrixFromInverseUv(matrix = IDENTITY_UV) {
  const [a, b, c, d, e, f] = matrix;
  return new Float32Array([a, d, 0, b, e, 0, c, f, 1]);
}
