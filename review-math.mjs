// All affine matrices map displayed color UV to the original density image UV.
// The CT orientation (including a vertical flip) is already part of that matrix.
const DEFAULT_WIDTH = 512;
const DEFAULT_HEIGHT = 304;
const EPSILON = 1e-10;

function dimensions(options) {
  const width = options.width ?? DEFAULT_WIDTH, height = options.height ?? DEFAULT_HEIGHT;
  if (![width, height].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("Analysis dimensions must be finite and positive.");
  }
  return [width, height];
}

function pixelLinear(matrix, width, height) {
  return [matrix[0], matrix[1] * width / height, matrix[3] * height / width, matrix[4]];
}

function singularValues([a, b, c, d]) {
  const determinant = a * d - b * c;
  const trace = a * a + b * b + c * c + d * d;
  const maximum = Math.sqrt(Math.max(0, (trace + Math.sqrt(Math.max(0, trace * trace - 4 * determinant * determinant))) / 2));
  return { minimum: maximum ? Math.abs(determinant) / maximum : 0, maximum, determinant };
}

function validateMatrix(matrix, width, height, referenceMatrix) {
  if (!Array.isArray(matrix) || matrix.length !== 6 || !matrix.every(Number.isFinite)) {
    throw new Error("The alignment must contain six finite affine coefficients.");
  }
  if (matrix.some((value) => Math.abs(value) > 20)) {
    throw new Error("Affine coefficients must stay within the supported range of -20 to 20.");
  }
  const linear = pixelLinear(matrix, width, height);
  const singular = singularValues(linear);
  if (singular.minimum < .1 || singular.maximum > 10 || singular.maximum / singular.minimum > 20) {
    throw new Error("The fit has an implausible scale or shear. Spread the matching landmarks across the anatomy.");
  }
  if (referenceMatrix !== undefined) {
    const reference = validateMatrix(referenceMatrix, width, height);
    if (singular.determinant * reference.determinant <= 0) {
      throw new Error("The landmark fit reverses the reference orientation. Check the order of the matching points.");
    }
    const [a, b, c, d] = pixelLinear(referenceMatrix, width, height);
    const [e, f, g, h] = linear;
    const relative = singularValues([
      (e * d - f * c) / reference.determinant, (f * a - e * b) / reference.determinant,
      (g * d - h * c) / reference.determinant, (h * a - g * b) / reference.determinant,
    ]);
    if (relative.minimum < .25 || relative.maximum > 4) {
      throw new Error("The landmark fit changes scale more than fourfold relative to the reference. Check the matching points.");
    }
  }
  return singular;
}

/** Shared browser/server acceptance rule; returns true, or throws a fit error. */
export function validateReviewMatrix(matrix, options = {}) {
  const [width, height] = dimensions(options);
  validateMatrix(matrix, width, height, options.referenceMatrix);
  return true;
}

function normalizedPoints(pairs, key, width, height) {
  const points = pairs.map((pair) => {
    const point = pair?.[key];
    if (!Array.isArray(point) || point.length !== 2 || !point.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
      throw new Error(`${key === "color" ? "Color" : "Density"} landmarks must be finite UV points inside the source image.`);
    }
    return [point[0] * width, point[1] * height];
  });
  const center = points.reduce((sum, point) => [sum[0] + point[0] / points.length, sum[1] + point[1] / points.length], [0, 0]);
  const centered = points.map((point) => [point[0] - center[0], point[1] - center[1]]);
  const scale = Math.sqrt(centered.reduce((sum, [x, y]) => sum + (x * x + y * y) / points.length, 0));
  if (!(scale > 1e-6)) throw new Error("Landmarks must be distinct and spread across the anatomy.");
  const normalized = centered.map(([x, y]) => [x / scale, y / scale]);
  let xx = 0, xy = 0, yy = 0;
  for (const [x, y] of normalized) { xx += x * x; xy += x * y; yy += y * y; }
  const determinant = xx * yy - xy * xy;
  if (determinant <= (xx + yy) ** 2 * 1e-6) {
    throw new Error(`${key === "color" ? "Color" : "Density"} landmarks are collinear or nearly collinear. Add a point away from that line.`);
  }
  return { points, center, scale, normalized, xx, xy, yy, determinant };
}

/** Least-squares color-to-source-density affine fit. Errors use CT analysis pixels. */
export function fitLandmarkTransform(pairs, options = {}) {
  if (!Array.isArray(pairs) || pairs.length < 3) {
    throw new Error("Choose at least three matching landmark pairs.");
  }
  const [width, height] = dimensions(options);
  const color = normalizedPoints(pairs, "color", width, height);
  const density = normalizedPoints(pairs, "density", width, height);
  const coefficients = [0, 1].map((axis) => {
    let xTarget = 0, yTarget = 0;
    for (let index = 0; index < pairs.length; index++) {
      xTarget += color.normalized[index][0] * density.normalized[index][axis];
      yTarget += color.normalized[index][1] * density.normalized[index][axis];
    }
    const ratio = density.scale / color.scale;
    return [
      ratio * (color.yy * xTarget - color.xy * yTarget) / color.determinant,
      ratio * (color.xx * yTarget - color.xy * xTarget) / color.determinant,
    ];
  });
  const [[a, b], [d, e]] = coefficients;
  const tx = density.center[0] - a * color.center[0] - b * color.center[1];
  const ty = density.center[1] - d * color.center[0] - e * color.center[1];
  const matrix = [a, b * height / width, tx / width, d * width / height, e, ty / height];
  validateMatrix(matrix, width, height, options.referenceMatrix);
  const errors = color.points.map(([x, y], index) => Math.hypot(
    a * x + b * y + tx - density.points[index][0],
    d * x + e * y + ty - density.points[index][1],
  ));
  return {
    inverse_uv: matrix,
    rms_pixels: Math.sqrt(errors.reduce((sum, value) => sum + value * value, 0) / errors.length),
    max_error_pixels: Math.max(...errors),
    count: pairs.length,
  };
}

// A symmetric 2x2 matrix stored as [xx, xy, yy]. Used for log-Euclidean stretch.
function symmetricFunction([xx, xy, yy], operation) {
  const midpoint = (xx + yy) / 2;
  const radius = Math.hypot((xx - yy) / 2, xy);
  if (radius < 1e-12) return [operation(midpoint), 0, operation(midpoint)];
  const high = operation(midpoint + radius), low = operation(midpoint - radius);
  const average = (high + low) / 2, factor = (high - low) / (2 * radius);
  return [average + factor * (xx - midpoint), factor * xy, average + factor * (yy - midpoint)];
}

function polar(matrix, width, height, handedness) {
  let [a, b, c, d] = pixelLinear(matrix, width, height);
  b *= handedness; d *= handedness;
  const angle = Math.atan2(c - b, a + d);
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  return {
    angle,
    logStretch: symmetricFunction([
      cosine * a + sine * c,
      ((cosine * b + sine * d) + (-sine * a + cosine * c)) / 2,
      -sine * b + cosine * d,
    ], Math.log),
    center: [width * (matrix[0] / 2 + matrix[1] / 2 + matrix[2]), height * (matrix[3] / 2 + matrix[4] / 2 + matrix[5])],
  };
}

function interpolateMatrix(left, right, mix, width, height) {
  const a = validateMatrix(left, width, height), b = validateMatrix(right, width, height);
  if (a.determinant * b.determinant <= 0) throw new Error("Cannot interpolate different image orientations.");
  const handedness = Math.sign(a.determinant);
  const first = polar(left, width, height, handedness), second = polar(right, width, height, handedness);
  const angleDelta = Math.atan2(Math.sin(second.angle - first.angle), Math.cos(second.angle - first.angle));
  const angle = first.angle + mix * angleDelta;
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  const [xx, xy, yy] = symmetricFunction(first.logStretch.map((value, index) => value + mix * (second.logStretch[index] - value)), Math.exp);
  const linear = [cosine * xx - sine * xy, (cosine * xy - sine * yy) * handedness,
    sine * xx + cosine * xy, (sine * xy + cosine * yy) * handedness];
  const center = first.center.map((value, index) => value + mix * (second.center[index] - value));
  const matrix = [linear[0], linear[1] * height / width, (center[0] - linear[0] * width / 2 - linear[1] * height / 2) / width,
    linear[2] * width / height, linear[3], (center[1] - linear[2] * width / 2 - linear[3] * height / 2) / height];
  validateMatrix(matrix, width, height);
  return matrix;
}

function validAnchor(anchor, width, height) {
  if (!anchor || !Number.isFinite(anchor.color_depth) || anchor.color_depth < 0 || anchor.color_depth > 1 ||
      !Number.isFinite(anchor.ct_frame) || anchor.ct_frame < 0) return false;
  try { validateMatrix(anchor.inverse_uv, width, height); return true; } catch { return false; }
}

/**
 * Absolute reviewed alignments never inherit an automatic transform or a legacy
 * scale/translation delta. Exact frames are authoritative; interpolation is
 * explicit and limited to short, uninterrupted spans with increasing CT depth.
 * maxFrame is the highest color frame index, so half a frame is .5/maxFrame.
 * profile.review_boundaries, when present, are additional blocked depth seams.
 * ctBoundaries are source CT frame seams; check these against reviewed CT
 * correspondence, independently of the automatic color-to-CT depth mapping.
 */
export function reviewAlignmentAt(profile, anchors, depth, options = {}) {
  if (!Number.isFinite(depth) || depth < 0 || depth > 1 || !Array.isArray(anchors)) return null;
  let width, height;
  try { [width, height] = dimensions(options); } catch { return null; }
  // Keep malformed absolute anchors in the ordered list so they block bridging.
  const ordered = anchors.filter((anchor) => Number.isFinite(anchor?.color_depth) && anchor.color_depth >= 0 && anchor.color_depth <= 1)
    .sort((a, b) => a.color_depth - b.color_depth);
  const exactTolerance = Number.isFinite(options.maxFrame) && options.maxFrame > 0 ? .5 / options.maxFrame : EPSILON;
  const exact = ordered.filter((anchor) => Math.abs(anchor.color_depth - depth) <= exactTolerance + EPSILON)
    .sort((a, b) => Math.abs(a.color_depth - depth) - Math.abs(b.color_depth - depth))[0];
  if (exact) {
    if (!validAnchor(exact, width, height)) return null;
    const duplicates = ordered.filter((anchor) => Math.abs(anchor.color_depth - exact.color_depth) <= EPSILON);
    if (duplicates.length > 1) return null;
    return { ...exact, inverse_uv: [...exact.inverse_uv], interpolated: false };
  }
  if (options.interpolate !== true) return null;
  const maxGap = options.maxGap ?? .04;
  if (!Number.isFinite(maxGap) || maxGap <= 0) return null;
  const rightIndex = ordered.findIndex((anchor) => anchor.color_depth > depth);
  if (rightIndex <= 0) return null;
  const left = ordered[rightIndex - 1], right = ordered[rightIndex];
  if (!validAnchor(left, width, height) || !validAnchor(right, width, height) ||
      ordered.some((anchor, index) => index !== rightIndex - 1 && Math.abs(anchor.color_depth - left.color_depth) <= EPSILON) ||
      ordered.some((anchor, index) => index !== rightIndex && Math.abs(anchor.color_depth - right.color_depth) <= EPSILON)) return null;
  const span = right.color_depth - left.color_depth;
  if (span <= EPSILON || span > maxGap + EPSILON || right.ct_frame <= left.ct_frame) return null;
  const boundaries = [...(Array.isArray(profile?.review_boundaries) ? profile.review_boundaries : []), ...(Array.isArray(options.boundaries) ? options.boundaries : [])];
  if (boundaries.some((boundary) => Number.isFinite(boundary) && boundary > left.color_depth + EPSILON && boundary <= right.color_depth + EPSILON)) return null;
  const ctBoundaries = Array.isArray(options.ctBoundaries) ? options.ctBoundaries : [];
  if (ctBoundaries.some((boundary) => Number.isFinite(boundary) && boundary > left.ct_frame && boundary <= right.ct_frame)) return null;
  const mix = (depth - left.color_depth) / span;
  try {
    return {
      color_depth: depth,
      ct_frame: left.ct_frame + mix * (right.ct_frame - left.ct_frame),
      inverse_uv: interpolateMatrix(left.inverse_uv, right.inverse_uv, mix, width, height),
      interpolated: true,
      source_anchor_ids: [left.id ?? left.slot ?? null, right.id ?? right.slot ?? null],
    };
  } catch { return null; }
}
