import { applyDisplayAdjustment, inverseUvToDisplayParameters } from "./viewer-math.mjs";

const EPSILON = 1e-9;
const ORIENTATIONS = {
  identity: [1, 0, 0, 0, 1, 0], flip_y: [1, 0, 0, 0, -1, 1],
  flip_x: [-1, 0, 1, 0, 1, 0], rotate_180: [-1, 0, 1, 0, -1, 1],
};

export function inverseAffine6(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 6 || !matrix.every(Number.isFinite)) return null;
  const [a, b, c, d, e, f] = matrix, determinant = a * e - b * d;
  if (Math.abs(determinant) < 1e-10) return null;
  return [e / determinant, -b / determinant, (b * f - e * c) / determinant,
    -d / determinant, a / determinant, (d * c - a * f) / determinant];
}

export function multiplyAffine6(left, right) {
  return [left[0] * right[0] + left[1] * right[3], left[0] * right[1] + left[1] * right[4], left[0] * right[2] + left[1] * right[5] + left[2],
    left[3] * right[0] + left[4] * right[3], left[3] * right[1] + left[4] * right[4], left[3] * right[2] + left[4] * right[5] + left[5]];
}

function bracket(values, position, key = (item) => item) {
  if (!values.length || !Number.isFinite(position) || position < key(values[0]) - EPSILON || position > key(values.at(-1)) + EPSILON) return null;
  let low = 0, high = values.length - 1;
  while (low < high) { const middle = (low + high) >>> 1; if (key(values[middle]) < position - EPSILON) low = middle + 1; else high = middle; }
  if (Math.abs(key(values[low]) - position) <= EPSILON) return { left: values[low], right: values[low], mix: 0, leftIndex: low, rightIndex: low };
  if (!low) return null;
  const left = values[low - 1], right = values[low];
  return { left, right, mix: (position - key(left)) / (key(right) - key(left)), leftIndex: low - 1, rightIndex: low };
}

/** Fractional native voxel index. CT frame numbers are not assumed uniform. */
export function ctFrameToIndex(sliceFrames, frame) {
  const result = bracket(sliceFrames, frame);
  return result ? result.leftIndex + (result.rightIndex - result.leftIndex) * result.mix : null;
}

function normalizeKnots(knots, maximumFrame, deleted = new Set()) {
  return (Array.isArray(knots) ? knots : []).filter((item) => !deleted.has(Number(item.color_frame ?? Math.round(item.color_depth * maximumFrame))))
    .map((item) => ({ ...item,
      color_frame: Number(item.color_frame ?? Math.round(item.color_depth * maximumFrame)),
      color_depth: Number(item.color_depth ?? item.color_frame / maximumFrame),
      ct_frame: Number(item.ct_frame ?? item.z_position),
    })).sort((a, b) => a.color_depth - b.color_depth);
}

function validKnot(item) {
  return item && !item.stale && Number.isFinite(item.color_depth) && Number.isFinite(item.ct_frame) && inverseAffine6(item.inverse_uv);
}

function strictlyIncreasing(knots, key) {
  return knots.every((item, index) => Number.isFinite(key(item)) && (!index || key(item) > key(knots[index - 1]) + EPSILON));
}

function matrixAt(left, right, mix, orientation, width, height) {
  if (mix <= EPSILON) return left;
  if (mix >= 1 - EPSILON) return right;
  const a = inverseUvToDisplayParameters(left, orientation, width, height), b = inverseUvToDisplayParameters(right, orientation, width, height);
  if (!a || !b || Math.sign(left[0] * left[4] - left[1] * left[3]) !== Math.sign(right[0] * right[4] - right[1] * right[3])) return null;
  const delta = Math.atan2(Math.sin((b.rotation_deg - a.rotation_deg) * Math.PI / 180), Math.cos((b.rotation_deg - a.rotation_deg) * Math.PI / 180)) * 180 / Math.PI;
  return applyDisplayAdjustment(ORIENTATIONS[orientation], {
    tx: a.x_position + (b.x_position - a.x_position) * mix,
    ty: a.y_position + (b.y_position - a.y_position) * mix,
    rotation_deg: a.rotation_deg + delta * mix,
    scale_x: Math.exp(Math.log(a.x_scale) + Math.log(b.x_scale / a.x_scale) * mix),
    scale_y: Math.exp(Math.log(a.y_scale) + Math.log(b.y_scale / a.y_scale) * mix),
  }, width, height);
}

function medianStep(values) {
  const steps = values.slice(1).map((value, index) => value - values[index]).filter((value) => value > 0).sort((a, b) => a - b);
  return steps.length ? steps[Math.floor((steps.length - 1) / 2)] : 1;
}

function restrictions(volume, source, profile, frames, maximumFrame) {
  const rgbSeams = [...(profile.review_boundaries || [])], ctSeams = [];
  const rgbGaps = [], ctGaps = [];
  const addRange = (list, lo, hi) => { if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) list.push([lo, hi]); };
  const addMissing = (values, output, divisor = 1) => {
    const expected = medianStep(values);
    for (let index = 1; index < values.length; index++) if (values[index] - values[index - 1] > expected * 2 + EPSILON) addRange(output, values[index - 1] / divisor, values[index] / divisor);
  };
  addMissing(frames, ctGaps);
  const rgbFrames = (source.rgb?.layers || []).map((item) => Number(item.frame ?? item.depth)).filter(Number.isFinite).sort((a, b) => a - b);
  addMissing(rgbFrames, rgbGaps, maximumFrame);
  for (const segment of volume.acquisition_segments || []) if (segment.start_index > 0 && Number.isFinite(frames[segment.start_index])) ctSeams.push(frames[segment.start_index]);
  for (const boundary of source.boundaries || []) {
    const reason = boundary.reason, applied = boundary.applied_transform;
    // The source manifest also contains rejected metric outliers. They are
    // review suggestions, not acquisition or geometry discontinuities.
    const acquisition = !reason || typeof reason !== "object" || boundary.gap ||
      ["metadata_gap", "position_jump", "acquisition_change", "fov_change"].some((key) => reason[key]);
    const corrected = applied && ["tx", "ty", "rotation_deg", "scale"].some((key) =>
      Number.isFinite(applied[key]) && Math.abs(applied[key] - (key === "scale" ? 1 : 0)) > EPSILON);
    if (!acquisition && !corrected) continue;
    const value = Number(boundary.right ?? boundary.boundary);
    if (Number.isFinite(value)) (boundary.modality === "density" || boundary.modality === "ct" ? ctSeams : rgbSeams).push(value / (boundary.modality === "density" || boundary.modality === "ct" ? 1 : maximumFrame));
  }
  for (const gap of source.gaps || []) {
    const isCt = gap.modality === "density" || gap.modality === "ct", divisor = isCt ? 1 : maximumFrame;
    addRange(isCt ? ctGaps : rgbGaps, Number(gap.left ?? gap.start_frame ?? gap.start ?? gap[0]) / divisor, Number(gap.right ?? gap.end_frame ?? gap.end ?? gap[1]) / divisor);
  }
  for (const range of profile.unsupported_ranges || []) addRange(rgbGaps, Number(range[0]), Number(range[1]));
  return { rgbSeams, ctSeams, rgbGaps, ctGaps };
}

function inGap(value, ranges) { return ranges.some(([low, high]) => value > low + EPSILON && value < high - EPSILON); }
function crosses(low, high, seams) { return seams.some((value) => value > low + EPSILON && value <= high + EPSILON); }
function overlaps(low, high, ranges) { return ranges.some(([a, b]) => high > a + EPSILON && low < b - EPSILON); }

function allowed(result, limits, maxGap, preserveExisting = false) {
  if (!result || !validKnot(result.left) || !validKnot(result.right)) return false;
  const { left, right } = result;
  if (left === right) return !inGap(left.color_depth, limits.rgbGaps) && !inGap(left.ct_frame, limits.ctGaps);
  return right.color_depth - left.color_depth <= maxGap + EPSILON && right.ct_frame > left.ct_frame &&
    (preserveExisting || (!(left.segment && right.segment && left.segment !== right.segment) &&
      !crosses(left.color_depth, right.color_depth, limits.rgbSeams) && !crosses(left.ct_frame, right.ct_frame, limits.ctSeams))) &&
    !overlaps(left.color_depth, right.color_depth, limits.rgbGaps) && !overlaps(left.ct_frame, right.ct_frame, limits.ctGaps);
}

/**
 * Live lookup from the current CT grid into the RGB volume baked by
 * build_rgb_volume.py. Each native CT slice has two vec4 rows:
 * [a,b,c,bakedZ, d,e,f,continuityGroup]. XY maps CT UV -> baked RGB UV. Z is a
 * normalized native voxel index, independent of the currently loaded LOD.
 * Invalid rows have continuityGroup=0. Positive integers identify uninterrupted
 * source regions. Fractional sampling requires BOTH neighboring groups to be
 * positive and equal. Exact integer sampling requires only that row. Never
 * fall back to stale identity or interpolate through a gap/acquisition seam.
 *
 * The baked transform was coefficient-linear. Corrections interpolate display
 * rotation/translation/log-scale relative to that original interpolation,
 * keeping unchanged regions exactly unchanged and every edited knot exact.
 * When an original interior knot was deleted, use the new endpoint curve
 * directly so the deleted transform cannot survive in the baked geometry.
 */
export function buildVolumeAlignmentLut({ volumeManifest = {}, rgbManifest = {}, knots = [], draft = null,
  profile = {}, sourceManifest = {}, orientation = profile.orientation || "flip_y", maxGap = .04, width = 512, height = 304 } = {}) {
  const depth = Number(volumeManifest.dimensions?.[2]) || 0;
  const data = new Float32Array(Math.max(0, depth) * 8);
  const output = { data, depth, enabled: false, reason: null, validCount: 0, invalidCount: depth, blockedCorrectionCount: 0, preservedCount: 0, selected: null };
  const fail = (reason) => { output.reason = reason; return output; };
  const frames = volumeManifest.slice_frames || Array.from({ length: depth }, (_, index) => index);
  if (!depth || frames.length !== depth || !strictlyIncreasing(frames, (value) => value)) return fail("invalid_ct_geometry");
  const maximumFrame = Math.max(1, Number(sourceManifest.colourPlanes) - 1 || 0,
    ...(sourceManifest.rgb?.layers || []).map((item) => Number(item.frame ?? item.depth) || 0),
    ...knots.map((item) => Number(item.color_frame) || 0), ...(rgbManifest.alignment_knots || []).map((item) => Number(item.color_frame) || 0));
  const deleted = new Set((profile.deleted_knot_frames || []).map(Number));
  const current = normalizeKnots(knots, maximumFrame, deleted);
  const draftFrame = Number(draft?.color_frame);
  if (draft && !deleted.has(draftFrame)) {
    const index = current.findIndex((item) => item.color_frame === draftFrame);
    if (index >= 0) current[index] = normalizeKnots([{ ...current[index], ...draft }], maximumFrame)[0];
  }
  const limits = restrictions(volumeManifest, sourceManifest, profile, frames, maximumFrame);
  if (draft) {
    const selected = normalizeKnots([draft], maximumFrame)[0], index = ctFrameToIndex(frames, selected.ct_frame), inverseUv = inverseAffine6(selected.inverse_uv);
    const matching = current.findIndex((item) => item.color_frame === draftFrame);
    const monotonic = matching >= 0 && (!matching || current[matching - 1].ct_frame < selected.ct_frame) &&
      (matching === current.length - 1 || current[matching + 1].ct_frame > selected.ct_frame);
    const valid = !!(validKnot(selected) && index !== null && inverseUv && monotonic && allowed({ left: selected, right: selected }, limits, maxGap));
    output.selected = { colorFrame: selected.color_frame, ctFrame: selected.ct_frame, index,
      z: index === null ? null : index / Math.max(1, depth - 1), inverseUv, valid, reason: valid ? null : "invalid_selected_knot" };
  }
  if (!ORIENTATIONS[orientation]) return fail("invalid_orientation");
  if (!Number.isFinite(maxGap) || maxGap <= 0) return fail("invalid_interpolation_span");
  if (!Array.isArray(rgbManifest.alignment_knots) || !rgbManifest.alignment_knots.length) return fail("missing_alignment_snapshot");
  if (rgbManifest.status === "stale" || volumeManifest.status === "stale" || rgbManifest.live_alignment?.available === false) return fail("stale_source_geometry");
  if (rgbManifest.dimensions?.some((value, index) => value !== volumeManifest.dimensions[index]) ||
      (rgbManifest.subject && volumeManifest.subject && rgbManifest.subject !== volumeManifest.subject)) return fail("incompatible_rgb_volume");
  if (!current.length) return fail("no_current_knots");
  const baked = normalizeKnots(rgbManifest.alignment_knots, maximumFrame);
  if (!baked.every(validKnot) || !strictlyIncreasing(baked, (item) => item.ct_frame) || !strictlyIncreasing(baked, (item) => item.color_depth)) return fail("invalid_alignment_snapshot");
  if (!strictlyIncreasing(current, (item) => item.color_depth) || !strictlyIncreasing(current, (item) => item.ct_frame)) return fail("non_monotonic_z");
  const bakedAt = (colorDepth, preserveExisting = false) => {
    const support = bracket(baked, colorDepth, (item) => item.color_depth);
    if (!allowed(support, limits, preserveExisting ? Infinity : maxGap, preserveExisting)) return null;
    const { left, right, mix } = support;
    const matrix = left.inverse_uv.map((value, index) => value + (right.inverse_uv[index] - value) * mix);
    return inverseAffine6(matrix) ? { matrix, ctFrame: left.ct_frame + (right.ct_frame - left.ct_frame) * mix } : null;
  };
  const references = current.map((item) => bakedAt(item.color_depth));
  const retained = new Set(current.map((item) => item.color_frame));
  const continuityGroups = new Map();
  const side = (value, seams) => seams.reduce((count, seam) => count + (seam <= value + EPSILON ? 1 : 0), 0);
  const groupFor = (ctFrame, colorDepth, oldCtFrame, segment) => {
    const key = [side(ctFrame, limits.ctSeams), side(oldCtFrame, limits.ctSeams), side(colorDepth, limits.rgbSeams),
      side(ctFrame, limits.ctGaps.map((range) => range[1])), side(oldCtFrame, limits.ctGaps.map((range) => range[1])),
      side(colorDepth, limits.rgbGaps.map((range) => range[1])), segment || ""].join("|");
    if (!continuityGroups.has(key)) continuityGroups.set(key, continuityGroups.size + 1);
    return continuityGroups.get(key);
  };
  const matchesBaked = (item, original) => validKnot(item) && original && item.color_frame === original.color_frame &&
    Math.abs(item.ct_frame - original.ct_frame) < EPSILON && item.inverse_uv.every((value, index) => Math.abs(value - original.inverse_uv[index]) < EPSILON);
  for (let index = 0; index < depth; index++) {
    const support = bracket(current, frames[index], (item) => item.ct_frame);
    if (!support) continue;
    const { left, right, mix, leftIndex, rightIndex } = support;
    const colorDepth = left.color_depth + (right.color_depth - left.color_depth) * mix;
    const originalSupport = bracket(baked, colorDepth, (item) => item.color_depth);
    const preserveExisting = matchesBaked(left, originalSupport?.left) && matchesBaked(right, originalSupport?.right);
    const normallyAllowed = allowed(support, limits, maxGap);
    if (!allowed(support, limits, preserveExisting ? Infinity : maxGap, preserveExisting)) { if (!preserveExisting) output.blockedCorrectionCount++; continue; }
    const old = bakedAt(colorDepth, preserveExisting), referenceLeft = references[leftIndex], referenceRight = references[rightIndex];
    if (!old || !referenceLeft || !referenceRight) continue;
    const bakedIndex = ctFrameToIndex(frames, old.ctFrame);
    if (bakedIndex === null || inGap(old.ctFrame, limits.ctGaps)) continue;
    // A Z correction must not carry imagery from another acquisition segment.
    if (crosses(Math.min(old.ctFrame, frames[index]), Math.max(old.ctFrame, frames[index]), limits.ctSeams) ||
        overlaps(Math.min(old.ctFrame, frames[index]), Math.max(old.ctFrame, frames[index]), limits.ctGaps)) continue;
    const transform = matrixAt(left.inverse_uv, right.inverse_uv, mix, orientation, width, height);
    const inverse = inverseAffine6(transform);
    if (!inverse) continue;
    const removedInside = baked.some((item) => item.color_depth > left.color_depth + EPSILON && item.color_depth < right.color_depth - EPSILON && !retained.has(item.color_frame));
    const reference = removedInside ? old.matrix : matrixAt(referenceLeft.matrix, referenceRight.matrix, mix, orientation, width, height);
    if (!reference) continue;
    const remap = multiplyAffine6(reference, inverse);
    if (!inverseAffine6(remap)) continue;
    data.set([remap[0], remap[1], remap[2], bakedIndex / Math.max(1, depth - 1), remap[3], remap[4], remap[5], groupFor(frames[index], colorDepth, old.ctFrame, left.segment)], index * 8);
    output.validCount++;
    if (preserveExisting && !normallyAllowed) output.preservedCount++;
  }
  output.enabled = true;
  output.invalidCount = depth - output.validCount;
  output.reason = !output.validCount ? "no_supported_alignment" : output.blockedCorrectionCount ? "some_corrections_unsupported" : null;
  return output;
}
