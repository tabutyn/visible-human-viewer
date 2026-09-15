export const GIB = 1024 ** 3;
export const HARD_MEMORY_LIMIT = 8 * GIB;
export const TARGET_MEMORY_LIMIT = 6 * GIB;
export const MAX_VOLUME_BYTES = 4 * GIB;

export function volumeBytes(dimensions, bytesPerVoxel = 2) {
  if (!Array.isArray(dimensions) || dimensions.length !== 3) throw new TypeError("dimensions must contain x, y, and z");
  return dimensions.reduce((value, dimension) => value * Math.max(0, Number(dimension) || 0), bytesPerVoxel);
}

export function chooseVolumeLevel(levels, maxBytes = MAX_VOLUME_BYTES, maxTextureDimension = 2048) {
  return [...levels]
    .sort((left, right) => left.level - right.level)
    .find((level) => volumeBytes(level.dimensions) <= maxBytes && level.dimensions.every((value) => value <= maxTextureDimension)) || null;
}

export function displayBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MiB";
  return bytes >= GIB ? `${(bytes / GIB).toFixed(2)} GiB` : `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}

export function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

export function interpolation(points, x, component) {
  const values = [...points].sort((left, right) => left.hu - right.hu);
  if (!values.length) return component === "opacity" ? 0 : [0, 0, 0];
  if (x <= values[0].hu) return values[0][component];
  if (x >= values.at(-1).hu) return values.at(-1)[component];
  const rightIndex = values.findIndex((point) => point.hu >= x);
  const left = values[rightIndex - 1], right = values[rightIndex];
  const mix = (x - left.hu) / Math.max(1e-9, right.hu - left.hu);
  if (Array.isArray(left[component])) return left[component].map((value, index) => value + (right[component][index] - value) * mix);
  return left[component] + (right[component] - left[component]) * mix;
}

export function transferLut(opacityPoints, colorPoints, size = 4096, huMinimum = -1024) {
  const output = new Uint8Array(size * 4);
  for (let index = 0; index < size; index++) {
    const hu = huMinimum + index;
    const color = interpolation(colorPoints, hu, "color");
    output[index * 4] = Math.round(clamp(color[0], 0, 1) * 255);
    output[index * 4 + 1] = Math.round(clamp(color[1], 0, 1) * 255);
    output[index * 4 + 2] = Math.round(clamp(color[2], 0, 1) * 255);
    output[index * 4 + 3] = Math.round(clamp(interpolation(opacityPoints, hu, "opacity"), 0, 1) * 255);
  }
  return output;
}

export function brickOpacity(bricks, lut, huMinimum = -1024) {
  return Uint8Array.from(bricks, (brick) => {
    // Older immutable public manifests omit per-brick HU bounds. An unknown
    // brick must be visited, not silently culled from the volume ray.
    if (!validHuBounds(brick)) return 255;
    const start = clamp(Math.floor(brick.min_hu - huMinimum), 0, lut.length / 4 - 1);
    const end = clamp(Math.ceil(brick.max_hu - huMinimum), 0, lut.length / 4 - 1);
    let maximum = 0;
    for (let index = start; index <= end; index++) maximum = Math.max(maximum, lut[index * 4 + 3]);
    return maximum;
  });
}

function validHuBounds(brick) {
  return Number.isFinite(brick?.min_hu) && Number.isFinite(brick?.max_hu) && brick.min_hu <= brick.max_hu;
}

export function brickDensityMask(bricks, density) {
  return Uint8Array.from(bricks, (brick) =>
    !validHuBounds(brick) || (brick.min_hu <= density && density <= brick.max_hu) ? 255 : 0,
  );
}

export function planeAspect(axis, dimensions, spacing) {
  const physical = dimensions.map((value, index) => value * spacing[index]);
  if (axis === 0) return physical[1] / physical[2];
  if (axis === 1) return physical[0] / physical[2];
  return physical[0] / physical[1];
}

export function canvasPlaneScale(axis, dimensions, spacing, canvasAspect) {
  const imageAspect = planeAspect(axis, dimensions, spacing);
  return canvasAspect > imageAspect ? [imageAspect / canvasAspect, 1] : [1, canvasAspect / imageAspect];
}
