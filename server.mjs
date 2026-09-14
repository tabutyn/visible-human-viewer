import http from "node:http";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { access, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { alignmentReviewRoutes } from "./review-server.mjs";
import { validateReviewMatrix } from "./review-math.mjs";
import { sendCompressedFile, transferManifest, webpAlternative } from "./delivery.mjs";

const PORT = Number(process.env.PORT || 4173);
const HOST = process.env.HOST || "127.0.0.1";
const READ_ONLY = process.env.VISIBLE_HUMAN_MODE === "public";
const PUBLIC_SUBJECTS = new Set((process.env.VISIBLE_HUMAN_PUBLIC_SUBJECTS || "male").split(","));
if (!["127.0.0.1", "localhost", "::1"].includes(HOST) && !READ_ONLY) throw new Error("Non-loopback serving requires VISIBLE_HUMAN_MODE=public (read-only).");
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_ROOT = path.resolve(process.env.VISIBLE_HUMAN_ROOT || path.join(os.homedir(), "Visible-Human-Project"));
const PROCESSED_ROOT = path.resolve(process.env.VISIBLE_HUMAN_PROCESSED_ROOT || path.join(DATA_ROOT, "Processed", "v1"));
const DELIVERY_ROOT = path.resolve(process.env.VISIBLE_HUMAN_DELIVERY_ROOT || path.join(path.dirname(path.dirname(PROCESSED_ROOT)), "Delivery", "v1"));
const GE_HEADER_BYTES = 3416;
const jobs = new Map(); let jobSerial = 0;
const reviewRoutes = alignmentReviewRoutes({ here: HERE, processedRoot: PROCESSED_ROOT, jobs, sendJson, requestBody });

function sendJson(res, status, value) { const body = JSON.stringify(value); res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", "Content-Length": Buffer.byteLength(body) }); res.end(body); }
function safePart(value) { return typeof value === "string" && /^[a-z0-9_-]+$/i.test(value) ? value : null; }
async function exists(value) { try { await access(value); return true; } catch { return false; } }
function title(subject) { return subject[0].toUpperCase() + subject.slice(1); }
function rgbDepth(subject, name) { const m = name.match(/a_?v[mf](\d{4})([abc]?)\.raw$/i); if (!m) return null; const n = Number(m[1]) - 1001; return subject === "female" ? n * 3 + ({ a: 0, b: 1, c: 2 }[m[2].toLowerCase()] ?? 0) : n; }
function ctDepth(subject, name) { const frozen = name.match(/cvm(\d{4})f\.png$/i); if (subject === "male" && frozen) return Number(frozen[1]) - 1001; const m = name.match(/c_v[mf](\d{4})\.fre$/i); if (!m) return null; const n = Number(m[1]) - 1001; return subject === "female" ? n * 3 : n; }

async function rawLayers(subject, modality) {
  const source = title(subject); const rgb = modality === "rgb";
  const base = rgb ? path.join(DATA_ROOT, source, "Fullcolor", "fullbody") : subject === "male" ? path.join(DATA_ROOT, source, "Radiological", "frozenCT", "png") : path.join(DATA_ROOT, source, "Radiological", "normalCT", "extracted");
  if (!await exists(base)) return [];
  return (await readdir(base)).filter((name) => name.toLowerCase().endsWith(rgb ? ".raw" : subject === "male" ? ".png" : ".fre"))
    .map((name) => ({ frame: rgb ? rgbDepth(subject, name) : ctDepth(subject, name), path: path.join(base, name), name, raw: true }))
    .filter((item) => Number.isInteger(item.frame)).sort((a, b) => a.frame - b.frame);
}
async function readManifest(subject) { const file = path.join(PROCESSED_ROOT, subject, "manifest.json"); if (!await exists(file)) return null; return JSON.parse(await readFile(file, "utf8")); }
async function readVolumeManifest(subject, validate = false) {
  const file = path.join(PROCESSED_ROOT, subject, "volume-v1", "manifest.json");
  if (!await exists(file)) return null;
  try {
    const value = JSON.parse(await readFile(file, "utf8")); value.status = "ready";
    const expected = value.inputs?.corrections_sha256, corrections = path.join(PROCESSED_ROOT, subject, "corrections.json");
    if (validate && expected && (!await exists(corrections) || createHash("sha256").update(await readFile(corrections)).digest("hex") !== expected)) value.status = "stale";
    return value;
  } catch { return null; }
}
function rgbSnapshotError(knots) {
  if (!Array.isArray(knots) || knots.length < 2) return "The RGB volume has no usable alignment snapshot. Rebuild it to edit its alignment live.";
  let previous = null, orientation = null;
  for (const knot of knots) {
    if (!knot || !Number.isInteger(knot.color_frame) || knot.color_frame < 0 || !Number.isFinite(knot.color_depth) || knot.color_depth < 0 || knot.color_depth > 1 || !Number.isFinite(knot.z_position)) return "The RGB alignment snapshot contains invalid frame or depth values. Rebuild it.";
    if (previous && (knot.color_frame <= previous.color_frame || knot.color_depth <= previous.color_depth || knot.z_position <= previous.z_position)) return "The RGB alignment snapshot is not strictly increasing. Rebuild it.";
    try { validateReviewMatrix(knot.inverse_uv); } catch (error) { return `The RGB alignment snapshot is invalid: ${error.message}`; }
    const matrix = knot.inverse_uv, determinant = matrix[0] * matrix[4] - matrix[1] * matrix[3];
    if (orientation !== null && determinant * orientation <= 0) return "The RGB alignment snapshot reverses its image orientation. Rebuild it.";
    orientation = determinant;
    previous = knot;
  }
  return null;
}
async function readRgbVolumeManifest(subject, validate = false, live = false) {
  const directory = path.join(PROCESSED_ROOT, subject), file = path.join(directory, "rgb-volume-v1", "manifest.json");
  if (!await exists(file)) return null;
  try {
    const value = JSON.parse(await readFile(file, "utf8")); value.status = "ready";
    if (validate) {
      const checks = [
        ["source_manifest_sha256", path.join(directory, "manifest.json")],
        ["ct_volume_manifest_sha256", path.join(directory, "volume-v1", "manifest.json")],
        ["alignment_sha256", path.join(directory, "alignment.json")],
        ["candidate_sha256", path.join(directory, "alignment-candidate-v2.json")],
        ["reviewed_sha256", path.join(directory, "reviewed-alignment.json")],
      ];
      const changedInputs = [], missingGeometry = [];
      for (const [key, input] of checks) {
        const expected = value.inputs?.[key];
        if (!live) {
          if (expected && (!await exists(input) || createHash("sha256").update(await readFile(input)).digest("hex") !== expected)) value.status = "stale";
          continue;
        }
        const actual = await exists(input) ? createHash("sha256").update(await readFile(input)).digest("hex") : null;
        if ((expected ?? null) !== actual) changedInputs.push(key);
        if ((key === "source_manifest_sha256" || key === "ct_volume_manifest_sha256") && !expected) missingGeometry.push(key);
      }
      if (live) {
        const geometryChanged = changedInputs.some((key) => key === "source_manifest_sha256" || key === "ct_volume_manifest_sha256");
        const ctManifest = await readVolumeManifest(subject, true);
        let reason = null;
        if (missingGeometry.length) reason = "The RGB volume has no source geometry hashes. Rebuild it before editing alignment live.";
        else if (geometryChanged || !ctManifest || ctManifest.status !== "ready") reason = "The RGB source or CT geometry changed. Rebuild the volumes before editing alignment live.";
        const snapshotError = rgbSnapshotError(value.alignment_knots);
        // A current legacy volume can still be displayed, but it cannot be remapped.
        const missingSnapshot = value.alignment_knots === undefined || value.alignment_knots === null;
        if (!reason && snapshotError && (!missingSnapshot || changedInputs.length)) reason = snapshotError;
        value.status = reason ? "stale" : changedInputs.length ? "remappable" : "ready";
        value.live_alignment = {
          available: !reason && !snapshotError,
          changed_inputs: changedInputs,
          ...(reason || snapshotError ? { reason: reason || snapshotError } : {}),
        };
      }
    }
    return value;
  } catch { return null; }
}
async function serveVolumeBrick(req, res, subject, levelNumber, x, y, z, modality = "ct") {
  const manifest = modality === "rgb" ? await readRgbVolumeManifest(subject) : await readVolumeManifest(subject);
  const level = manifest?.levels?.find((item) => item.level === levelNumber);
  const brick = level?.bricks?.find((item) => item.x === x && item.y === y && item.z === z);
  if (!brick) return sendJson(res, 404, { error: "Volume brick unavailable", level: levelNumber, x, y, z });
  const volumeDirectory = path.join(PROCESSED_ROOT, subject, modality === "rgb" ? "rgb-volume-v1" : "volume-v1");
  const file = path.resolve(volumeDirectory, brick.file);
  if (!file.startsWith(`${volumeDirectory}${path.sep}`) || !await exists(file)) return sendJson(res, 404, { error: "Volume brick file unavailable" });
  const tag = `W/"${brick.sha256}"`;
  if (req.headers["if-none-match"] === tag) { res.writeHead(304, { ETag: tag, Vary:"Accept-Encoding" }); res.end(); return; }
  return sendCompressedFile(req, res, file, {
    "Content-Type": "application/octet-stream",
    "Cache-Control": "private, no-cache",
    ETag: tag,
    "X-Visible-Human-Format": modality === "rgb" ? "RGB8-PLANAR" : "U16LE",
    "X-Visible-Human-Extent": brick.extent.join(","),
  }, { processedRoot: PROCESSED_ROOT, deliveryRoot: DELIVERY_ROOT });
}
async function readAlignment(subject) {
  const directory = path.join(PROCESSED_ROOT, subject), profilePath = path.join(directory, "alignment.json"), manifestPath = path.join(directory, "manifest.json"), correctionsPath = path.join(directory, "corrections.json");
  if (!await exists(profilePath)) return { profile: null, status: "missing" };
  try {
    const profile = JSON.parse(await readFile(profilePath, "utf8")), expected = profile.inputs || {};
    if (!expected.manifest_sha256 || !expected.corrections_sha256) return { profile: null, status: "invalid" };
    const digest = async (file) => createHash("sha256").update(await readFile(file)).digest("hex");
    const [manifestSha, correctionsSha] = await Promise.all([digest(manifestPath), digest(correctionsPath)]);
    if (manifestSha !== expected.manifest_sha256 || correctionsSha !== expected.corrections_sha256) return { profile: null, status: "stale" };
    return { profile, status: "ready" };
  } catch { return { profile: null, status: "invalid" }; }
}
async function readCandidate(subject) {
  const directory = path.join(PROCESSED_ROOT, subject), candidatePath = path.join(directory, "alignment-candidate-v2.json");
  if (!await exists(candidatePath)) return { profile: null, status: "missing" };
  try {
    const profile = JSON.parse(await readFile(candidatePath, "utf8")), expected = profile.inputs || {};
    const digest = async (file) => createHash("sha256").update(await readFile(file)).digest("hex");
    const [manifestSha, correctionsSha, alignmentSha] = await Promise.all([digest(path.join(directory, "manifest.json")), digest(path.join(directory, "corrections.json")), digest(path.join(directory, "alignment.json"))]);
    if (manifestSha !== expected.manifest_sha256 || correctionsSha !== expected.corrections_sha256 || alignmentSha !== expected.baseline_alignment_sha256) return { profile: null, status: "stale" };
    const manifest = JSON.parse(await readFile(path.join(directory, "manifest.json"), "utf8"));
    let reviewed = { anchors: [] };
    try { reviewed = JSON.parse(await readFile(path.join(directory, "reviewed-alignment.json"), "utf8")); } catch {}
    const confirmed = new Map((reviewed.anchors || []).filter((item) => item.z_confirmed !== false).map((item) => [item.color_frame, item]));
    const skipped = new Set(profile.control_review_skips || []);
    const maxFrame = Math.max(1, ...manifest.rgb.layers.map((item) => item.frame));
    const deleted = new Set((profile.deleted_knot_frames || []).map(Number));
    const knotFrame = (item) => Number.isFinite(item.color_frame) ? item.color_frame : Math.round(item.color_depth * maxFrame);
    profile.parameter_knots = (profile.parameter_knots || []).filter((item) => !deleted.has(knotFrame(item)));
    profile.spatial_knots = (profile.spatial_knots || []).filter((item) => !deleted.has(knotFrame(item)));
    profile.local_solutions = (profile.local_solutions || []).filter((item) => !deleted.has(knotFrame(item)));
    profile.z_review_queue = (profile.z_review_queue || []).filter((item) => !deleted.has(item.color_frame));
    if (profile.parameter_knots.length) {
      profile.z_match_knots ||= profile.depth_knots;
      profile.depth_knots = profile.parameter_knots.map((item) => ({ color_depth: item.color_depth, ct_frame: item.z_position, confidence: item.confidence, segment: item.segment }));
    }
    const depthKnots = [...(profile.depth_knots || [])].sort((a,b) => a.color_depth-b.color_depth);
    const zAt = (depth) => {
      if (!depthKnots.length) return depth * maxFrame;
      const right = depthKnots.findIndex((item) => item.color_depth >= depth);
      if (right <= 0) return depthKnots[0].ct_frame;
      if (right < 0) return depthKnots.at(-1).ct_frame;
      const left=depthKnots[right-1], next=depthKnots[right], mix=(depth-left.color_depth)/(next.color_depth-left.color_depth);
      return left.ct_frame+mix*(next.ct_frame-left.ct_frame);
    };
    const queue = new Map();
    for (const item of profile.parameter_knots || profile.spatial_knots || []) {
      const frame = Number.isFinite(item.color_frame) ? item.color_frame : Math.round(item.color_depth * maxFrame);
      const saved = confirmed.get(frame);
      queue.set(frame, { color_frame: frame, color_depth: item.color_depth, recommended_ct_frame: saved?.ct_frame ?? item.z_position ?? zAt(item.color_depth),
        status: saved ? "confirmed" : skipped.has(frame) ? "skipped" : "pending", kind: "curve control point", confidence: item.confidence });
    }
    for (const item of profile.z_review_queue || []) {
      const saved = confirmed.get(item.color_frame), existing = queue.get(item.color_frame) || {};
      queue.set(item.color_frame, { ...existing, ...item, recommended_ct_frame: saved?.ct_frame ?? item.recommended_ct_frame,
        status: saved ? "confirmed" : skipped.has(item.color_frame) ? "skipped" : item.status, kind: existing.kind ? `${existing.kind} + Z check` : "Z check" });
    }
    const campaign = Array.isArray(profile.control_review_frames) ? new Set(profile.control_review_frames.map(Number)) : null;
    profile.control_review_queue = [...queue.values()]
      .filter((item) => item.status === "confirmed" || !campaign || campaign.has(item.color_frame))
      .sort((a,b) => a.color_frame-b.color_frame);
    return { profile, status: profile.status || "needs_review" };
  } catch { return { profile: null, status: "invalid" }; }
}
function normalise(subject, source, rgb, density, alignment = { profile: null, status: "missing" }, candidate = { profile: null, status: "missing" }) {
  const rgbLayers = source?.rgb?.layers || rgb; const densityLayers = source?.density?.layers || density;
  const planes = Math.max(-1, ...rgbLayers.map((x) => x.frame ?? x.depth)) + 1;
  return { version: 1, subject, colourPlanes: planes, depthRange: [0, Math.max(0, planes - 1)], processed: Boolean(source?.baked),
    rgb: { width: source?.rgb?.width || 2048, height: source?.rgb?.height || 1216, format: source?.rgb?.format || "planar-rgb-raw", layers: rgbLayers.map((x) => ({ ...x, frame: x.frame ?? x.depth, path: undefined })) },
    density: { width: source?.density?.width || (source ? 1024 : 512), height: source?.density?.height || (source ? 608 : 512), format: source?.density?.format || "u16be-hu-plus-1024", layers: densityLayers.map((x) => ({ ...x, frame: x.frame ?? x.depth, path: undefined })) },
    boundaries: source?.boundaries || [], gaps: source?.gaps || [], registration: alignment.profile, registrationStatus: alignment.status,
    registrationCandidate: candidate.profile, registrationCandidateStatus: candidate.status, _rgb: rgbLayers, _density: densityLayers };
}
async function model(subject) { const manifest = await readManifest(subject); const profiles = manifest ? await Promise.all([readAlignment(subject), readCandidate(subject)]) : [];
  return normalise(subject, manifest, manifest ? [] : await rawLayers(subject, "rgb"), manifest ? [] : await rawLayers(subject, "density"), profiles[0], profiles[1]); }
async function allSubjects() { const out = []; for (const id of ["male", "female"]) { const x = await model(id); if (x.rgb.layers.length || x.density.layers.length) out.push({ id, label: title(id), colourPlanes: x.colourPlanes, processed: x.processed, registrationStatus: x.registrationStatus }); } return out; }
async function serveLayer(req, res, subject, modality, frame, sourceOnly = false) {
  const value = await model(subject);
  const items = sourceOnly ? await rawLayers(subject, modality) : modality === "rgb" ? value._rgb : value._density;
  const item = items.find((x) => (x.frame ?? x.depth) === frame);
  if (!item) return sendJson(res, 404, { error: "Layer unavailable", frame, modality });
  const file = item.path || path.resolve(PROCESSED_ROOT, subject, item.file || item.name || "");
  if (!file.startsWith(DATA_ROOT) && !file.startsWith(PROCESSED_ROOT)) return sendJson(res, 400, { error: "Unsafe layer path" });
  const info = await stat(file); const isFre = file.endsWith(".fre"); const isPng = file.endsWith(".png"); const isRaw = item.raw || file.endsWith(".raw") || isFre; const offset = isFre ? GE_HEADER_BYTES : 0;
  // Browsers decode PNG into 8-bit canvas pixels. Decode official frozen
  // 16-bit PNG to the same U16BE wire format as a baked density slice instead.
  if (modality === "density" && isPng) {
    const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(HERE, ".venv", "bin", "python");
    const code = "import cv2,numpy as n,sys; x=cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED); assert x is not None and x.dtype==n.uint16; sys.stdout.buffer.write(x.astype('>u2',copy=False).tobytes())";
    const child = spawn(python, ["-c", code, file], { stdio: ["ignore", "pipe", "pipe"] });
    let failure = ""; child.stderr.on("data", (chunk) => { failure += chunk.toString(); });
    child.once("error", (error) => { if (!res.headersSent) sendJson(res, 503, { error: `PNG16 decoder unavailable: ${error.message}` }); });
    child.once("close", (exitCode) => { if (exitCode && !res.writableEnded) res.destroy(new Error(failure || "PNG16 decoder failed")); });
    res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Length": 512 * 512 * 2, "X-Visible-Human-Width": "512", "X-Visible-Human-Height": "512", "X-Visible-Human-Format": "U16BE" });
    child.stdout.pipe(res); return;
  }
  const headers = { "Content-Type": isPng ? "image/png" : "application/octet-stream", "Cache-Control": "private, no-cache",
    "X-Visible-Human-Width": String(sourceOnly ? (modality === "rgb" ? 2048 : 512) : modality === "rgb" ? value.rgb.width : value.density.width), "X-Visible-Human-Height": String(sourceOnly ? (modality === "rgb" ? 1216 : 512) : modality === "rgb" ? value.rgb.height : value.density.height),
    "X-Visible-Human-Format": modality === "rgb" ? (isRaw ? "RGB-24-bit-planar" : "PNG") : (isFre ? "GE-16-bit-big-endian" : "U16BE") };
  if(modality === "rgb" && isPng && !sourceOnly){
    const alternative=await webpAlternative(file,{processedRoot:PROCESSED_ROOT,deliveryRoot:DELIVERY_ROOT});
    if(alternative){res.writeHead(200,{...headers,"Content-Type":"image/webp","X-Visible-Human-Format":"WEBP","Content-Length":alternative.bytes});return createReadStream(alternative.file).pipe(res);}
  }
  return sendCompressedFile(req,res,file,headers,{processedRoot:PROCESSED_ROOT,deliveryRoot:DELIVERY_ROOT,offset,compress:!isPng});
}
async function overrideFile(subject) { return path.join(PROCESSED_ROOT, subject, "overrides.json"); }
async function getOverrides(subject) { const file = await overrideFile(subject); return await exists(file) ? JSON.parse(await readFile(file, "utf8")) : { version: 1, boundaries: {} }; }
async function saveOverrides(subject, value) { const file = await overrideFile(subject); await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, `${JSON.stringify(value, null, 2)}\n`); }
async function calibrationFile(subject) { return path.join(PROCESSED_ROOT, subject, "manual-alignment.json"); }
async function getCalibration(subject) {
  const file = await calibrationFile(subject);
  if (!await exists(file)) return { version: 1, subject, anchors: [] };
  try {
    const value = JSON.parse(await readFile(file, "utf8"));
    return { version: 1, subject, anchors: Array.isArray(value.anchors) ? value.anchors : [] };
  } catch { return { version: 1, subject, anchors: [] }; }
}
async function saveCalibration(subject, value) {
  const file = await calibrationFile(subject); await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, `${JSON.stringify({ version: 1, subject, anchors: value.anchors }, null, 2)}\n`);
}
function calibrationAnchor(slot, body) {
  const values = {
    slot,
    color_depth: Number(body.color_depth),
    color_frame: Number(body.color_frame),
    scale_x: Number(body.scale_x),
    scale_y: Number(body.scale_y),
    tx: Number(body.tx),
    ty: Number(body.ty),
  };
  if (!Number.isInteger(slot) || slot < 1 || slot > 10) return null;
  if (!Number.isFinite(values.color_depth) || values.color_depth < 0 || values.color_depth > 1) return null;
  if (!Number.isInteger(values.color_frame) || values.color_frame < 0) return null;
  if (!Number.isFinite(values.scale_x) || values.scale_x < .25 || values.scale_x > 4) return null;
  if (!Number.isFinite(values.scale_y) || values.scale_y < .25 || values.scale_y > 4) return null;
  if (!Number.isFinite(values.tx) || Math.abs(values.tx) > 512 || !Number.isFinite(values.ty) || Math.abs(values.ty) > 304) return null;
  return { ...values, updated_at: new Date().toISOString() };
}
async function getBoundary(subject, id) {
  const manifest = await readManifest(subject); const source = manifest?.boundaries?.find((item) => String(item.id || item.boundary) === id) || { id };
  const override = (await getOverrides(subject)).boundaries[id];
  return override ? { ...source, ...override, transform: override.transform || source.transform } : source;
}
async function makeJob(subject, boundary) {
  const id = String(++jobSerial); const job = { id, subject, state: "queued", progress: 0, startedAt: new Date().toISOString(), log: [] }; jobs.set(id, job);
  const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(HERE, ".venv", "bin", "python");
  const manifest = await readManifest(subject);
  const args = [path.join(HERE, "preprocess.py"), "--root", DATA_ROOT, "--output", path.join(PROCESSED_ROOT, subject), "--subject", subject];
  if (boundary && manifest?.baked) args.push("--rebuild-boundary", boundary); else args.push("--bake");
  try {
    const child = spawn(python, args, { stdio: ["ignore", "pipe", "pipe"] }); job.state = "running";
    const collect = (chunk) => { job.log.push(chunk.toString().slice(0, 2000)); job.log = job.log.slice(-10); };
    child.stdout.on("data", collect); child.stderr.on("data", collect);
    child.on("error", (error) => Object.assign(job, { state: "failed", error: error.message, finishedAt: new Date().toISOString() }));
    child.on("close", (code) => Object.assign(job, { state: code === 0 ? "complete" : "failed", progress: code === 0 ? 1 : 0, exitCode: code, finishedAt: new Date().toISOString() }));
  } catch (error) { Object.assign(job, { state: "failed", error: error.message, finishedAt: new Date().toISOString() }); }
  return job;
}
async function makeVolumeJob(subject, memoryLimitGib = 8) {
  const id = String(++jobSerial);
  const job = { id, subject, kind: "volume", state: "queued", progress: 0, startedAt: new Date().toISOString(), log: [] };
  jobs.set(id, job);
  const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(HERE, ".venv", "bin", "python");
  const args = [path.join(HERE, "build_volume.py"), "--root", DATA_ROOT, "--output", path.join(PROCESSED_ROOT, subject), "--subject", subject, "--memory-limit-gib", String(memoryLimitGib)];
  try {
    const child = spawn(python, args, { stdio: ["ignore", "pipe", "pipe"] }); job.state = "running";
    const collect = (chunk) => {
      const text = chunk.toString(); job.log.push(text.slice(0, 2000)); job.log = job.log.slice(-10);
      for (const line of text.split("\n")) try {
        const event = JSON.parse(line); if (event.event === "progress") job.progress = event.total ? event.completed / event.total : 0;
      } catch {}
    };
    child.stdout.on("data", collect); child.stderr.on("data", collect);
    child.on("error", (error) => Object.assign(job, { state: "failed", error: error.message, finishedAt: new Date().toISOString() }));
    child.on("close", (code) => Object.assign(job, { state: code === 0 ? "complete" : "failed", progress: code === 0 ? 1 : job.progress, exitCode: code, finishedAt: new Date().toISOString() }));
  } catch (error) { Object.assign(job, { state: "failed", error: error.message, finishedAt: new Date().toISOString() }); }
  return job;
}
async function makeRgbVolumeJob(subject) {
  const id = String(++jobSerial);
  const job = { id, subject, kind: "rgb-volume", state: "queued", progress: 0, startedAt: new Date().toISOString(), log: [] };
  jobs.set(id, job);
  const python = process.env.VISIBLE_HUMAN_PYTHON || path.join(HERE, ".venv", "bin", "python");
  const args = [path.join(HERE, "build_rgb_volume.py"), "--processed", PROCESSED_ROOT, "--subject", subject];
  try {
    const child = spawn(python, args, { stdio: ["ignore", "pipe", "pipe"] }); job.state = "running";
    const collect = (chunk) => {
      const text = chunk.toString(); job.log.push(text.slice(-2000)); job.log = job.log.slice(-10);
      for (const line of text.split("\n")) try { const event = JSON.parse(line); if (event.event === "progress") job.progress = event.total ? event.completed / event.total : 0; } catch {}
    };
    child.stdout.on("data", collect); child.stderr.on("data", collect);
    child.on("error", (error) => Object.assign(job, { state: "failed", error: error.message, finishedAt: new Date().toISOString() }));
    child.on("close", (code) => Object.assign(job, { state: code === 0 ? "complete" : "failed", progress: code === 0 ? 1 : job.progress, exitCode: code, finishedAt: new Date().toISOString() }));
  } catch (error) { Object.assign(job, { state: "failed", error: error.message, finishedAt: new Date().toISOString() }); }
  return job;
}
async function requestBody(req) { const chunks = []; for await (const piece of req) chunks.push(piece); return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"); }
function defaultMedicalView() { return { preset: "lung", yaw: -Math.PI/2, pitch: 0, zoom: 1.15, cameraPan: [0, 0], planeOffset: 0, planeNormal: [0, 0, 1] }; }
function validVector(value, length) { return Array.isArray(value) && value.length === length && value.every(Number.isFinite); }
function normalizeMedicalView(value) {
  const fallback=defaultMedicalView(),preset=["bone","soft","lung","body","mip","custom"].includes(value?.preset)?value.preset:fallback.preset;
  return {
    preset,
    yaw:Number.isFinite(value?.yaw)?value.yaw:fallback.yaw,
    pitch:Number.isFinite(value?.pitch)?Math.max(-1.45,Math.min(1.45,value.pitch)):fallback.pitch,
    zoom:Number.isFinite(value?.zoom)?Math.max(.3,Math.min(8,value.zoom)):fallback.zoom,
    cameraPan:validVector(value?.cameraPan,2)?value.cameraPan:fallback.cameraPan,
    planeOffset:Number.isFinite(value?.planeOffset)?Math.max(-1,Math.min(1,value.planeOffset)):fallback.planeOffset,
    planeNormal:validVector(value?.planeNormal,3)&&Math.hypot(...value.planeNormal)>.0001?value.planeNormal.map((item)=>item/Math.hypot(...value.planeNormal)):fallback.planeNormal,
    updatedAt:new Date().toISOString(),
  };
}
async function medicalView(subject) {
  const file=path.join(PROCESSED_ROOT,subject,"medical-view.json");
  try{return normalizeMedicalView(JSON.parse(await readFile(file,"utf8")));}catch{return defaultMedicalView();}
}
async function saveMedicalView(subject,value) {
  const normalized=normalizeMedicalView(value),file=path.join(PROCESSED_ROOT,subject,"medical-view.json");
  await mkdir(path.dirname(file),{recursive:true});await writeFile(file,`${JSON.stringify(normalized,null,2)}\n`);return normalized;
}

async function route(req, res) {
  const url = new URL(req.url, `http://${req.headers.host || HOST}`); const p = url.pathname.split("/").filter(Boolean);
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  if(READ_ONLY && !["GET","HEAD"].includes(req.method))return sendJson(res,403,{error:"This release is read-only. Editing, saving, and build jobs are disabled."});
  if(READ_ONLY && p[0]==="api" && ["subjects","review"].includes(p[1]) && p[2] && (!PUBLIC_SUBJECTS.has(p[2])||p.includes("jobs")))return sendJson(res,404,{error:"Unavailable in this release"});
  if(req.method === "GET" && url.pathname === "/api/config")return sendJson(res,200,{readOnly:READ_ONLY,automaticVolume:process.env.VISIBLE_HUMAN_AUTO_DETAIL === "native" && !READ_ONLY ? "native" : "preview",consentVersion:1});
  if (req.method === "GET" && url.pathname === "/api/subjects") return sendJson(res, 200, (await allSubjects()).filter(item=>!READ_ONLY||PUBLIC_SUBJECTS.has(item.id)));
  if (req.method === "GET" && ["/review-math.mjs", "/review-workflow.mjs"].includes(url.pathname)) {
    const body = await readFile(path.join(HERE, url.pathname.slice(1)));
    res.writeHead(200, { "Content-Type": "text/javascript; charset=utf-8", "Cache-Control": "no-store" }); res.end(body); return;
  }
  if (p[0] === "api" && p[1] === "review" && p[2] === "jobs" && req.method === "GET") return sendJson(res, jobs.has(p[3]) ? 200 : 404, jobs.get(p[3]) || { error: "Job not found" });
  if (p[0] === "api" && p[1] === "subjects" && safePart(p[2])) {
    if (p[3] === "medical-view" && req.method === "GET") return sendJson(res, 200, await medicalView(p[2]));
    if (p[3] === "medical-view" && req.method === "PUT") return sendJson(res, 200, await saveMedicalView(p[2],await requestBody(req)));
    if (req.method === "GET" && p[3] === "manifest") { const value = await model(p[2]); delete value._rgb; delete value._density; return sendJson(res, 200, value); }
    if (req.method === "GET" && (p[3] === "rgb" || p[3] === "density") && p[4] === "layers" && /^\d+$/.test(p[5] || "")) return serveLayer(req, res, p[2], p[3], Number(p[5]), url.searchParams.get("source") === "raw");
    if (p[3] === "volume" && req.method === "GET" && p[4] === "manifest") {
      const value = await readVolumeManifest(p[2], true);
      if (!value) return sendJson(res, 404, { error: "Medical volume has not been built" });
      return value.status === "stale" ? sendJson(res, 409, { error: "Medical volume is stale because CT corrections changed. Rebuild it." }) : sendJson(res, 200, await transferManifest(value,p[2],"volume-v1",{processedRoot:PROCESSED_ROOT,deliveryRoot:DELIVERY_ROOT}));
    }
    if (p[3] === "volume" && p[4] === "rgb" && req.method === "GET" && p[5] === "manifest") {
      const live = url.searchParams.get("live") === "1";
      const value = await readRgbVolumeManifest(p[2], true, live);
      if (!value) return sendJson(res, 404, { error: "Registered RGB volume has not been built" });
      return value.status === "stale" ? sendJson(res, 409, { error: live ? value.live_alignment.reason : "Registered RGB volume is stale because its alignment inputs changed. Rebuild it.", ...(live ? { status: value.status, live_alignment: value.live_alignment } : {}) }) : sendJson(res, 200, await transferManifest(value,p[2],"rgb-volume-v1",{processedRoot:PROCESSED_ROOT,deliveryRoot:DELIVERY_ROOT}));
    }
    if (p[3] === "volume" && p[4] === "rgb" && req.method === "GET" && p[5] === "bricks" && p.length === 10 && p.slice(6, 10).every((value) => /^\d+$/.test(value || ""))) {
      return serveVolumeBrick(req, res, p[2], ...p.slice(6, 10).map(Number), "rgb");
    }
    if (p[3] === "volume" && p[4] === "rgb" && p[5] === "build" && req.method === "POST") return sendJson(res, 202, await makeRgbVolumeJob(p[2]));
    if (p[3] === "volume" && req.method === "GET" && p[4] === "bricks" && p.length === 9 && p.slice(5, 9).every((value) => /^\d+$/.test(value || ""))) {
      return serveVolumeBrick(req, res, p[2], ...p.slice(5, 9).map(Number));
    }
    if (p[3] === "volume" && p[4] === "build" && req.method === "POST") {
      const body = await requestBody(req), memoryLimitGib = Number(body.memory_limit_gib ?? 8);
      if (!Number.isFinite(memoryLimitGib) || memoryLimitGib < 1 || memoryLimitGib > 8) return sendJson(res, 400, { error: "memory_limit_gib must be between 1 and 8" });
      return sendJson(res, 202, await makeVolumeJob(p[2], memoryLimitGib));
    }
  }
  if (p[0] === "api" && p[1] === "review" && safePart(p[2])) {
    if (await reviewRoutes(req, res, p)) return;
    const subject = p[2], boundary = safePart(p[4]);
    if (p[3] === "calibration") {
      const value = await getCalibration(subject);
      if (req.method === "GET" && !p[4]) return sendJson(res, 200, value);
      const slot = Number(p[5]);
      if (p[4] === "anchors" && req.method === "PUT") {
        const anchor = calibrationAnchor(slot, await requestBody(req));
        if (!anchor) return sendJson(res, 400, { error: "Invalid calibration anchor" });
        value.anchors = value.anchors.filter((item) => item.slot !== slot).concat(anchor).sort((a, b) => a.slot - b.slot);
        await saveCalibration(subject, value); return sendJson(res, 200, anchor);
      }
      if (p[4] === "anchors" && req.method === "DELETE" && Number.isInteger(slot) && slot >= 1 && slot <= 10) {
        value.anchors = value.anchors.filter((item) => item.slot !== slot);
        await saveCalibration(subject, value); return sendJson(res, 200, value);
      }
    }
    if (p[3] === "boundaries" && boundary) { const value = await getOverrides(subject); if (req.method === "GET") return sendJson(res, 200, await getBoundary(subject, boundary)); if (req.method === "PUT") { value.boundaries[boundary] = { id: boundary, ...(await requestBody(req)), updatedAt: new Date().toISOString() }; await saveOverrides(subject, value); return sendJson(res, 200, value.boundaries[boundary]); } }
    if (p[3] === "rebuild" && req.method === "POST") { const body = await requestBody(req); return sendJson(res, 202, await makeJob(subject, safePart(body.boundary))); }
    if (p[3] === "jobs" && req.method === "GET") return sendJson(res, jobs.has(p[4]) ? 200 : 404, jobs.get(p[4]) || { error: "Job not found" });
  }
  const staticFiles = { "/": ["index.html", "text/html; charset=utf-8"], "/index.html": ["index.html", "text/html; charset=utf-8"], "/viewer.js": ["viewer.js", "text/javascript; charset=utf-8"], "/viewer-math.mjs": ["viewer-math.mjs", "text/javascript; charset=utf-8"], "/medical-renderer.mjs": ["medical-renderer.mjs", "text/javascript; charset=utf-8"], "/volume-math.mjs": ["volume-math.mjs", "text/javascript; charset=utf-8"], "/volume-alignment.mjs": ["volume-alignment.mjs", "text/javascript; charset=utf-8"], "/viewer.css": ["viewer.css", "text/css; charset=utf-8"] };
  staticFiles["/entry.mjs"]=["entry.mjs","text/javascript; charset=utf-8"];
  staticFiles["/release-data.mjs"]=["release-data.mjs","text/javascript; charset=utf-8"];
  staticFiles["/LICENSE"]=["LICENSE","text/plain; charset=utf-8"];
  staticFiles["/NOTICE.md"]=["NOTICE.md","text/plain; charset=utf-8"];
  if (req.method === "GET" && staticFiles[url.pathname]) { const [name, type] = staticFiles[url.pathname]; const file = path.join(HERE, name); const info = await stat(file); res.writeHead(200, { "Content-Type": type, "Content-Length": info.size, "Cache-Control": "no-store" }); return createReadStream(file).pipe(res); }
  res.writeHead(404); res.end("Not Found");
}
http.createServer((req, res) => route(req, res).catch((error) => { console.error(error); if (!res.headersSent) sendJson(res, 500, { error: error.message }); else res.destroy(error); })).listen(PORT, HOST, () => console.log(`Visible Human Viewer: http://${HOST}:${PORT}`));
