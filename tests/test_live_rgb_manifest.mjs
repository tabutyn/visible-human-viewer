import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const SERVER = fileURLToPath(new URL("../server.mjs", import.meta.url));

const sha = (value) => createHash("sha256").update(value).digest("hex");
const snapshot = [
  { color_frame: 0, color_depth: 0, z_position: 7, inverse_uv: [1, 0, 0, 0, -1, 1] },
  { color_frame: 10, color_depth: 1, z_position: 17, inverse_uv: [1.2, 0, -.1, 0, -1, 1] },
];

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "vh-live-rgb-"));
  const processed = join(root, "Processed", "v1"), subject = join(processed, "male");
  await mkdir(join(subject, "volume-v1"), { recursive: true });
  await mkdir(join(subject, "rgb-volume-v1"), { recursive: true });
  const source = JSON.stringify({ subject: "male", rgb: { layers: [{ frame: 0 }, { frame: 10 }] } });
  const corrections = JSON.stringify({ version: 1 });
  const volume = JSON.stringify({ dimensions: [2, 2, 11], slice_frames: Array.from({ length: 11 }, (_, i) => i + 7), inputs: { corrections_sha256: sha(corrections) } });
  const alignment = JSON.stringify({ version: 1, name: "original baseline" });
  const candidate = JSON.stringify({ version: 2, name: "original candidate" });
  const reviewed = JSON.stringify({ anchors: [] });
  const inputs = { "manifest.json": source, "corrections.json": corrections, "volume-v1/manifest.json": volume, "alignment.json": alignment, "alignment-candidate-v2.json": candidate, "reviewed-alignment.json": reviewed };
  const manifest = { version: 1, format: "rgb8-planar", alignment_knots: snapshot, inputs: { source_manifest_sha256: sha(source), ct_volume_manifest_sha256: sha(volume), alignment_sha256: sha(alignment), candidate_sha256: sha(candidate), reviewed_sha256: sha(reviewed) }, levels: [] };
  const saveManifest = (value) => writeFile(join(subject, "rgb-volume-v1/manifest.json"), JSON.stringify(value));
  const reset = async () => {
    await Promise.all(Object.entries(inputs).map(([name, contents]) => writeFile(join(subject, name), contents)));
    await saveManifest(manifest);
  };
  await reset();
  const child = spawn(process.execPath, [SERVER], { env: { ...process.env, HOST: "127.0.0.1", PORT: String(48000 + Math.floor(Math.random() * 1000)), VISIBLE_HUMAN_ROOT: root, VISIBLE_HUMAN_PROCESSED_ROOT: processed } });
  t.after(() => child.kill());
  const base = await new Promise((resolve, reject) => {
    let stderr = "", stdout = "";
    const timer = setTimeout(() => reject(new Error(`Server startup timed out: ${stderr}`)), 5000);
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.stdout.on("data", (chunk) => { stdout += chunk; const match = stdout.match(/http:\/\/127\.0\.0\.1:\d+/); if (match) { clearTimeout(timer); resolve(match[0]); } });
    child.once("error", (error) => { clearTimeout(timer); reject(error); });
    child.once("exit", (code) => { clearTimeout(timer); reject(new Error(`Server exited (${code}): ${stderr}`)); });
  });
  const get = async (live = true) => {
    const response = await fetch(`${base}/api/subjects/male/volume/rgb/manifest${live ? "?live=1" : ""}`);
    return { status: response.status, body: await response.json() };
  };
  return { subject, manifest, inputs, reset, saveManifest, get };
}

test("live RGB endpoint exposes an immutable, validated build snapshot", async (t) => {
  const { subject, manifest, get } = await fixture(t);
  const before = await readFile(join(subject, "rgb-volume-v1/manifest.json"));
  const response = await get();
  assert.equal(response.status, 200);
  assert.equal(response.body.status, "ready");
  assert.deepEqual(response.body.alignment_knots, snapshot);
  assert.deepEqual(response.body.inputs, manifest.inputs);
  assert.deepEqual(response.body.live_alignment, { available: true, changed_inputs: [] });
  assert.deepEqual(await readFile(join(subject, "rgb-volume-v1/manifest.json")), before);
  assert.equal((await get(false)).body.live_alignment, undefined);
});

test("only alignment changes are remappable and the standard route keeps rejecting stale RGB", async (t) => {
  const f = await fixture(t);
  for (const [file, key] of [["alignment.json", "alignment_sha256"], ["alignment-candidate-v2.json", "candidate_sha256"], ["reviewed-alignment.json", "reviewed_sha256"]]) {
    await f.reset();
    await writeFile(join(f.subject, file), `${f.inputs[file]}\n`);
    const response = await f.get();
    assert.equal(response.status, 200, file);
    assert.equal(response.body.status, "remappable");
    assert.deepEqual(response.body.live_alignment, { available: true, changed_inputs: [key] });
    assert.deepEqual(response.body.alignment_knots, snapshot);
    assert.deepEqual(response.body.inputs, f.manifest.inputs);
    assert.equal((await f.get(false)).status, 409);
  }
  await f.reset();
  await f.saveManifest({ ...f.manifest, inputs: { ...f.manifest.inputs, reviewed_sha256: null } });
  assert.equal((await f.get()).body.status, "remappable", "a newly created review file must not be mistaken for current RGB");
});

test("live RGB rejects changed or unverified geometry, including CT corrections", async (t) => {
  const f = await fixture(t);
  for (const name of ["manifest.json", "volume-v1/manifest.json", "corrections.json"]) {
    await f.reset();
    await writeFile(join(f.subject, name), `${f.inputs[name]}\n`);
    const response = await f.get();
    assert.equal(response.status, 409, name);
    assert.equal(response.body.live_alignment.available, false);
    assert.match(response.body.error, /geometry changed/);
  }
  for (const key of ["source_manifest_sha256", "ct_volume_manifest_sha256"]) {
    await f.reset();
    await f.saveManifest({ ...f.manifest, inputs: { ...f.manifest.inputs, [key]: null } });
    const response = await f.get();
    assert.equal(response.status, 409);
    assert.match(response.body.error, /no source geometry hashes/);
  }
});

test("live RGB rejects broken snapshot ordering, depth values, and affine matrices", async (t) => {
  const f = await fixture(t);
  for (const knots of [
    [], [snapshot[0]],
    [snapshot[1], snapshot[0]],
    [snapshot[0], { ...snapshot[1], z_position: 7 }],
    [snapshot[0], { ...snapshot[1], color_depth: 0 }],
    [snapshot[0], { ...snapshot[1], color_frame: 0 }],
    [snapshot[0], { ...snapshot[1], z_position: "NaN" }],
    [snapshot[0], { ...snapshot[1], inverse_uv: [1, 1, 0, 1, 1, 0] }],
    [snapshot[0], { ...snapshot[1], inverse_uv: [1, 0, 0, 0, 1, 0] }],
    [snapshot[0], { ...snapshot[1], inverse_uv: [1, 0, 0, 0, null, 1] }],
  ]) {
    await f.saveManifest({ ...f.manifest, alignment_knots: knots });
    const response = await f.get();
    assert.equal(response.status, 409, JSON.stringify(knots));
    assert.equal(response.body.live_alignment.available, false);
  }
});

test("current legacy RGB may display without a snapshot, but editing stale legacy RGB is denied", async (t) => {
  const f = await fixture(t);
  const legacy = structuredClone(f.manifest);
  delete legacy.alignment_knots;
  await f.saveManifest(legacy);
  const response = await f.get();
  assert.equal(response.status, 200);
  assert.equal(response.body.status, "ready");
  assert.equal(response.body.live_alignment.available, false);
  assert.match(response.body.live_alignment.reason, /no usable alignment snapshot/);
  await writeFile(join(f.subject, "reviewed-alignment.json"), '{"anchors":[{}]}');
  assert.equal((await f.get()).status, 409);
});
