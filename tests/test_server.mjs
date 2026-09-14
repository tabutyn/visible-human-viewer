import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

const SERVER = fileURLToPath(new URL("../server.mjs", import.meta.url));

async function start(root, processed) {
  const port = 46000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, [SERVER], { env: { ...process.env, PORT: String(port), VISIBLE_HUMAN_ROOT: root, VISIBLE_HUMAN_PROCESSED_ROOT: processed, VISIBLE_HUMAN_PYTHON: "/usr/bin/true" } });
  await new Promise((resolve, reject) => { const timer=setTimeout(()=>reject(new Error("server timeout")),3000); child.stdout.on("data",()=>{clearTimeout(timer);resolve();}); child.once("error",reject); });
  return { child, base: `http://127.0.0.1:${port}` };
}

test("subject, layer, review, and global job routes", async () => {
  const root = await mkdtemp(join(tmpdir(), "vh-server-")); const processed = join(root, "Processed", "v1");
  const color = join(root, "Male", "Fullcolor", "fullbody"); const frozen = join(root, "Male", "Radiological", "frozenCT", "png");
  await mkdir(color, { recursive: true }); await mkdir(frozen, { recursive: true });
  await writeFile(join(color, "a_vm1001.raw"), Buffer.alloc(8)); await writeFile(join(frozen, "cvm1006f.png"), Buffer.from("not-a-png"));
  const { child, base } = await start(root, processed);
  try {
    const page = await (await fetch(base)).text(); assert.match(page, /Knot \/ slice editor/); assert.match(page, /calibration-scale_x/); assert.match(page, /Medical Rendering/);
    const subjects = await (await fetch(`${base}/api/subjects`)).json(); assert.equal(subjects[0].id, "male");
    const manifest = await (await fetch(`${base}/api/subjects/male/manifest`)).json(); assert.equal(manifest.rgb.layers[0].frame, 0); assert.equal(manifest.density.layers[0].frame, 5);
    assert.equal((await fetch(`${base}/api/subjects/male/rgb/layers/77`)).status, 404);
    const saved = await fetch(`${base}/api/review/male/boundaries/rgb-5`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ transform: { tx: 1, ty: 2, scale: 1, rotation_deg: 0 } }) }); assert.equal(saved.status, 200);
    const review = await (await fetch(`${base}/api/review/male/boundaries/rgb-5`)).json(); assert.equal(review.transform.tx, 1);
    let calibration = await (await fetch(`${base}/api/review/male/calibration`)).json(); assert.deepEqual(calibration.anchors, []);
    const anchorBody = { color_depth: .11, color_frame: 206, scale_x: 1.2, scale_y: .9, tx: 4, ty: -3 };
    const anchor = await (await fetch(`${base}/api/review/male/calibration/anchors/2`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(anchorBody) })).json();
    assert.equal(anchor.slot, 2); assert.equal(anchor.scale_x, 1.2);
    assert.equal((await fetch(`${base}/api/review/male/calibration/anchors/3`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ ...anchorBody, scale_x: 99 }) })).status, 400);
    calibration = await (await fetch(`${base}/api/review/male/calibration`)).json(); assert.equal(calibration.anchors[0].tx, 4);
    await fetch(`${base}/api/review/male/calibration/anchors/2`, { method: "DELETE" });
    calibration = await (await fetch(`${base}/api/review/male/calibration`)).json(); assert.deepEqual(calibration.anchors, []);
    const volume = join(processed, "male", "volume-v1"), brick = Buffer.from([0,4,100,4]), volumeCorrections=Buffer.from('{"version":1}');
    await mkdir(join(volume, "level-0"), { recursive: true });
    await writeFile(join(processed,"male","corrections.json"),volumeCorrections);
    await writeFile(join(volume, "level-0", "000-000-000.u16le"), brick);
    await writeFile(join(volume, "manifest.json"), JSON.stringify({ version: 1, format: "u16le-hu-plus-1024", brick_size: 2, inputs:{corrections_sha256:createHash("sha256").update(volumeCorrections).digest("hex")}, levels: [{ level: 0, dimensions: [2,1,1], grid: [1,1,1], bricks: [{ x: 0, y: 0, z: 0, extent: [2,1,1], file: "level-0/000-000-000.u16le", sha256: "test-brick" }] }] }));
    const volumeManifest = await (await fetch(`${base}/api/subjects/male/volume/manifest`)).json(); assert.equal(volumeManifest.levels[0].dimensions[0], 2);
    const volumeBrick = await fetch(`${base}/api/subjects/male/volume/bricks/0/0/0/0`); assert.deepEqual(Buffer.from(await volumeBrick.arrayBuffer()), brick); assert.equal(volumeBrick.headers.get("etag"), 'W/"test-brick"');
    assert.equal((await fetch(`${base}/api/subjects/male/volume/bricks/0/0/0/1`)).status, 404);
    await writeFile(join(processed,"male","corrections.json"),Buffer.from('{"version":2}'));assert.equal((await fetch(`${base}/api/subjects/male/volume/manifest`)).status,409);
    const volumeJob = await (await fetch(`${base}/api/subjects/male/volume/build`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ memory_limit_gib: 8 }) })).json(); assert.equal(volumeJob.kind, "volume");
    const job = await (await fetch(`${base}/api/review/male/rebuild`, { method: "POST" })).json();
    const status = await (await fetch(`${base}/api/review/jobs/${job.id}`)).json(); assert.ok(["running", "complete"].includes(status.state));
  } finally { child.kill(); }
});

test("registration profile is exposed only while its inputs match", async () => {
  const root = await mkdtemp(join(tmpdir(), "vh-alignment-")), processed = join(root, "Processed", "v1"), subject = join(processed, "male");
  await mkdir(subject, { recursive: true });
  const manifest = JSON.stringify({ baked: true, rgb: { width: 2048, height: 1216, layers: [{ frame: 0, file: "rgb/000000.png" }, { frame: 1, file: "rgb/000001.png" }, { frame: 2, file: "rgb/000002.png" }] }, density: { width: 1024, height: 608, layers: [{ frame: 0, file: "density/000000.u16be" }, { frame: 1, file: "density/000001.u16be" }, { frame: 2, file: "density/000002.u16be" }] }, boundaries: [], gaps: [] });
  const corrections = JSON.stringify({ version: 1, transforms: {} });
  const sha = (value) => createHash("sha256").update(value).digest("hex");
  const alignment = JSON.stringify({ version: 1, orientation: "flip_y", inputs: { manifest_sha256: sha(manifest), corrections_sha256: sha(corrections) }, coverage: { color_depth: [0, 1] }, depth_knots: [{ color_depth: 0, ct_frame: 0 }], spatial_knots: [{ color_depth: 0, inverse_uv: [1, 0, 0, 0, -1, 1], confidence: 1 }] });
  await writeFile(join(subject, "manifest.json"), manifest); await writeFile(join(subject, "corrections.json"), corrections); await writeFile(join(subject, "alignment.json"), alignment);
  await writeFile(join(subject, "alignment-candidate-v2.json"), JSON.stringify({ version: 2, status: "needs_review", inputs: { manifest_sha256: sha(manifest), corrections_sha256: sha(corrections), baseline_alignment_sha256: sha(alignment) }, qc: { accepted: false, promotion_blockers: ["adaptive_z_review_incomplete"] }, z_review_queue: [{ color_frame: 0, recommended_ct_frame: 0, status: "pending" }], control_review_frames: [0], parameter_knots: [{ color_frame: 0, color_depth: 0, z_position: 0, confidence: 1 }, { color_frame: 1, color_depth: .5, z_position: 1, confidence: .1 }, { color_frame: 2, color_depth: 1, z_position: 2, confidence: .1 }], coverage: { color_depth: [0, 1] }, depth_knots: [{ color_depth: 0, ct_frame: 0 }], spatial_knots: [{ color_depth: 0, inverse_uv: [1, 0, 0, 0, -1, 1], confidence: 1 }] }));
  const { child, base } = await start(root, processed);
  try {
    let response = await (await fetch(`${base}/api/subjects/male/manifest`)).json(); assert.equal(response.registrationStatus, "ready"); assert.equal(response.registration.orientation, "flip_y"); assert.equal(response.registrationCandidateStatus, "needs_review");
    assert.deepEqual(response.registrationCandidate.control_review_queue.map((item) => item.color_frame), [0]);
    assert.equal((await fetch(`${base}/api/review/male/candidate`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "promote" }) })).status, 409);
    assert.equal((await fetch(`${base}/api/review/male/candidate`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ color_frame: 0, status: "rejected" }) })).status, 200);
    const reviews = await (await fetch(`${base}/api/review/male/frames`)).json();
    const body = { ct_frame: 0, inverse_uv: [1, 0, 0, 0, -1, 1], parameters: { z_position: 0, x_position: 0, y_position: 0, rotation_deg: 0, x_scale: 1, y_scale: 1 }, landmarks: [], inputs: reviews.inputs };
    const save = (value) => fetch(`${base}/api/review/male/frames/0`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(value) });
    assert.equal((await save(body)).status, 200);
    assert.equal((await fetch(`${base}/api/review/male/frames/1`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })).status, 400);
    const zOnly = { ...body, ct_frame: 1, parameters: { ...body.parameters, z_position: 1 }, reviewed_fields: ["z_position"], z_confirmed: true };
    assert.equal((await fetch(`${base}/api/review/male/frames/1`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(zOnly) })).status, 200);
    const changedCandidate = await (await fetch(`${base}/api/review/male/candidate`)).json();
    assert.equal(changedCandidate.candidate.status, "needs_rebuild");
    assert.ok(changedCandidate.candidate.qc.promotion_blockers.includes("reviewed_anchors_changed"));
    assert.equal((await save({ ...body, inverse_uv: [1, 1, 0, 1, 1, 0] })).status, 400);
    assert.equal((await save({ ...body, inverse_uv: [20, 0, 0, 0, .05, 0] })).status, 400);
    assert.equal((await save({ ...body, ct_frame: 10 })).status, 400);
    assert.equal((await save({ ...body, parameters: { ...body.parameters, z_position: 1 } })).status, 400);
    assert.equal((await save({ ...body, inputs: {} })).status, 409);
    const saved = await (await fetch(`${base}/api/review/male/frames`)).json();
    assert.deepEqual(saved.anchors[0].inverse_uv, body.inverse_uv); assert.equal(saved.anchors[0].stale, false);
    assert.deepEqual(saved.anchors[1].reviewed_fields, ["z_position"]); assert.equal(saved.anchors[1].z_confirmed, true);
    const profilePath = join(subject, "alignment.json");
    const profile = await import("node:fs/promises").then((fs) => fs.readFile(profilePath, "utf8"));
    await writeFile(profilePath, `${profile}\n`);
    const stale = await (await fetch(`${base}/api/review/male/frames`)).json(); assert.equal(stale.anchors[0].stale, true);
    await writeFile(join(subject, "corrections.json"), `${corrections}\n`);
    response = await (await fetch(`${base}/api/subjects/male/manifest`)).json(); assert.equal(response.registrationStatus, "stale"); assert.equal(response.registration, null);
    assert.equal((await fetch(`${base}/api/review/male/frames/0`, { method: "DELETE" })).status, 200);
    const deleted = await (await fetch(`${base}/api/review/male/candidate`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ action: "delete_knot", color_frame: 1 }) })).json();
    assert.equal(deleted.deleted_frame, 1); assert.deepEqual(deleted.deleted_knot_frames, [1]);
    assert.deepEqual((await (await fetch(`${base}/api/review/male/frames`)).json()).anchors, []);
  } finally { child.kill(); }
});
