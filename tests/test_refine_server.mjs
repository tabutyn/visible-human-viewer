import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";
import { alignmentReviewRoutes } from "../review-server.mjs";
import { applyDisplayAdjustment } from "../viewer-math.mjs";

const FLIP_Y = [1, 0, 0, 0, -1, 1];
const sha = (value) => createHash("sha256").update(value).digest("hex");
const adjustedMatrix = applyDisplayAdjustment(FLIP_Y, {
  tx: 3, ty: -2, scale_x: 1.04, scale_y: .98, rotation_deg: 2,
});

// Exercise the real spawn/stdin/stdout protocol without decoding patient data.
// VISIBLE_HUMAN_PYTHON points at Node, which runs this temporary .py fixture.
const childSource = `
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { setTimeout: delay } = require("node:timers/promises");
(async () => {
  const fixture = __dirname;
  const config = JSON.parse(await fs.readFile(path.join(fixture, "scenario.json"), "utf8"));
  const args = process.argv.slice(2);
  assert.equal(args.length, 4);
  assert.equal(args[0], "--processed");
  assert.equal(args[2], "--color-frame");
  assert.match(args[3], /^\\d+$/);
  assert.ok(["male", "female"].includes(path.basename(args[1])));
  for (const name of ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTHONDONTWRITEBYTECODE"]) {
    assert.equal(process.env[name], "1");
  }
  let input = "";
  for await (const chunk of process.stdin) input += chunk;
  const body = JSON.parse(input);
  assert.deepEqual(Object.keys(body).sort(), ["ct_frame", "inputs", "inverse_uv", "landmarks", "orientation", "z_bounds"]);
  assert.equal(body.z_bounds.length, 2);
  await fs.appendFile(path.join(fixture, "calls.jsonl"), JSON.stringify({ args, body }) + "\\n");
  if (config.hold) {
    let released = false;
    for (let i = 0; i < 500; i++) {
      try { await fs.stat(path.join(fixture, "release")); released = true; break; }
      catch (error) { if (error.code !== "ENOENT") throw error; }
      await delay(10);
    }
    assert.ok(released, "test did not release solver fixture");
  }
  process.stdout.write(config.raw_stdout ?? JSON.stringify(config.result ?? {
    accepted: true, ct_frame: body.ct_frame, inverse_uv: body.inverse_uv,
  }));
  process.exitCode = config.exit_code ?? 0;
})().catch(error => { process.stderr.write(error.stack); process.exitCode = 1; });
`;

async function directorySnapshot(root) {
  const entries = await readdir(root, { withFileTypes: true });
  const result = {};
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    result[entry.name] = entry.isDirectory()
      ? await directorySnapshot(join(root, entry.name))
      : sha(await readFile(join(root, entry.name)));
  }
  return result;
}

async function fixture(t, { anchors = [], scenario = {} } = {}) {
  const root = await mkdtemp(join(tmpdir(), "vh-refine-route-"));
  const here = join(root, "runner"), processedRoot = join(root, "processed");
  t.after(async () => {
    // Release a held subprocess even when an assertion fails before its response.
    await writeFile(join(here, "release"), "done");
    await rm(root, { recursive: true, force: true });
  });
  await mkdir(here, { recursive: true });
  await writeFile(join(here, "refine_alignment.py"), childSource);
  await writeFile(join(here, "scenario.json"), JSON.stringify(scenario));
  const manifest = JSON.stringify({
    rgb: { layers: Array.from({ length: 10 }, (_, frame) => ({ frame })) },
    density: { layers: Array.from({ length: 30 }, (_, frame) => ({ frame })) },
  });
  const alignment = JSON.stringify({ version: 2, orientation: "flip_y" });
  const inputs = { manifest_sha256: sha(manifest), alignment_sha256: sha(alignment) };
  for (const subject of ["male", "female"]) {
    const directory = join(processedRoot, subject);
    await mkdir(directory, { recursive: true });
    await Promise.all([
      writeFile(join(directory, "manifest.json"), manifest),
      writeFile(join(directory, "alignment.json"), alignment),
      writeFile(join(directory, "alignment-candidate-v2.json"), JSON.stringify({ version: 2, status: "needs_review" })),
      writeFile(join(directory, "corrections.json"), JSON.stringify({ version: 1 })),
      writeFile(join(directory, "reviewed-alignment.json"), JSON.stringify({
        anchors: anchors.map(anchor => ({ inputs, z_confirmed: true, ...anchor })),
      })),
    ]);
  }
  const handler = alignmentReviewRoutes({
    here, processedRoot, jobs: new Map(),
    sendJson(res, status, body) {
      assert.equal(res.reply, undefined, "route must reply only once");
      res.reply = { status, body };
    },
    async requestBody(req) { return structuredClone(req.body); },
  });
  const defaultBody = { ct_frame: 14, inverse_uv: FLIP_Y, orientation: "flip_y", inputs, landmarks: [] };
  function start(overrides = {}, { subject = "male", frame = 4, method = "POST" } = {}) {
    const res = new EventEmitter();
    res.destroyed = false;
    const completion = handler({ method, body: { ...defaultBody, ...overrides } }, res, ["api", "review", subject, "refine", String(frame)])
      .then(handled => { assert.equal(handled, true); return res.reply; });
    return { res, completion };
  }
  async function calls() {
    try { return (await readFile(join(here, "calls.jsonl"), "utf8")).trim().split("\n").map(line => JSON.parse(line)); }
    catch (error) { if (error.code === "ENOENT") return []; throw error; }
  }
  async function waitForChild() {
    const deadline = Date.now() + 3000;
    while (Date.now() < deadline) {
      const values = await calls();
      if (values.length) return values[0];
      await delay(10);
    }
    throw new Error("Refinement subprocess never received its request");
  }
  return {
    here, processedRoot, inputs, defaultBody, start, calls, waitForChild,
    request: (body, options) => start(body, options).completion,
    scenario: value => writeFile(join(here, "scenario.json"), JSON.stringify(value)),
    release: () => writeFile(join(here, "release"), "done"),
    snapshot: () => directorySnapshot(processedRoot),
  };
}

function assertClose(actual, expected) { assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} != ${expected}`); }

test("Auto Align server preview contract", { concurrency: false }, async t => {
  const previousPython = process.env.VISIBLE_HUMAN_PYTHON;
  process.env.VISIBLE_HUMAN_PYTHON = process.execPath;
  t.after(() => {
    if (previousPython === undefined) delete process.env.VISIBLE_HUMAN_PYTHON;
    else process.env.VISIBLE_HUMAN_PYTHON = previousPython;
  });

  await t.test("returns all six adjusted editor values without changing any processed file", async t => {
    const f = await fixture(t, { scenario: { result: { accepted: true, ct_frame: 14.25, inverse_uv: adjustedMatrix } } });
    const before = await f.snapshot();
    const response = await f.request();
    assert.equal(response.status, 200, JSON.stringify(response.body));
    assert.equal(response.body.preview_only, true);
    assert.equal(response.body.accepted, true);
    assert.deepEqual(response.body.inputs, f.inputs);
    assert.deepEqual(response.body.inverse_uv, adjustedMatrix);
    const parameters = response.body.parameters;
    assert.equal(parameters.z_position, 14.25);
    for (const [key, expected] of Object.entries({ x_position: 3, y_position: -2, rotation_deg: 2, x_scale: 1.04, y_scale: .98 })) {
      assertClose(parameters[key], expected);
    }
    const [call] = await f.calls();
    assert.deepEqual(call.args, ["--processed", join(f.processedRoot, "male"), "--color-frame", "4"]);
    assert.deepEqual(call.body, { ...f.defaultBody, z_bounds: [7.001, 29] });
    assert.deepEqual(await f.snapshot(), before, "preview must not save anchors, rebuild candidates, or create history");
  });

  await t.test("rejects malformed input before starting a subprocess", async t => {
    const f = await fixture(t), before = await f.snapshot();
    const invalid = [
      ["missing matrix", { inverse_uv: undefined }],
      ["singular matrix", { inverse_uv: [1, 1, 0, 1, 1, 0] }],
      ["sheared matrix", { inverse_uv: [1, .02, 0, 0, -1, 1] }],
      ["nonfinite matrix", { inverse_uv: [NaN, 0, 0, 0, -1, 1] }],
      ["extreme scale", { inverse_uv: applyDisplayAdjustment(FLIP_Y, { scale_x: .2 }) }],
      ["extreme rotation", { inverse_uv: applyDisplayAdjustment(FLIP_Y, { rotation_deg: 21 }) }],
      ["extreme translation", { inverse_uv: applyDisplayAdjustment(FLIP_Y, { tx: 513 }) }],
      ["negative depth", { ct_frame: -1 }],
      ["out of range depth", { ct_frame: 30 }],
      ["string depth", { ct_frame: "14" }],
      ["missing orientation", { orientation: undefined }],
      ["unsupported orientation", { orientation: "transpose" }],
      ["landmarks not an array", { landmarks: {} }],
      ["out of range landmark", { landmarks: [{ color: [1.01, 0], density: [0, 0] }] }],
      ["nonfinite landmark", { landmarks: [{ color: [0, 0], density: [0, NaN] }] }],
      ["too many landmarks", { landmarks: Array.from({ length: 31 }, () => ({ color: [0, 0], density: [0, 0] })) }],
    ];
    for (const [name, body] of invalid) assert.equal((await f.request(body)).status, 400, name);
    assert.equal((await f.request({}, { frame: 999 })).status, 400);
    assert.equal((await f.request({}, { method: "GET" })).status, 405);
    assert.deepEqual(await f.calls(), []);
    assert.deepEqual(await f.snapshot(), before);
  });

  await t.test("rejects stale and missing input hashes before running", async t => {
    const f = await fixture(t);
    for (const inputs of [undefined, {}, { ...f.inputs, alignment_sha256: "stale" }]) {
      const response = await f.request({ inputs });
      assert.equal(response.status, 409);
      assert.match(response.body.error, /reload/i);
    }
    assert.deepEqual(await f.calls(), []);
  });

  await t.test("a rejected optimization returns the exact current matrix and CT depth", async t => {
    const f = await fixture(t, { scenario: { result: { accepted: false, ct_frame: 14.5, inverse_uv: adjustedMatrix, reason: "No reliable improvement" } } });
    const before = await f.snapshot(), response = await f.request();
    assert.equal(response.status, 200);
    assert.equal(response.body.accepted, false);
    assert.equal(response.body.ct_frame, 14);
    assert.deepEqual(response.body.inverse_uv, FLIP_Y);
    assert.equal(response.body.parameters.z_position, 14);
    assert.equal(response.body.reason, "No reliable improvement");
    assert.deepEqual(await f.snapshot(), before);
  });

  await t.test("confines Z to current confirmed neighbors and ignores stale/unconfirmed reviews", async t => {
    const f = await fixture(t, { anchors: [
      { color_frame: 0, ct_frame: 7 },
      { color_frame: 2, ct_frame: 10 },
      { color_frame: 3, ct_frame: 14.75, z_confirmed: false },
      { color_frame: 4, ct_frame: 14 },
      { color_frame: 5, ct_frame: 13, inputs: {} },
      { color_frame: 6, ct_frame: 18 },
    ] });
    assert.equal((await f.request()).status, 200);
    const [call] = await f.calls();
    assert.deepEqual(call.body.z_bounds, [10.001, 17.999]);
    for (const ct_frame of [10, 18]) {
      const response = await f.request({ ct_frame });
      assert.equal(response.status, 409);
      assert.match(response.body.error, /confirmed depth order/);
    }
    assert.equal((await f.calls()).length, 1);
    await f.scenario({ result: { accepted: true, ct_frame: 18.1, inverse_uv: FLIP_Y } });
    assert.equal((await f.request()).status, 422, "subprocess output cannot cross confirmed neighbors");
  });

  await t.test("enforces the hard RGB 0 to CT 7 anchor in input and solver output", async t => {
    const f = await fixture(t);
    assert.equal((await f.request({ ct_frame: 7 }, { frame: 0 })).status, 200);
    assert.deepEqual((await f.calls())[0].body.z_bounds, [7, 7]);
    assert.equal((await f.request({ ct_frame: 7.1 }, { frame: 0 })).status, 409);
    assert.equal((await f.request({ ct_frame: 7 }, { frame: 1 })).status, 409,
      "later RGB frames must remain beyond CT 7 even without a saved first-frame anchor");
    await f.scenario({ result: { accepted: true, ct_frame: 7.01, inverse_uv: FLIP_Y } });
    assert.equal((await f.request({ ct_frame: 7 }, { frame: 0 })).status, 422);
  });

  await t.test("landmark pairs lock the corresponding CT plane during spatial refinement", async t => {
    const f = await fixture(t), landmarks = [{ color: [.2, .3], density: [.25, .7] }];
    assert.equal((await f.request({ landmarks })).status, 200);
    const [call] = await f.calls();
    assert.deepEqual(call.body.landmarks, landmarks);
    assert.deepEqual(call.body.z_bounds, [14, 14]);
    await f.scenario({ result: { accepted: true, ct_frame: 14.1, inverse_uv: FLIP_Y } });
    assert.equal((await f.request({ landmarks })).status, 422);
  });

  await t.test("an inconsistent neighbor cannot override the hard first-frame correspondence", async t => {
    const f = await fixture(t, { anchors: [{ color_frame: 2, ct_frame: 6 }] });
    const response = await f.request({ ct_frame: 7 }, { frame: 0 });
    assert.equal(response.status, 409, "no solution can preserve both RGB 0 → CT 7 and the reversed neighbor");
    assert.deepEqual(await f.calls(), []);
  });

  await t.test("serializes refinement across subjects and allows a new solve after completion", async t => {
    const f = await fixture(t, { scenario: { hold: true } });
    const first = f.start();
    await f.waitForChild();
    try {
      for (const subject of ["male", "female"]) {
        const response = await f.request({}, { subject });
        assert.equal(response.status, 409);
        assert.match(response.body.error, /already running/);
      }
      assert.equal((await f.calls()).length, 1);
    } finally { await f.release(); }
    assert.equal((await first.completion).status, 200);
    assert.equal((await f.request({}, { subject: "female" })).status, 200);
    assert.equal((await f.calls()).length, 2);
  });

  await t.test("cancels a disconnected request without saving and releases the solver lock", async t => {
    const f = await fixture(t, { scenario: { hold: true } }), before = await f.snapshot();
    const pending = f.start();
    await f.waitForChild();
    pending.res.destroyed = true;
    pending.res.emit("close");
    assert.equal(await pending.completion, undefined);
    await f.scenario({});
    assert.equal((await f.request()).status, 200);
    assert.deepEqual(await f.snapshot(), before);
  });

  for (const filename of ["manifest.json", "alignment.json", "alignment-candidate-v2.json", "reviewed-alignment.json", "corrections.json"]) {
    await t.test(`discards a result when ${filename} changes during the solve`, async t => {
      const f = await fixture(t, { scenario: { hold: true } }), pending = f.start();
      await f.waitForChild();
      const file = join(f.processedRoot, "male", filename);
      await writeFile(file, (await readFile(file, "utf8")) + "\n");
      const changed = await f.snapshot();
      await f.release();
      const response = await pending.completion;
      assert.equal(response.status, 409);
      assert.match(response.body.error, /changed during Auto Align/);
      assert.deepEqual(await f.snapshot(), changed, "stale result must not overwrite the external edit");
    });
  }

  await t.test("rejects unusable subprocess results and releases the solver lock after errors", async t => {
    const f = await fixture(t), before = await f.snapshot();
    const invalid = [
      { raw_stdout: "not JSON" },
      { exit_code: 1, result: { error: "Synthetic solver failure" } },
      { result: { accepted: "yes", ct_frame: 14, inverse_uv: FLIP_Y } },
      { result: { accepted: true, ct_frame: 14, inverse_uv: [1, 1, 0, 1, 1, 0] } },
      { result: { accepted: true, ct_frame: 14, inverse_uv: [1, .02, 0, 0, -1, 1] } },
      { result: { accepted: true, ct_frame: 14, inverse_uv: [1, 0, 0, 0, 1, 0] } },
      { result: { accepted: true, ct_frame: 14, inverse_uv: applyDisplayAdjustment(FLIP_Y, { rotation_deg: 25 }) } },
    ];
    for (const scenario of invalid) {
      await f.scenario(scenario);
      assert.equal((await f.request()).status, 422, JSON.stringify(scenario));
    }
    await f.scenario({});
    assert.equal((await f.request()).status, 200);
    assert.deepEqual(await f.snapshot(), before);
  });

  await t.test("rejects a starting matrix whose reflection contradicts the fixed orientation", async t => {
    const f = await fixture(t);
    const response = await f.request({ inverse_uv: [1, 0, 0, 0, 1, 0] });
    assert.equal(response.status, 400, "flip_y cannot silently become an unflipped source image");
    assert.deepEqual(await f.calls(), []);
  });
});
