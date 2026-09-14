// Real-data, hardware-WebGPU checks of the live 3D knot editor.
// CDP blocks EVERY non-GET request before it reaches the application server.
import assert from "node:assert/strict";
import { acceptContentWarning, requestNativeDetail } from "./browser_consent.mjs";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inflateSync } from "node:zlib";

const artifacts = await mkdtemp(join(tmpdir(), "vhp-live-alignment-"));
const normalSavedView = Boolean(process.env.VISIBLE_HUMAN_MEDICAL_SAVED_URL);
const url = new URL(process.env.VISIBLE_HUMAN_MEDICAL_SAVED_URL || process.env.VISIBLE_HUMAN_LIVE_URL || "http://127.0.0.1:4173/?subject=male#align3d");
const subject = url.searchParams.get("subject") || "male";
const browser = spawn(process.env.VISIBLE_HUMAN_CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
  "--disable-component-update", "--disable-sync", "--enable-unsafe-webgpu", "--use-angle=metal",
  "--remote-debugging-pipe", `--user-data-dir=${artifacts}/profile`, "about:blank",
], { stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"] });
let serial = 0, buffer = Buffer.alloc(0), session, stderr = "", saveFixture = null;
const pending = new Map(), exceptions = [], consoleErrors = [], blockedWrites = [], interceptionErrors = [], screenshots = new Map();
const screenshotPlaneRows = new Map();
browser.stderr.on("data", (chunk) => { stderr = (stderr + chunk).slice(-4000); });
browser.on("error", (error) => { for (const item of pending.values()) item.reject(error); });
browser.stdio[4].on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]); let boundary;
  while ((boundary = buffer.indexOf(0)) >= 0) {
    const message = JSON.parse(buffer.subarray(0, boundary)); buffer = buffer.subarray(boundary + 1);
    if (message.id) {
      const item = pending.get(message.id); if (!item) continue;
      clearTimeout(item.timer); pending.delete(message.id);
      message.error ? item.reject(new Error(JSON.stringify(message.error))) : item.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params.exceptionDetails);
    else if (message.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(message.params.type)) {
      consoleErrors.push(message.params.args.map((arg) => arg.value || arg.description).join(" "));
    } else if (message.method === "Fetch.requestPaused") {
      const { requestId, request } = message.params;
      if (request.method === "GET") send("Fetch.continueRequest", { requestId }, message.sessionId).catch((error) => interceptionErrors.push(error.message));
      else {
        blockedWrites.push({ method: request.method, url: request.url });
        if (saveFixture && request.method === "PUT" && new URL(request.url).pathname === `/api/review/${subject}/frames/${saveFixture.frame}`) {
          saveFixture.requestId = requestId; saveFixture.body = JSON.parse(request.postData); continue;
        }
        send("Fetch.fulfillRequest", { requestId, responseCode: 200, responseHeaders: [{ name: "Content-Type", value: "application/json" }], body: Buffer.from("{}").toString("base64") }, message.sessionId).catch((error) => interceptionErrors.push(error.message));
      }
    }
  }
});
function send(method, params = {}, sessionId = session) {
  const id = ++serial;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timeout: ${stderr}`)); }, 30000);
    pending.set(id, { resolve, reject, timer });
    browser.stdio[3].write(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }) + "\0");
  });
}
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(condition, timeout = 100000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await evaluate(`Boolean(${condition})`)) return;
    const error = await evaluate("document.querySelector('#medical-workspace')?.dataset.gpuError || ''");
    if (error) throw new Error(error);
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  const diagnostic = await evaluate("({status:document.querySelector('#calibration-status')?.textContent,message:document.querySelector('#volume-message')?.textContent,data:{...document.querySelector('#medical-workspace')?.dataset}})");
  throw new Error(`Timed out waiting for ${condition}: ${JSON.stringify(diagnostic)}`);
}
async function screenshot(name) {
  const capture = await send("Page.captureScreenshot", { format: "png" }), file = join(artifacts, name);
  const bytes = Buffer.from(capture.data, "base64"); screenshots.set(file, bytes);
  screenshotPlaneRows.set(file, await evaluate(`(() => {
    const c=document.querySelector('#medical-knot-outline'),data=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
    let top=c.height,bottom=0;for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++)if(data[(x+y*c.width)*4+3]>8){top=Math.min(top,y);bottom=Math.max(bottom,y);}
    return [Math.max(0,top-12),Math.min(c.height,bottom+12)];
  })()`));
  await writeFile(file, bytes);
  return file;
}
function screenshotPixels(bytes) {
  let width, height, channels;
  const compressed = [];
  for (let offset = 8; offset < bytes.length;) {
    const length = bytes.readUInt32BE(offset), type = bytes.toString("ascii", offset + 4, offset + 8), chunk = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = chunk.readUInt32BE(0); height = chunk.readUInt32BE(4);
      assert.equal(chunk[8], 8); assert.equal(chunk[12], 0);
      channels = { 2: 3, 6: 4 }[chunk[9]]; assert.ok(channels, "Screenshot PNG must contain RGB or RGBA pixels");
    } else if (type === "IDAT") compressed.push(chunk);
    offset += length + 12;
  }
  const packed = inflateSync(Buffer.concat(compressed)), stride = width * channels, pixels = Buffer.alloc(stride * height);
  const paeth = (a, b, c) => { const p = a + b - c, da = Math.abs(p - a), db = Math.abs(p - b), dc = Math.abs(p - c); return da <= db && da <= dc ? a : db <= dc ? b : c; };
  for (let y = 0; y < height; y++) {
    const start = y * (stride + 1), filter = packed[start]; assert.ok(filter <= 4);
    for (let x = 0; x < stride; x++) {
      const index = y * stride + x, left = x >= channels ? pixels[index - channels] : 0, above = y ? pixels[index - stride] : 0, corner = y && x >= channels ? pixels[index - stride - channels] : 0;
      pixels[index] = (packed[start + 1 + x] + [0, left, above, Math.floor((left + above) / 2), paeth(left, above, corner)][filter]) & 255;
    }
  }
  return { width, height, channels, pixels };
}
function volumeDifference(beforePath, afterPath, excludePlane = false) {
  const before = screenshotPixels(screenshots.get(beforePath)), after = screenshotPixels(screenshots.get(afterPath));
  assert.equal(before.width, after.width); assert.equal(before.height, after.height);
  let changed = 0, compared = 0, totalDifference = 0;
  const beforeRows = screenshotPlaneRows.get(beforePath), afterRows = screenshotPlaneRows.get(afterPath);
  const planeRows = [Math.min(beforeRows[0], afterRows[0]), Math.max(beforeRows[1], afterRows[1])];
  // Exclude the preset, plane's top label, editor, and bottom edge. Count
  // anatomical image pixels, not DOM status changes or green selection lines.
  for (let y = 150; y < before.height - 24; y++) for (let x = 260; x < before.width - 290; x++) {
    if (excludePlane && y >= planeRows[0] && y <= planeRows[1]) continue;
    const a = (x + y * before.width) * before.channels, b = (x + y * after.width) * after.channels;
    const green = (p, i) => p[i + 1] > 180 && p[i + 1] > p[i] * 1.2 && p[i + 1] > p[i + 2] * 1.5;
    if (green(before.pixels, a) || green(after.pixels, b)) continue;
    const delta = Math.max(...[0, 1, 2].map(channel => Math.abs(before.pixels[a + channel] - after.pixels[b + channel])));
    compared++; totalDifference += delta;
    if (delta > 8) changed++;
  }
  return { changedPixels: changed, comparedPixels: compared, meanMaximumChannelDifference: totalDifference / compared };
}
const workspace = "document.querySelector('#medical-workspace')";
const snapshot = "window.visibleHumanAlignment.snapshot()";
const rendered = `${workspace}.dataset.knotTexture === 'ready' && Number(${workspace}.dataset.alignmentRevision) > 0 && ${workspace}.dataset.renderedAlignmentRevision === ${workspace}.dataset.alignmentRevision`;
const form = "Object.fromEntries(['review-ct-frame','calibration-tx','calibration-ty','calibration-scale_x','calibration-scale_y','calibration-rotation_deg'].map(id=>[id,document.getElementById(id).value]))";
async function checkIdentity() {
  const value = await evaluate(`(() => {
    const s=${snapshot},d=s.draft,index=s.knots.findIndex(k=>k.color_frame===d.color_frame),host=document.querySelector('#medical-knot-editor-host');
    return {frame:d.color_frame,ct:d.ct_frame,index,count:s.knots.length,origin:d.origin,unsaved:d.unsaved,
      label:document.querySelector('#selected-knot-label').textContent,editorFrame:Number(document.querySelector('#review-color-frame').value),
      editorCt:Number(document.querySelector('#review-ct-frame').value),selectedKnot:Number(${workspace}.dataset.selectedKnot),selectedCt:Number(${workspace}.dataset.selectedCt),
      editorInHost:Boolean(host.querySelector('.calibration-panel')),textureFrame:Number(${workspace}.dataset.knotTextureFrame)};
  })()`);
  assert.ok(value.index >= 0, "selected draft must belong to the actual working set");
  assert.equal(value.editorFrame, value.frame);
  assert.equal(value.selectedKnot, value.frame);
  assert.equal(value.textureFrame, value.frame);
  assert.ok(Math.abs(value.editorCt - value.ct) <= .00501, "CT editor preserves expected display precision");
  assert.ok(Math.abs(value.selectedCt - value.ct) <= .00001);
  assert.ok(value.label.includes(`RGB ${value.frame} → CT ${value.ct.toFixed(2)}`));
  assert.ok(value.label.includes(`${value.index + 1} / ${value.count}`));
  assert.equal(value.editorInHost, true);
  return value;
}
try {
  const target = await send("Target.createTarget", { url: "about:blank" });
  session = (await send("Target.attachToTarget", { targetId: target.targetId, flatten: true })).sessionId;
  await send("Page.enable"); await send("Runtime.enable");
  await send("Fetch.enable", { patterns: [{ urlPattern: "*", requestStage: "Request" }] });
  await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1050, deviceScaleFactor: 1, mobile: false });
  const started = Date.now();
  await send("Page.navigate", { url: url.href });
  await acceptContentWarning(evaluate,waitFor);
  if(normalSavedView)await requestNativeDetail(evaluate,waitFor);
  if (normalSavedView) {
    await waitFor(`document.body.dataset.activeTab === 'medical' && window.visibleHumanAlignment?.snapshot().knots.length > 2 && ${workspace}?.dataset.savedAlignment === 'ready' && ${workspace}.dataset.renderedRgbLevel === '0' && ${workspace}.dataset.renderedAlignmentRevision === ${workspace}.dataset.alignmentRevision`, 170000);
    const beforeReviews = await evaluate(`fetch('/api/review/${subject}/frames').then(response=>response.json())`);
    const beforeView = await evaluate(`fetch('/api/subjects/${subject}/medical-view').then(response=>response.json())`);
    const manifestStatus = await evaluate(`fetch('/api/subjects/${subject}/volume/rgb/manifest').then(response=>response.status)`);
    assert.equal(manifestStatus, 409, "fixture must reproduce the stale baked registration hash");
    assert.equal(await evaluate("document.querySelector('.calibration-panel').checkVisibility()"), false);
    assert.equal(await evaluate("document.querySelector('#medical-knot-editor-host').hidden"), true);
    assert.equal(await evaluate(`${workspace}.dataset.selectedKnot`), "");
    assert.equal(await evaluate(`${workspace}.dataset.knotTexture || null`), null);
    const viewFields = { cameraYaw: "yaw", cameraPitch: "pitch", cameraZoom: "zoom", planeOffset: "planeOffset" };
    for (const [display, saved] of Object.entries(viewFields)) {
      const value = await evaluate(`Number(${workspace}.dataset.${display})`);
      assert.ok(Math.abs(value - beforeView[saved]) < .000001, `${saved} startup value must remain saved`);
    }
    const expected = await evaluate(`(async()=>{const {buildVolumeAlignmentLut}=await import('/volume-alignment.mjs'),s=${snapshot};const [volumeManifest,rgbManifest]=await Promise.all([fetch('/api/subjects/${subject}/volume/manifest').then(r=>r.json()),fetch('/api/subjects/${subject}/volume/rgb/manifest?live=1').then(r=>r.json())]);const lut=buildVolumeAlignmentLut({volumeManifest,rgbManifest,knots:s.knots,draft:null,profile:s.profile,sourceManifest:s.manifest,orientation:'flip_y'});return{valid:lut.validCount,enabled:lut.enabled};})()`);
    assert.equal(expected.enabled, true);
    assert.equal(await evaluate(`Number(${workspace}.dataset.liveValidSlices)`), expected.valid);
    const firstImage = await screenshot("01-normal-saved-view.png");
    // Unsaved editor drafts must never affect the normal volume or its camera.
    const beforeInteraction = await evaluate(`({pan:${workspace}.dataset.cameraPan,normal:${workspace}.dataset.planeNormal,zoom:${workspace}.dataset.cameraZoom})`);
    await evaluate(`(()=>{const s=${snapshot};window.visibleHumanAlignment.select(s.knots[Math.floor(s.knots.length/2)].color_frame);const x=document.querySelector('#calibration-tx');x.value=String(Number(x.value)+30);x.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    await waitFor(`${snapshot}.draft?.unsaved && ${workspace}.dataset.renderedAlignmentRevision === ${workspace}.dataset.alignmentRevision`);
    assert.equal(await evaluate(`${workspace}.dataset.selectedKnot`), "");
    assert.deepEqual(await evaluate(`({pan:${workspace}.dataset.cameraPan,normal:${workspace}.dataset.planeNormal,zoom:${workspace}.dataset.cameraZoom})`), beforeInteraction);
    const secondImage = await screenshot("02-unsaved-draft-ignored.png");
    const pixelDifference = volumeDifference(firstImage, secondImage);
    assert.ok(pixelDifference.changedPixels < 10, `Normal rendering must ignore unsaved drafts: ${JSON.stringify(pixelDifference)}`);
    assert.deepEqual(await evaluate(`fetch('/api/review/${subject}/frames').then(response=>response.json())`), beforeReviews);
    const afterView = await evaluate(`fetch('/api/subjects/${subject}/medical-view').then(response=>response.json())`);
    const viewTransforms = (value) => Object.fromEntries(Object.entries(value).filter(([key]) => key !== "updatedAt"));
    assert.deepEqual(viewTransforms(afterView), viewTransforms(beforeView), "another open tab may refresh the timestamp, but the saved view must remain the same");
    assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
    assert.equal(consoleErrors.length, 0, JSON.stringify(consoleErrors));
    assert.equal(interceptionErrors.length, 0, JSON.stringify(interceptionErrors));
    const report = { passed: true, mode: "normal-saved-alignment", milliseconds: Date.now() - started, manifestStatus, validSlices: expected.valid, savedViewTransformsUnchanged: true, savedReviewsUnchanged: true, pixelDifference, images: [firstImage, secondImage], blockedWrites };
    await writeFile(join(artifacts, "report.json"), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ ...report, artifacts }, null, 2));
  } else {
  await waitFor(`document.body.dataset.activeTab === 'align3d' && window.visibleHumanAlignment?.snapshot().knots.length > 2 && ${workspace}?.dataset.renderedRgbLevel === '1' && ${rendered}`);
  const beforeReviews = await evaluate(`fetch('/api/review/${subject}/frames').then(response=>response.json())`);
  const initial = await checkIdentity(), images = [], edits = [];
  assert.equal(await evaluate("document.querySelectorAll('.calibration-panel').length"), 1);
  assert.equal(await evaluate("document.querySelector('#review-color-frame').checkVisibility()"), false);
  assert.equal(await evaluate("[...document.querySelectorAll('.calibration-panel summary')].some(x=>/Saved frames|Landmarks/.test(x.textContent))"), false);
  assert.equal(await evaluate("document.querySelector('#knot-blend').checkVisibility()"), true);

  // Choose a saved knot with neighbors to verify reset uses saved, not generated values.
  await evaluate("(() => {const preset=document.querySelector('#volume-preset');preset.value='custom';preset.dispatchEvent(new Event('change'));const blend=document.querySelector('#knot-blend');blend.value='0';blend.dispatchEvent(new Event('input'));})()");
  await evaluate(`(() => { const s=${snapshot},saved=s.knots.filter((k,i)=>k.source==='human'&&i>0&&i<s.knots.length-1).sort((a,b)=>Math.abs(a.color_depth-.3)-Math.abs(b.color_depth-.3))[0];window.visibleHumanAlignment.select((saved||s.knots[Math.floor(s.knots.length/2)]).color_frame); })()`);
  await waitFor(rendered);
  const selected = await checkIdentity(), original = await evaluate(`${snapshot}.draft`), originalForm = await evaluate(form);
  images.push(await screenshot("01-selected-knot.png"));
  let volumeOnlyDifference, translatedImage;
  for (const [id, delta] of [["calibration-tx", 5], ["calibration-ty", -4], ["calibration-scale_x", .035], ["calibration-scale_y", .025], ["calibration-rotation_deg", 1], ["review-ct-frame", .5]]) {
    const revision = Number(await evaluate(`${workspace}.dataset.alignmentRevision`));
    await evaluate(`(() => { const input=document.getElementById('${id}');input.value=String(Number(input.value)+${delta});input.dispatchEvent(new Event('input',{bubbles:true})); })()`);
    await waitFor(`${rendered} && Number(${workspace}.dataset.alignmentRevision) > ${revision}`);
    const identity = await checkIdentity();
    assert.equal(identity.frame, selected.frame, "editing does not create or navigate to another knot");
    assert.equal(identity.count, selected.count);
    assert.equal(identity.unsaved, true);
    edits.push({ id, identity, revision: Number(await evaluate(`${workspace}.dataset.renderedAlignmentRevision`)) });
    if (id === "calibration-tx") {
      translatedImage = await screenshot("01b-translated-volume.png");
      volumeOnlyDifference = volumeDifference(images[0], translatedImage, true);
      assert.ok(volumeOnlyDifference.changedPixels > 500, `The RGB volume outside the selected plane must respond to translation: ${JSON.stringify(volumeOnlyDifference)}`);
    }
  }
  images.push(await screenshot("02-edited-knot.png"));
  const pixelDifference = volumeDifference(images[0], images[1]);
  assert.ok(pixelDifference.changedPixels > 500, `The 3D anatomical image must visibly respond to knot edits: ${JSON.stringify(pixelDifference)}`);
  await evaluate("document.querySelector('#reset-calibration').click()");
  await waitFor(`${rendered} && !${snapshot}.draft.unsaved`);
  const restored = await checkIdentity();
  assert.equal(restored.frame, selected.frame);
  assert.deepEqual(await evaluate(form), originalForm);
  assert.deepEqual(await evaluate(`${snapshot}.draft`), original);
  images.push(await screenshot("03-reset-knot.png"));

  await evaluate("document.querySelector('#editor-next-knot').click()");
  await waitFor(`${rendered} && ${snapshot}.draft.color_frame !== ${selected.frame}`);
  const next = await checkIdentity(); assert.equal(next.index, selected.index + 1);
  images.push(await screenshot("04-next-knot.png"));
  await evaluate("document.querySelector('#editor-previous-knot').click()");
  await waitFor(`${rendered} && ${snapshot}.draft.color_frame === ${selected.frame}`);
  assert.deepEqual(await evaluate(`${snapshot}.draft`), original);

  // Hold a save response locally, navigate to B, then release A's response.
  // This reproduces the navigation race without writing either anchor to disk.
  saveFixture = { frame: selected.frame };
  await evaluate("(() => {const input=document.querySelector('#calibration-tx');input.value=String(Number(input.value)+2);input.dispatchEvent(new Event('input',{bubbles:true}));})()");
  await waitFor(`${rendered} && ${snapshot}.draft.unsaved`);
  await evaluate("document.querySelector('#save-calibration').click()");
  const saveDeadline = Date.now() + 10000;
  while (!saveFixture.requestId && Date.now() < saveDeadline) await new Promise(resolve => setTimeout(resolve, 50));
  assert.ok(saveFixture.requestId, "save request should be held by the local fixture");
  await evaluate("document.querySelector('#editor-next-knot').click()");
  await waitFor(`${rendered} && ${snapshot}.draft.color_frame !== ${selected.frame}`);
  const navigatedDraft = await evaluate(`${snapshot}.draft`), navigatedForm = await evaluate(form);
  const savedAnchor = { ...beforeReviews.anchors.find(anchor => anchor.color_frame === selected.frame), ...saveFixture.body, color_frame: selected.frame, color_depth: original.color_depth };
  await send("Fetch.fulfillRequest", { requestId: saveFixture.requestId, responseCode: 200, responseHeaders: [{ name: "Content-Type", value: "application/json" }], body: Buffer.from(JSON.stringify(savedAnchor)).toString("base64") });
  saveFixture = null;
  await waitFor(`${rendered} && !document.querySelector('#save-calibration').disabled`);
  assert.deepEqual(await evaluate(`${snapshot}.draft`), navigatedDraft, "completing A's save must not replace any of B's draft values");
  assert.deepEqual(await evaluate(form), navigatedForm);
  assert.equal(await evaluate("Number(document.querySelector('#human-anchor-audit').value)"), navigatedDraft.color_frame);
  assert.deepEqual(await evaluate(`${snapshot}.knots.find(k=>k.color_frame===${selected.frame}).inverse_uv`), savedAnchor.inverse_uv, "saved A should still merge into the working set");
  assert.deepEqual(await evaluate(`fetch('/api/review/${subject}/frames').then(response=>response.json())`), beforeReviews);
  assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
  assert.equal(consoleErrors.length, 0, JSON.stringify(consoleErrors));
  assert.equal(interceptionErrors.length, 0, JSON.stringify(interceptionErrors));
  const report = { passed: true, milliseconds: Date.now() - started, initial, selected, restored, next, edits, pixelDifference, volumeOnlyDifference, delayedSaveNavigation: true, images: [...images, translatedImage], blockedWrites };
  await writeFile(join(artifacts, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ...report, artifacts }, null, 2));
  }
} finally {
  browser.kill();
  for (const item of pending.values()) clearTimeout(item.timer);
  for (const stream of browser.stdio) stream?.destroy?.();
}
