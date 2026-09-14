// Exercises real Auto Align previews and Undo, never saves dataset or view changes.
import assert from "node:assert/strict";
import { acceptContentWarning } from "./browser_consent.mjs";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const artifacts = await mkdtemp(join(tmpdir(), "vhp-refine-browser-"));
const base = process.env.VISIBLE_HUMAN_REVIEW_URL || "http://127.0.0.1:4173";
const frames = (process.env.VISIBLE_HUMAN_REFINE_FRAMES || "139,626,1329").split(",").map(Number);
const browser = spawn(process.env.VISIBLE_HUMAN_CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
  "--disable-component-update", "--disable-sync", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
  "--remote-debugging-pipe", `--user-data-dir=${artifacts}/profile`, "about:blank",
], { stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"] });
let serial = 0, buffer = Buffer.alloc(0), session, stderr = "";
const pending = new Map(), exceptions = [];
browser.stderr.on("data", (chunk) => { stderr = (stderr + chunk).slice(-3000); });
browser.on("error", (error) => { for (const item of pending.values()) item.reject(error); });
browser.stdio[4].on("data", (chunk) => {
  buffer = Buffer.concat([buffer, chunk]); let boundary;
  while ((boundary = buffer.indexOf(0)) >= 0) {
    const message = JSON.parse(buffer.subarray(0, boundary)); buffer = buffer.subarray(boundary + 1);
    if (message.id) { const item = pending.get(message.id); if (!item) continue; clearTimeout(item.timer); pending.delete(message.id); message.error ? item.reject(new Error(JSON.stringify(message.error))) : item.resolve(message.result); }
    else if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params.exceptionDetails);
  }
});
function send(method, params = {}, sessionId = session) {
  const id = ++serial; return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timeout: ${stderr}`)); }, 110000);
    pending.set(id, { resolve, reject, timer }); browser.stdio[3].write(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }) + "\0");
  });
}
async function evaluate(expression) {
  const value = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (value.exceptionDetails) throw new Error(JSON.stringify(value.exceptionDetails)); return value.result.value;
}
async function waitFor(condition) {
  return evaluate(`new Promise((resolve,reject)=>{const end=Date.now()+100000;function check(){if(${condition})resolve(true);else if(Date.now()>end)reject(new Error('UI timeout: '+document.querySelector('#calibration-status')?.textContent));else setTimeout(check,100);}check();})`);
}
async function screenshot(name) {
  const value = await send("Page.captureScreenshot", { format: "png" }), file = join(artifacts, name);
  await writeFile(file, Buffer.from(value.data, "base64")); return file;
}
const formExpression = "Object.fromEntries(['review-ct-frame','calibration-tx','calibration-ty','calibration-scale_x','calibration-scale_y','calibration-rotation_deg'].map(id=>[id,document.getElementById(id).value]))";
try {
  const target = await send("Target.createTarget", { url: "about:blank" });
  session = (await send("Target.attachToTarget", { targetId: target.targetId, flatten: true })).sessionId;
  await send("Page.enable"); await send("Runtime.enable");
  await send("Page.addScriptToEvaluateOnNewDocument", { source: `
    const nativeFetch=window.fetch.bind(window);window.__refinements=[];window.__blockedWrites=[];
    window.fetch=async(input,options={})=>{
      const url=new URL(typeof input==='string'?input:input.url,location.href),method=(options.method||input.method||'GET').toUpperCase();
      const refine=method==='POST'&&/^\\/api\\/review\\/male\\/refine\\/\\d+$/.test(url.pathname);
      if(method!=='GET'&&!refine){window.__blockedWrites.push(url.pathname);return new Response('{}',{headers:{'Content-Type':'application/json'}});}
      const response=await nativeFetch(input,options);if(refine)window.__refinements.push({status:response.status,...await response.clone().json()});return response;
    };
  ` });
  await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${base}/?subject=male&frame=${frames[0]}#alignment` });
  await acceptContentWarning(evaluate,waitFor);
  await waitFor("document.querySelector('#auto-align')&&!document.querySelector('#auto-align').disabled&&document.querySelector('#loading-status').textContent.includes('male · RGB')");
  assert.equal(await evaluate("document.body.dataset.activeTab"), "alignment");
  const beforeReviews = await evaluate("fetch('/api/review/male/frames').then(r=>r.json())");
  const reports = [];
  for (const frame of frames) {
    await evaluate(`document.querySelector('#review-color-frame').value='${frame}';document.querySelector('#review-color-frame').dispatchEvent(new Event('change'));`);
    await waitFor(`document.querySelector('#loading-status').textContent.includes('RGB ${frame}')&&!document.querySelector('#auto-align').disabled`);
    const before = await evaluate(formExpression), beforeImage = await screenshot(`${frame}-before.png`), start = Date.now();
    await evaluate("document.querySelector('#auto-align').click()");
    await waitFor("!document.querySelector('#auto-align').disabled");
    const result = await evaluate("window.__refinements.at(-1)");
    assert.equal(result.status, 200, JSON.stringify(result));
    assert.equal(result.preview_only, true);
    await waitFor(`document.querySelector('#loading-status').textContent.includes('RGB ${frame} · CT ${result.ct_frame.toFixed(1)} ·')`);
    await evaluate("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))");
    const after = await evaluate(formExpression), status = await evaluate("document.querySelector('#calibration-status').textContent");
    const afterImage = await screenshot(`${frame}-after.png`);
    if (result.accepted) {
      assert.match(status, /preview/);
      assert.equal(await evaluate("document.querySelector('#undo-auto-align').disabled"), false);
      await evaluate("document.querySelector('#undo-auto-align').click()");
      assert.deepEqual(await evaluate(formExpression), before);
    } else assert.deepEqual(after, before);
    const report = { frame, milliseconds: Date.now() - start, result, before, after, beforeImage, afterImage };
    reports.push(report); console.log(JSON.stringify(report));
  }
  assert.deepEqual(await evaluate("fetch('/api/review/male/frames').then(r=>r.json())"), beforeReviews);
  assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
  await writeFile(join(artifacts, "report.json"), JSON.stringify(reports, null, 2));
  console.log(JSON.stringify({ passed: true, artifacts, frames }));
} finally {
  browser.kill();
  for (const item of pending.values()) clearTimeout(item.timer);
  // Chrome can close before its debugging pipe does; do not leave Node alive.
  for (const stream of browser.stdio) stream?.destroy?.();
}
