// Real Chrome/WebGL smoke check. Exercises previews only; never saves real anchors.
import assert from "node:assert/strict";
import { acceptContentWarning } from "./browser_consent.mjs";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const artifacts = await mkdtemp(join(tmpdir(), "vhp-review-browser-"));
const browser = spawn(process.env.VISIBLE_HUMAN_CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
  "--disable-component-update", "--disable-sync", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
  "--remote-debugging-pipe", `--user-data-dir=${artifacts}/profile`, "about:blank",
], { stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"] });
let serial=0, buffer=Buffer.alloc(0), session, stderr="";
const pending=new Map(), exceptions=[];
browser.stderr.on("data", (chunk)=>{stderr=(stderr+chunk).slice(-3000);});
browser.on("error", (error)=>{for(const item of pending.values())item.reject(error);});
browser.stdio[4].on("data",(chunk)=>{
  buffer=Buffer.concat([buffer,chunk]); let boundary;
  while((boundary=buffer.indexOf(0))>=0){
    const message=JSON.parse(buffer.subarray(0,boundary)); buffer=buffer.subarray(boundary+1);
    if(message.id){const item=pending.get(message.id);if(!item)continue;clearTimeout(item.timer);pending.delete(message.id);message.error?item.reject(new Error(JSON.stringify(message.error))):item.resolve(message.result);}
    else if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails);
  }
});
function send(method,params={},sessionId=session){
  const id=++serial;return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`${method} timeout: ${stderr}`));},30000);
    pending.set(id,{resolve,reject,timer});browser.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+"\0");
  });
}
async function evaluate(expression){const value=await send("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));return value.result.value;}
const waitFor=(condition)=>evaluate(`new Promise((resolve,reject)=>{const end=Date.now()+20000;function check(){if(${condition})resolve(true);else if(Date.now()>end)reject(new Error('UI timeout: '+document.querySelector('#loading-status')?.textContent));else setTimeout(check,80);}check();})`);
async function screen(name){const result=await send("Page.captureScreenshot",{format:"png"});const file=join(artifacts,name);await writeFile(file,Buffer.from(result.data,"base64"));return file;}
try {
  const target=await send("Target.createTarget",{url:"about:blank"});session=(await send("Target.attachToTarget",{targetId:target.targetId,flatten:true})).sessionId;
  await send("Page.enable");await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride",{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
  await send("Page.navigate",{url:process.env.VISIBLE_HUMAN_REVIEW_URL || "http://127.0.0.1:4173/#alignment"});
  await acceptContentWarning(evaluate,waitFor);
  await waitFor("document.querySelector('#calibration-count')?.textContent.includes('saved') && document.querySelector('#loading-status')?.textContent.includes('male · RGB')");
  assert.ok(await evaluate("document.querySelector('#human-anchor-audit').options.length>20"));
  await evaluate("document.querySelector('#human-anchor-next').click()");
  await waitFor("document.querySelector('#calibration-status').textContent==='Knot loaded · absolute values'");
  assert.ok(await evaluate("document.querySelector('#problem-select').options.length>0"));
  assert.equal(await evaluate("document.querySelector('#registration-profile').options.length"),2);
  const candidateQueue=await evaluate("document.querySelector('#z-suggestion').options.length");assert.ok(candidateQueue>10);
  await evaluate("const profile=document.querySelector('#registration-profile');profile.value='candidate';profile.dispatchEvent(new Event('change'));");
  await waitFor("document.querySelector('#registration-profile').value==='candidate' && document.querySelector('#candidate-status').textContent.includes('saved')");
  await evaluate("new Promise(resolve=>setTimeout(resolve,500))");
  assert.equal(await evaluate("document.querySelector('#interpolation-ticks').children.length===document.querySelector('#human-anchor-audit').options.length"),true);
  const humanOnlyFrame=await evaluate(`(async()=>{
    const [value,reviews]=await Promise.all([fetch('/api/subjects/male/manifest').then(r=>r.json()),fetch('/api/review/male/frames').then(r=>r.json())]);
    const max=value.colourPlanes-1,automatic=new Set(value.registrationCandidate.spatial_knots.map(k=>Math.round(k.color_depth*max)));
    const current=reviews.anchors.filter(a=>!a.stale),currentFrames=new Set(current.map(a=>a.color_frame));
    const builtHuman=new Set((value.registrationCandidate.local_solutions||[]).filter(k=>k.reviewed_constraint===true).map(k=>k.color_frame));
    const generated=[...automatic].filter(frame=>!builtHuman.has(frame)||currentFrames.has(frame));
    const expected=new Set([...generated,...currentFrames]);
    if(document.querySelector('#interpolation-ticks').children.length!==expected.size)return -1;
    return current.find(a=>!automatic.has(a.color_frame))?.color_frame??current[0]?.color_frame??-1;
  })()`);
  assert.ok(humanOnlyFrame>=0);
  await evaluate(`document.querySelector('#interpolation-ticks [data-frame="${humanOnlyFrame}"]').click()`);
  await waitFor(`document.querySelector('#depth-index').textContent.startsWith('${humanOnlyFrame} /')`);
  assert.equal(await evaluate(`document.querySelector('#human-anchor-audit').value==='${humanOnlyFrame}'&&document.querySelector('#human-anchor-delete').textContent==='Delete RGB ${humanOnlyFrame} knot'`),true);
  assert.equal(await evaluate(`(async()=>{
    const saved=(await (await fetch('/api/review/male/frames')).json()).anchors.find(a=>a.color_frame===${humanOnlyFrame});
    return document.querySelector('#calibration-status').textContent.startsWith('Knot loaded') &&
      Number(document.querySelector('#review-ct-frame').value)===saved.ct_frame &&
      Math.abs(Number(document.querySelector('#calibration-scale_x').value)-saved.parameters.x_scale)<.0006 &&
      Math.abs(Number(document.querySelector('#calibration-tx').value)-saved.parameters.x_position)<.006;
  })()`),true);
  await evaluate("document.querySelector('#next-knot').click()");
  await waitFor(`!document.querySelector('#depth-index').textContent.startsWith('${humanOnlyFrame} /')`);
  await evaluate(`document.querySelector('#interpolation-ticks [data-frame="${humanOnlyFrame}"]').click()`);
  await waitFor(`document.querySelector('#depth-index').textContent.startsWith('${humanOnlyFrame} /')`);
  assert.equal(await evaluate(`(async()=>{const saved=(await (await fetch('/api/review/male/frames')).json()).anchors.find(a=>a.color_frame===${humanOnlyFrame});return Number(document.querySelector('#review-ct-frame').value)===saved.ct_frame&&Math.abs(Number(document.querySelector('#calibration-scale_y').value)-saved.parameters.y_scale)<.0006})()`),true);
  const auditFrame=await evaluate("Number(document.querySelector('#interpolation-ticks .knot-tick:last-child').dataset.frame)");
  assert.ok(Number.isFinite(auditFrame));
  await evaluate(`document.querySelector('#interpolation-ticks [data-frame="${auditFrame}"]').click()`);
  await waitFor(`document.querySelector('#depth-index').textContent.startsWith('${auditFrame} /')`);
  assert.equal(await evaluate(`document.querySelector('#human-anchor-audit').value==='${auditFrame}'&&!document.querySelector('#human-anchor-delete').disabled&&document.querySelector('#human-anchor-delete').textContent==='Delete RGB ${auditFrame} knot'`),true);
  assert.equal(await evaluate("(async()=>{const value=await (await fetch('/api/subjects/male/manifest')).json();return document.querySelector('#promote-candidate').disabled===!value.registrationCandidate.qc.accepted})()"),true);
  await evaluate("document.querySelector('#review-color-frame').value='1300';document.querySelector('#review-color-frame').dispatchEvent(new Event('change'));");
  await waitFor("document.querySelector('#depth-index').textContent.startsWith('1300 /') && document.querySelector('#loading-status').textContent.includes('RGB 1300')");
  assert.equal(await evaluate("document.querySelector('#save-calibration').disabled&&document.querySelector('#save-calibration').textContent==='Select a timeline knot'&&document.querySelector('#human-anchor-delete').disabled"),true);
  assert.match(await evaluate("document.querySelector('#knot-context').textContent"),/Spatial:|At |Blocked boundary:/);
  await evaluate("document.querySelector('#previous-knot').click()");
  await waitFor("Number(document.querySelector('#depth-index').textContent.split(' ')[0])<1300");
  assert.match(await evaluate("document.querySelector('#calibration-status').textContent"),/Knot loaded · absolute values/);
  assert.equal(await evaluate("document.querySelector('#human-anchor-audit').value===document.querySelector('#depth-index').textContent.split(' ')[0]&&!document.querySelector('#human-anchor-delete').disabled"),true);
  assert.notEqual(await evaluate("Number(document.querySelector('#calibration-scale_x').value)"),1);
  const previousKnot=await evaluate("Number(document.querySelector('#depth-index').textContent.split(' ')[0])");
  await evaluate("document.querySelector('#next-knot').click()");
  await waitFor(`Number(document.querySelector('#depth-index').textContent.split(' ')[0])>${previousKnot}`);
  await evaluate("document.querySelector('#review-color-frame').value='1300';document.querySelector('#review-color-frame').dispatchEvent(new Event('change'));");
  await waitFor("document.querySelector('#depth-index').textContent.startsWith('1300 /')");
  const base=await screen("male-1300-before.png");
  const plane=await evaluate("Number(document.querySelector('#review-ct-frame').value)");
  await evaluate("document.querySelector('#ct-next').click()");
  await waitFor(`Number(document.querySelector('#review-ct-frame').value)>${plane}`);
  await evaluate("document.querySelector('#ct-previous').click();document.querySelector('#match-landmarks').click()");
  await waitFor("document.querySelector('#compare-mode').value==='split'");
  const split=await screen("male-1300-source-split.png");
  // Controlled point pairs exercise side/UV conversion and fit direction.
  // These are synthetic UI inputs and are never saved to the dataset.
  await evaluate(`(()=>{
    const canvas=document.querySelector('#viewer-canvas'),r=canvas.getBoundingClientRect(),aspect=1024/304;
    const w=Math.min(r.width,r.height*aspect),h=w/aspect,left=r.left+(r.width-w)/2,top=r.top+(r.height-h)/2;
    for(const [u,v] of [[.3,.3],[.7,.3],[.5,.7],[.3,.7]])for(const side of [0,1])canvas.dispatchEvent(new PointerEvent('pointerdown',{button:0,clientX:left+(side+u)*w/2,clientY:top+v*h,pointerId:1,bubbles:true}));
  })()`);
  await waitFor("document.querySelector('#landmark-status').textContent.startsWith('4 pairs')");
  assert.equal(await evaluate("document.querySelector('#landmark-markers').children.length"),8);
  await evaluate("document.querySelector('#fit-landmarks').click()");
  await waitFor("document.querySelector('#calibration-status').textContent.startsWith('4 points')");
  assert.equal(await evaluate("document.querySelector('#compare-mode').value"),"blend");
  await evaluate("document.querySelector('#reset-calibration').click();document.querySelector('#suggest-alignment').click()");
  await waitFor("document.querySelector('#calibration-status').textContent.startsWith('Outline preview')");
  const suggestion=await screen("male-1300-outline-suggestion.png");
  const warning=await evaluate("document.querySelector('#calibration-status').textContent");
  await evaluate("const subject=document.querySelector('#subject-select');subject.value='female';subject.dispatchEvent(new Event('change'));");
  await waitFor("document.querySelector('#loading-status').textContent.startsWith('female · RGB')");
  const femaleIssues=await evaluate("document.querySelector('#problem-select').options.length");assert.ok(femaleIssues>0);
  assert.equal(await evaluate("document.querySelector('#scan-alignment').disabled"),false);
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));
  console.log(JSON.stringify({passed:true,candidateQueue,base,split,suggestion,warning,femaleIssues,exceptions},null,2));
} finally {browser.kill();for(const item of pending.values())clearTimeout(item.timer);}
