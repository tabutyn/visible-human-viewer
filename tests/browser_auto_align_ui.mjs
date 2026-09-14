// Browser interaction regression checks for unsaved Auto Align previews.
// Refinement responses are controlled; all real write requests are blocked.
import assert from "node:assert/strict";
import { acceptContentWarning } from "./browser_consent.mjs";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const artifacts = await mkdtemp(join(tmpdir(), "vhp-auto-align-browser-"));
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
  await send("Page.addScriptToEvaluateOnNewDocument",{source:`
    const readOnlyFetch=window.fetch.bind(window);
    window.fetch=(url,options={})=>{
      if(!["GET","HEAD"].includes((options.method||"GET").toUpperCase()))return Promise.resolve(new Response("{}",{status:200,headers:{"content-type":"application/json"}}));
      return readOnlyFetch(url,options);
    };
  `});
  await send("Emulation.setDeviceMetricsOverride",{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
  await send("Page.navigate",{url:process.env.VISIBLE_HUMAN_REVIEW_URL || "http://127.0.0.1:4173/?subject=male#alignment"});
  await acceptContentWarning(evaluate,waitFor);
  await waitFor("document.querySelector('#calibration-count')?.textContent.includes('saved') && document.querySelectorAll('#interpolation-ticks .knot-tick').length>2");
  await evaluate("const ticks=document.querySelectorAll('#interpolation-ticks .knot-tick');ticks[Math.floor(ticks.length/2)].click();");
  await waitFor("document.querySelector('#calibration-status').textContent==='Knot loaded · absolute values'");
  await evaluate("while(Number.parseInt(document.querySelector('#landmark-status').textContent)>0)document.querySelector('#undo-landmark').click();document.querySelector('#match-landmarks').click()");
  await waitFor("document.querySelector('#compare-mode').value==='split'");
  await evaluate(`(()=>{
    const canvas=document.querySelector('#viewer-canvas'),r=canvas.getBoundingClientRect(),aspect=1024/304;
    const w=Math.min(r.width,r.height*aspect),h=w/aspect,left=r.left+(r.width-w)/2,top=r.top+(r.height-h)/2;
    for(const [u,v] of [[.3,.3],[.7,.3],[.5,.7],[.3,.7]])for(const side of [0,1])canvas.dispatchEvent(new PointerEvent('pointerdown',{button:0,clientX:left+(side+u)*w/2,clientY:top+v*h,pointerId:1,bubbles:true}));
  })()`);
  await waitFor("document.querySelector('#landmark-status').textContent.startsWith('4 pairs')");
  // Exercise the async refinement contract without saving or changing real
  // anchors. Deferred responses expose the races caused by editing/navigation.
  await evaluate(`(async()=>{
    const {AlignmentReview}=await import('/review-workflow.mjs');
    const h=window.autoAlignSmoke={prototype:AlignmentReview.prototype,originalAuto:AlignmentReview.prototype.autoAlign,fetch:window.fetch,requests:[],writes:0};
    AlignmentReview.prototype.autoAlign=function(){h.review=this;h.before=structuredClone(this.draft);return h.originalAuto.call(this);};
    window.fetch=(input,options)=>{
      if(String(input).includes('/refine/'))return new Promise(resolve=>h.requests.push({body:JSON.parse(options.body),resolve:value=>resolve(new Response(JSON.stringify(value),{headers:{'content-type':'application/json'}}))}));
      if(options?.method==='PUT'&&String(input).includes('/frames/')){h.writes++;throw new Error('Unexpected anchor save');}
      return h.fetch(input,options);
    };
    h.finish=async(extra={})=>{
      const {applyDisplayAdjustment,inverseUvToDisplayParameters}=await import('/viewer-math.mjs');
      const request=h.requests.at(-1);
      const p=inverseUvToDisplayParameters(request.body.inverse_uv,request.body.orientation);
      h.expected=applyDisplayAdjustment(h.review.sourceMatrix,{tx:p.x_position+1.234567,ty:p.y_position-.765432,scale_x:p.x_scale*1.002,scale_y:p.y_scale*.998,rotation_deg:p.rotation_deg+.23});
      request.resolve({accepted:true,inverse_uv:h.expected,ct_frame:request.body.ct_frame,metrics:{before:1,after:.8},warnings:[],method:'UI contract fixture',...extra});
    };
    document.querySelector('#auto-align').click();
  })()`);
  await waitFor("window.autoAlignSmoke.requests.length===1");
  assert.equal(await evaluate(`(()=>{
    const h=autoAlignSmoke,body=h.requests[0].body;
    return document.querySelector('#save-calibration').disabled&&document.querySelector('#auto-align').disabled&&
      JSON.stringify(body.inverse_uv)===JSON.stringify(h.review.currentMatrix())&&body.ct_frame===h.before.ct_frame&&
      body.orientation===h.review.profile.orientation&&JSON.stringify(body.landmarks)===JSON.stringify(h.before.landmarks)&&
      JSON.stringify(body.inputs)===JSON.stringify(h.review.inputs);
  })()`),true);
  await evaluate("autoAlignSmoke.finish()");
  await waitFor("document.querySelector('#calibration-status').textContent.startsWith('Auto Align preview')");
  assert.equal(await evaluate(`(()=>{
    const h=autoAlignSmoke,draft=h.review.draft;
    return h.review.currentMatrix().every((v,i)=>Math.abs(v-h.expected[i])<1e-10)&&
      JSON.stringify(draft.original)===JSON.stringify(h.before.original)&&draft.origin===h.before.origin&&draft.color_frame===h.before.color_frame&&
      JSON.stringify(draft.landmarks)===JSON.stringify(h.before.landmarks)&&!document.querySelector('#undo-auto-align').disabled&&h.writes===0;
  })()`),true);
  await evaluate("document.querySelector('#undo-auto-align').click()");
  await waitFor("document.querySelector('#calibration-status').textContent==='Auto Align undone'");
  assert.equal(await evaluate("JSON.stringify(autoAlignSmoke.review.draft)===JSON.stringify(autoAlignSmoke.before)"),true);

  await evaluate("document.querySelector('#auto-align').click()");
  await waitFor("autoAlignSmoke.requests.length===2");
  await evaluate("autoAlignSmoke.finish({ct_frame:autoAlignSmoke.before.ct_frame+.2})");
  await waitFor("document.querySelector('#calibration-status').textContent.includes('landmarks cleared (Z changed)')");
  assert.equal(await evaluate("autoAlignSmoke.review.draft.landmarks.length===0&&autoAlignSmoke.before.landmarks.length===4"),true);
  await evaluate("document.querySelector('#undo-auto-align').click()");
  await waitFor("document.querySelector('#calibration-status').textContent==='Auto Align undone'");
  assert.equal(await evaluate("JSON.stringify(autoAlignSmoke.review.draft)===JSON.stringify(autoAlignSmoke.before)"),true);

  await evaluate("document.querySelector('#auto-align').click()");
  await waitFor("autoAlignSmoke.requests.length===3");
  await evaluate("autoAlignSmoke.finish({accepted:false,warnings:['No improvement']})");
  await waitFor("document.querySelector('#calibration-status').textContent==='Auto Align · no reliable improvement'");
  assert.equal(await evaluate("JSON.stringify(autoAlignSmoke.review.draft)===JSON.stringify(autoAlignSmoke.before)&&document.querySelector('#undo-auto-align').disabled"),true);

  await evaluate("document.querySelector('#auto-align').click()");
  await waitFor("autoAlignSmoke.requests.length===4");
  await evaluate("const tx=document.querySelector('#calibration-tx');tx.value=String(Number(tx.value)+3);tx.dispatchEvent(new Event('input'));");
  await evaluate("autoAlignSmoke.edited=structuredClone(autoAlignSmoke.review.draft);autoAlignSmoke.finish()");
  await waitFor("document.querySelector('#calibration-status').textContent==='Auto Align discarded · values changed'");
  assert.equal(await evaluate("JSON.stringify(autoAlignSmoke.review.draft)===JSON.stringify(autoAlignSmoke.edited)"),true);

  await evaluate("document.querySelector('#auto-align').click()");
  await waitFor("autoAlignSmoke.requests.length===5");
  await evaluate("document.querySelector('#review-ct-frame').value='123.456';autoAlignSmoke.finish()");
  await waitFor("document.querySelector('#calibration-status').textContent==='Auto Align discarded · values changed'");
  assert.equal(await evaluate("document.querySelector('#review-ct-frame').value==='123.456'&&JSON.stringify(autoAlignSmoke.review.draft)===JSON.stringify(autoAlignSmoke.before)"),true);
  await evaluate("autoAlignSmoke.review.setInputs();document.querySelector('#auto-align').click()");
  await waitFor("autoAlignSmoke.requests.length===6");
  await evaluate("autoAlignSmoke.review.profile.uiTestVersion=1;autoAlignSmoke.finish()");
  await waitFor("document.querySelector('#calibration-status').textContent==='Auto Align discarded · values changed'");
  assert.equal(await evaluate("JSON.stringify(autoAlignSmoke.review.draft)===JSON.stringify(autoAlignSmoke.before)"),true);
  await evaluate("delete autoAlignSmoke.review.profile.uiTestVersion;document.querySelector('#auto-align').click()");
  await waitFor("autoAlignSmoke.requests.length===7");
  await evaluate("document.querySelector('#previous-knot').click()");
  await waitFor("document.querySelector('#calibration-status').textContent==='Knot loaded · absolute values'");
  await evaluate("autoAlignSmoke.navigated=structuredClone(autoAlignSmoke.review.draft);autoAlignSmoke.finish()");
  await waitFor("!document.querySelector('#auto-align').disabled");
  assert.equal(await evaluate("JSON.stringify(autoAlignSmoke.review.draft)===JSON.stringify(autoAlignSmoke.navigated)&&document.querySelector('#calibration-status').textContent==='Knot loaded · absolute values'&&autoAlignSmoke.writes===0"),true);
  await evaluate("window.fetch=autoAlignSmoke.fetch;autoAlignSmoke.prototype.autoAlign=autoAlignSmoke.originalAuto");
  assert.equal(await evaluate("location.hash"),"#alignment");
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));
  console.log(JSON.stringify({passed:true,cases:["exact request and preview","knot identity preserved","undo","Z landmark clearing and undo","rejection","edited-value response discarded","uncommitted input response discarded","changed profile response discarded","navigation response discarded"],exceptions},null,2));
} finally {browser.kill();for(const item of pending.values())clearTimeout(item.timer);}
