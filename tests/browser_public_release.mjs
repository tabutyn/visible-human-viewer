// Real-data release check. Uses a fresh headless browser, never the user's tab.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { acceptContentWarning } from "./browser_consent.mjs";

const base=process.env.VISIBLE_HUMAN_PUBLIC_URL||"http://127.0.0.1:4173/";
const temporary=await mkdtemp(join(tmpdir(),"vh-public-browser-"));
const chrome=spawn(process.env.VISIBLE_HUMAN_CHROME||"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",[
  "--headless=new","--no-first-run","--no-default-browser-check","--disable-background-networking",
  "--disable-component-update","--disable-sync","--enable-unsafe-webgpu","--use-angle=metal",
  "--remote-debugging-pipe",`--user-data-dir=${temporary}/profile`,"about:blank",
],{stdio:["ignore","ignore","pipe","pipe","pipe"]});
let serial=0,buffer=Buffer.alloc(0),session,stderr="",phase="warning";
const pending=new Map(),requests=[],exceptions=[],gpuErrors=[];
chrome.stderr.on("data",chunk=>{stderr=(stderr+chunk).slice(-4000);});
chrome.on("error",error=>{for(const item of pending.values())item.reject(error);});
chrome.stdio[4].on("data",chunk=>{
  buffer=Buffer.concat([buffer,chunk]);let boundary;
  while((boundary=buffer.indexOf(0))>=0){
    const message=JSON.parse(buffer.subarray(0,boundary));buffer=buffer.subarray(boundary+1);
    if(message.id){
      const item=pending.get(message.id);if(!item)continue;
      clearTimeout(item.timer);pending.delete(message.id);
      message.error?item.reject(new Error(JSON.stringify(message.error))):item.resolve(message.result);
    }else if(message.method==="Network.requestWillBeSent")requests.push({...message.params.request,phase});
    else if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails);
    else if(message.method==="Runtime.consoleAPICalled"){
      const text=message.params.args.map(arg=>arg.value||arg.description).join(" ");
      if(/WebGPU:|shader compilation failed|Invalid CommandBuffer/.test(text))gpuErrors.push(text);
    }
  }
});
function send(method,params={},sessionId=session){
  const id=++serial;
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`${method} timeout: ${stderr}`));},120000);
    pending.set(id,{resolve,reject,timer});
    chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+"\0");
  });
}
async function evaluate(expression){
  const value=await send("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});
  if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));return value.result.value;
}
async function waitFor(condition){
  const end=Date.now()+100000;
  while(Date.now()<end){
    if(await evaluate(`Boolean(${condition})`))return;
    const error=await evaluate("document.querySelector('#medical-workspace')?.dataset.gpuError||''");
    if(error)throw new Error(error);
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  throw new Error(`Timed out: ${condition}; ${await evaluate("document.querySelector('#volume-message')?.textContent")}`);
}
const pause=milliseconds=>evaluate(`new Promise(resolve=>setTimeout(resolve,${milliseconds}))`);
const datasetRequests=items=>items.filter(item=>/\/api\/(subjects|review)|\/(viewer\.js|medical-renderer\.mjs)/.test(item.url));
async function screenshot(name){
  const file=join(temporary,name),capture=await send("Page.captureScreenshot",{format:"png"});
  await writeFile(file,Buffer.from(capture.data,"base64"));return file;
}
try{
  const target=await send("Target.createTarget",{url:"about:blank"});
  session=(await send("Target.attachToTarget",{targetId:target.targetId,flatten:true})).sessionId;
  for(const domain of ["Page","Runtime","Network"])await send(`${domain}.enable`);
  await send("Emulation.setDeviceMetricsOverride",{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await send("Page.navigate",{url:base});
  await waitFor("document.querySelector('#content-consent')?.dataset.ready==='true' && document.querySelector('#enter-viewer').disabled");
  await pause(800);
  assert.deepEqual(datasetRequests(requests),[],"No renderer or dataset requests before consent");
  assert.equal(await evaluate("document.querySelector('.viewer-shell').hidden"),true);
  const warningScreenshot=await screenshot("content-warning.png");
  await evaluate("document.querySelector('#confirm-age').click()");
  assert.equal(await evaluate("document.querySelector('#enter-viewer').disabled"),true);
  await evaluate("document.querySelector('#leave-viewer').click()");
  assert.equal(await evaluate("document.querySelector('#confirm-age').checked"),false);
  assert.deepEqual(datasetRequests(requests),[]);
  // An expired remembered confirmation must not open the viewer.
  await evaluate("sessionStorage.setItem('visible-human-content-consent-v1',JSON.stringify({version:1,acceptedAt:Date.now()-9*60*60*1000}))");
  await send("Page.reload");await pause(1000);
  await waitFor("document.querySelector('#content-warning')?.hidden===false");
  assert.deepEqual(datasetRequests(requests),[]);
  phase="preview";const start=Date.now();
  await acceptContentWarning(evaluate,waitFor);
  await waitFor("document.querySelector('#medical-workspace')?.dataset.rgbRendered==='true' && document.querySelector('#volume-more-detail')?.hidden===false");
  const previewMs=Date.now()-start;
  await pause(2000);
  assert.equal(await evaluate("window.visibleHumanConfig.readOnly"),true,"Use VISIBLE_HUMAN_MODE=public");
  assert.equal(await evaluate("document.body.dataset.activeTab"),"medical");
  assert.equal(await evaluate("document.querySelector('#medical-workspace').dataset.firstRenderedRgbLevel"),"2");
  assert.equal(await evaluate("document.querySelector('#medical-workspace').dataset.rgbLevel"),"2");
  assert.equal(await evaluate("getComputedStyle(document.querySelector('.calibration-panel')).display"),"none");
  const bricks=requests.filter(item=>/\/bricks\//.test(item.url));
  assert.equal(bricks.length,8,"Only four CT + four RGB L2 bricks are downloaded");
  assert.ok(bricks.every(item=>/\/bricks\/2\//.test(item.url)),"No automatic finer-detail download");
  assert.ok(!requests.some(item=>/\/layers\//.test(item.url)),"Normal 3D does not load photographic slices");
  await evaluate("document.querySelector('#alignment-tab').click();document.querySelector('#align3d-tab').click()");
  assert.equal(await evaluate("document.body.dataset.activeTab"),"medical");
  await evaluate("new Promise(resolve=>{document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'w',bubbles:true}));setTimeout(()=>{document.body.dispatchEvent(new KeyboardEvent('keyup',{key:'w',bubbles:true}));resolve();},100);})");
  await pause(400);
  assert.ok(!requests.some(item=>!["GET","HEAD"].includes(item.method)),"Public navigation cannot save or mutate data");
  const performance=await evaluate("document.querySelector('#volume-performance').textContent");
  const detailLabel=await evaluate("document.querySelector('#volume-more-detail').textContent");
  const volumeScreenshot=await screenshot("preview-3d.png");
  // Consent persists in this tab, but Hide imagery clears it and disposes the renderer.
  phase="remembered";await send("Page.reload");
  await waitFor("document.querySelector('#content-warning')?.hidden===true");
  await waitFor("document.querySelector('#medical-workspace')?.dataset.rgbRendered==='true'");
  phase="hidden";await evaluate("document.querySelector('#hide-imagery').click()");await pause(1200);
  await waitFor("document.querySelector('#content-warning')?.hidden===false && document.querySelector('#enter-viewer')?.disabled===true");
  assert.equal(await evaluate("sessionStorage.getItem('visible-human-content-consent-v1')"),null);
  assert.deepEqual(datasetRequests(requests.filter(item=>item.phase==="hidden")),[],"Hide imagery reloads into an image-free page");
  assert.deepEqual(exceptions,[]);assert.deepEqual(gpuErrors,[]);
  console.log(JSON.stringify({passed:true,previewMs,performance,detailLabel,previewBricks:bricks.length,warningScreenshot,volumeScreenshot},null,2));
}finally{
  chrome.kill();for(const item of pending.values())clearTimeout(item.timer);
}
