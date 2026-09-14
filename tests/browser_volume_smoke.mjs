import assert from "node:assert/strict";
import { acceptContentWarning, requestNativeDetail } from "./browser_consent.mjs";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const SERVER = fileURLToPath(new URL("../server.mjs", import.meta.url));

const temporary=await mkdtemp(join(tmpdir(),"vh-volume-browser-")),root=join(temporary,"archive"),processed=join(root,"Processed","v1"),subject=join(processed,"male"),volume=join(subject,"volume-v1"),brickDirectory=join(volume,"level-0"),lowBrickDirectory=join(volume,"level-1"),rgbVolume=join(subject,"rgb-volume-v1"),rgbBrickDirectory=join(rgbVolume,"level-0"),lowRgbBrickDirectory=join(rgbVolume,"level-1");
await mkdir(brickDirectory,{recursive:true});await mkdir(lowBrickDirectory,{recursive:true});await mkdir(rgbBrickDirectory,{recursive:true});await mkdir(lowRgbBrickDirectory,{recursive:true});
const size=32,values=new Uint16Array(size**3);
for(let z=0;z<size;z++)for(let y=0;y<size;y++)for(let x=0;x<size;x++){const radius=Math.hypot(x-15.5,y-15.5,z-15.5);values[x+size*(y+size*z)]=radius<11?1900:1024;}
const payload=Buffer.from(values.buffer),sha=createHash("sha256").update(payload).digest("hex");
await writeFile(join(brickDirectory,"000-000-000.u16le"),payload);
const lowSize=16,lowValues=new Uint16Array(size*size*lowSize);for(let z=0;z<lowSize;z++)for(let y=0;y<lowSize;y++)for(let x=0;x<lowSize;x++)lowValues[x+size*(y+size*z)]=Math.hypot(x-7.5,y-7.5,z-7.5)<5.5?1900:1024;
const lowPayload=Buffer.from(lowValues.buffer),lowSha=createHash("sha256").update(lowPayload).digest("hex");await writeFile(join(lowBrickDirectory,"000-000-000.u16le"),lowPayload);
await writeFile(join(subject,"manifest.json"),JSON.stringify({baked:true,rgb:{width:1,height:1,format:"PNG",layers:[{frame:0,file:"rgb/000000.png"}]},density:{width:1,height:1,format:"U16BE",layers:[]},boundaries:[],gaps:[]}));
await writeFile(join(subject,"corrections.json"),JSON.stringify({version:1,transforms:{density:{}}}));
const histogram=Array(4096).fill(0);histogram[1024]=size**3;histogram[1900]=Math.round(4/3*Math.PI*11**3);
await writeFile(join(volume,"manifest.json"),JSON.stringify({version:1,subject:"male",format:"u16le-hu-plus-1024",bytes_per_voxel:2,hu_offset:1024,dimensions:[size,size,size],spacing_mm:[1,1,1],origin_mm:[0,0,0],direction:[1,0,0,0,1,0,0,0,1],slice_frames:Array.from({length:size},(_,i)=>i),brick_size:size,histogram:{min_hu:-1024,max_hu:3071,bins:histogram},levels:[{level:0,factor:1,dimensions:[size,size,size],grid:[1,1,1],bytes:payload.length,stored_bytes:payload.length,bricks:[{x:0,y:0,z:0,file:"level-0/000-000-000.u16le",extent:[size,size,size],bytes:payload.length,min_hu:0,max_hu:876,sha256:sha}]},{level:1,factor:2,dimensions:[lowSize,lowSize,lowSize],grid:[1,1,1],bytes:lowPayload.length,stored_bytes:lowPayload.length,bricks:[{x:0,y:0,z:0,file:"level-1/000-000-000.u16le",extent:[lowSize,lowSize,lowSize],bytes:lowPayload.length,min_hu:0,max_hu:876,sha256:lowSha}]}],acquisition_segments:[],inputs:{ct_sha256:sha,corrections_sha256:null}}));
const channelBytes=size**3,rgbPayload=Buffer.concat([Buffer.alloc(channelBytes,255),Buffer.alloc(channelBytes,72),Buffer.alloc(channelBytes,24)]),rgbSha=createHash("sha256").update(rgbPayload).digest("hex");
const lowChannelBytes=size*size*lowSize,lowRgbPayload=Buffer.concat([Buffer.alloc(lowChannelBytes,255),Buffer.alloc(lowChannelBytes,72),Buffer.alloc(lowChannelBytes,24)]),lowRgbSha=createHash("sha256").update(lowRgbPayload).digest("hex");
await writeFile(join(rgbBrickDirectory,"000-000-000.rgb8p"),rgbPayload);await writeFile(join(lowRgbBrickDirectory,"000-000-000.rgb8p"),lowRgbPayload);await writeFile(join(rgbVolume,"manifest.json"),JSON.stringify({version:1,subject:"male",format:"rgb8-planar",bytes_per_voxel:3,dimensions:[size,size,size],spacing_mm:[1,1,1],brick_size:size,levels:[{level:0,factor:1,dimensions:[size,size,size],grid:[1,1,1],bytes:rgbPayload.length,stored_bytes:rgbPayload.length,bricks:[{x:0,y:0,z:0,file:"level-0/000-000-000.rgb8p",extent:[size,size,size],bytes:rgbPayload.length,sha256:rgbSha}]},{level:1,factor:2,dimensions:[lowSize,lowSize,lowSize],grid:[1,1,1],bytes:lowRgbPayload.length,stored_bytes:lowRgbPayload.length,bricks:[{x:0,y:0,z:0,file:"level-1/000-000-000.rgb8p",extent:[lowSize,lowSize,lowSize],bytes:lowRgbPayload.length,sha256:lowRgbSha}]}],inputs:{}}));

const port=47500+Math.floor(Math.random()*300),externalUrl=process.env.VISIBLE_HUMAN_VOLUME_URL;
const server=externalUrl?null:spawn(process.execPath,[SERVER],{env:{...process.env,PORT:String(port),VISIBLE_HUMAN_ROOT:root,VISIBLE_HUMAN_PROCESSED_ROOT:processed},stdio:["ignore","pipe","pipe"]});
if(server)await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error("server timeout")),5000);server.stdout.once("data",()=>{clearTimeout(timer);resolve();});server.once("error",reject);});

const chrome=spawn(process.env.VISIBLE_HUMAN_CHROME||"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",[
  "--headless=new","--no-first-run","--no-default-browser-check","--disable-background-networking","--disable-component-update","--disable-sync","--enable-unsafe-webgpu","--use-angle=metal","--remote-debugging-pipe",`--user-data-dir=${temporary}/profile`,"about:blank",
],{stdio:["ignore","ignore","pipe","pipe","pipe"]});
let serial=0,buffer=Buffer.alloc(0),session,stderr="";const pending=new Map(),exceptions=[],consoleErrors=[];
chrome.stderr.on("data",(chunk)=>{stderr=(stderr+chunk).slice(-4000);});
chrome.on("error",(error)=>{for(const item of pending.values())item.reject(error);});
chrome.stdio[4].on("data",(chunk)=>{buffer=Buffer.concat([buffer,chunk]);let boundary;while((boundary=buffer.indexOf(0))>=0){const message=JSON.parse(buffer.subarray(0,boundary));buffer=buffer.subarray(boundary+1);if(message.id){const item=pending.get(message.id);if(!item)continue;clearTimeout(item.timer);pending.delete(message.id);message.error?item.reject(new Error(JSON.stringify(message.error))):item.resolve(message.result);}else if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails);else if(message.method==="Runtime.consoleAPICalled"&&["error","warning"].includes(message.params.type))consoleErrors.push(message.params.args.map((arg)=>arg.value||arg.description).join(" "));}});
function send(method,params={},sessionId=session){const id=++serial;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`${method} timeout: ${stderr}`));},externalUrl?180000:30000);pending.set(id,{resolve,reject,timer});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+"\0");});}
async function evaluate(expression){const value=await send("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));return value.result.value;}
async function waitFor(condition){const end=Date.now()+(externalUrl?170000:30000);while(Date.now()<end){if(await evaluate(`Boolean(${condition})`))return true;const failure=await evaluate("document.querySelector('#volume-message')?.textContent||''");if(/validation failed|WebGPU:/.test(failure))throw new Error(failure);await new Promise((resolve)=>setTimeout(resolve,100));}const detail=await evaluate("document.querySelector('#volume-message')?.textContent+' | '+(document.querySelector('#medical-workspace')?.dataset.gpuError||'no GPU error')");throw new Error(`volume timeout: ${detail}`);}
try{
  const target=await send("Target.createTarget",{url:"about:blank"});session=(await send("Target.attachToTarget",{targetId:target.targetId,flatten:true})).sessionId;await send("Page.enable");await send("Runtime.enable");
  const loadStarted=Date.now();
  await send("Emulation.setDeviceMetricsOverride",{width:1500,height:950,deviceScaleFactor:1,mobile:false});await send("Page.navigate",{url:externalUrl||`http://127.0.0.1:${port}/?subject=male#medical`});
  await acceptContentWarning(evaluate,waitFor);
  await waitFor("document.body.dataset.activeTab==='medical' && document.querySelector('#medical-workspace')?.dataset.firstRenderedRgbLevel !== undefined");
  const previewMs=Date.now()-loadStarted,firstRgbLevel=await evaluate("document.querySelector('#medical-workspace').dataset.firstRenderedRgbLevel");
  await requestNativeDetail(evaluate,waitFor);
  await waitFor(`document.body.dataset.activeTab==='medical' && document.querySelector('#medical-workspace')?.dataset.rgbVolume==='ready' && document.querySelector('#medical-workspace')?.dataset.rgbRendered==='true' && document.querySelector('#medical-workspace')?.dataset.rgbLevel==='0' && document.querySelector('#medical-workspace')?.dataset.renderedRgbLevel==='0' && document.querySelector('#volume-performance')?.textContent.includes('L0')${externalUrl?"":" && document.querySelector('#volume-message')?.textContent.includes('32×32×32')"}`);
  assert.equal(await evaluate("document.querySelector('#medical-workspace').hidden"),false);assert.ok(await evaluate("document.querySelector('#volume-canvas').width>0"));
  assert.equal(await evaluate("document.querySelector('#medical-workspace').dataset.rgbLevel"),"0");assert.equal(firstRgbLevel,externalUrl?"2":"1");
  assert.equal(await evaluate("document.querySelector('.plane-overlay')"),null);
  await evaluate("(()=>{const c=document.querySelector('#volume-canvas'),r=c.getBoundingClientRect(),x=r.left+r.width/2,y=r.top+r.height/2;c.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,metaKey:true,pointerId:11,clientX:x,clientY:y}));c.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,metaKey:true,pointerId:11,clientX:x,clientY:y+100}));c.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,metaKey:true,pointerId:11,clientX:x,clientY:y+100}));})()");
  assert.ok(Number(await evaluate("document.querySelector('#medical-workspace').dataset.planeOffset"))>.29);
  await evaluate("(()=>{const c=document.querySelector('#volume-canvas'),r=c.getBoundingClientRect(),x=r.left+r.width/2,y=r.top+r.height/2;c.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,ctrlKey:true,pointerId:12,clientX:x,clientY:y}));c.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,ctrlKey:true,pointerId:12,clientX:x,clientY:y-100}));c.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,ctrlKey:true,pointerId:12,clientX:x,clientY:y-100}));})()");
  assert.ok(Number(await evaluate("document.querySelector('#medical-workspace').dataset.planeRotateX"))< -34);
  const cameraBefore=await evaluate("({pan:document.querySelector('#medical-workspace').dataset.cameraPan,zoom:Number(document.querySelector('#medical-workspace').dataset.cameraZoom)})");await evaluate("new Promise(resolve=>{for(const key of ['w','d','e'])document.body.dispatchEvent(new KeyboardEvent('keydown',{key,bubbles:true}));setTimeout(()=>{for(const key of ['w','d','e'])document.body.dispatchEvent(new KeyboardEvent('keyup',{key,bubbles:true}));resolve();},180);})");const cameraAfter=await evaluate("({pan:document.querySelector('#medical-workspace').dataset.cameraPan,zoom:Number(document.querySelector('#medical-workspace').dataset.cameraZoom)})");assert.notEqual(cameraAfter.pan,cameraBefore.pan);assert.ok(cameraAfter.zoom>cameraBefore.zoom);
  if(!externalUrl){
    await evaluate("document.querySelector('#alignment-tab').click()");await waitFor("document.querySelector('#volume-performance')?.textContent.includes('—')");
    await evaluate("document.querySelector('#medical-tab').click();document.querySelector('#medical-tab').click()");
    await waitFor("document.querySelector('#medical-workspace')?.dataset.rgbVolume==='ready' && document.querySelector('#medical-workspace')?.dataset.rgbRendered==='true' && document.querySelector('#volume-performance')?.textContent.includes('L0') && document.querySelector('#volume-message')?.textContent.includes('32×32×32')");
  }
  await evaluate("(()=>{const p=document.querySelector('#volume-preset');p.value='lung';p.dispatchEvent(new Event('change'));})()");assert.equal(await evaluate("document.querySelector('#volume-window-center').value"),"-600");
  await evaluate("(()=>{const p=document.querySelector('#volume-preset');p.value='custom';p.dispatchEvent(new Event('change'));const d=document.querySelector('#custom-density');d.value='400';d.dispatchEvent(new Event('input'));})()");
  assert.equal(await evaluate("document.querySelector('#custom-density-control').hidden"),false);assert.equal(await evaluate("document.querySelector('#custom-density').min"),"-64");assert.equal(await evaluate("document.querySelector('#custom-density').max"),"1870");assert.equal(await evaluate("document.querySelector('#custom-density-value').textContent"),"400 HU");assert.ok(!String(await evaluate("document.querySelector('#volume-message').textContent")).includes("first RGB hit"));
  await evaluate("new Promise(resolve=>setTimeout(resolve,1500))");
  const savedView=await evaluate("fetch('/api/subjects/male/medical-view').then(response=>response.json())");assert.equal(savedView.preset,"custom");assert.ok(Number.isFinite(savedView.yaw));assert.ok(savedView.zoom>1.15);assert.equal(savedView.cameraPan.length,2);assert.equal(savedView.planeNormal.length,3);
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));assert.equal(consoleErrors.length,0,JSON.stringify(consoleErrors));
  const screenshot=join(temporary,"medical-rendering.png"),capture=await send("Page.captureScreenshot",{format:"png"});await writeFile(screenshot,Buffer.from(capture.data,"base64"));
  const volumeRect=await evaluate("(()=>{const r=document.querySelector('#volume-canvas').getBoundingClientRect();return{x:r.left,y:r.top,width:r.width,height:r.height,scale:1};})()"),volumeScreenshot=join(temporary,"medical-volume.png"),volumeCapture=await send("Page.captureScreenshot",{format:"png",clip:volumeRect});await writeFile(volumeScreenshot,Buffer.from(volumeCapture.data,"base64"));
  const centerVoxel=await evaluate("document.querySelector('#medical-workspace').dataset.centerVoxel"),gpuError=await evaluate("document.querySelector('#medical-workspace').dataset.gpuError||null");if(!externalUrl)assert.equal(centerVoxel,"1900");
  console.log(JSON.stringify({passed:true,firstRgbLevel,previewMs,nativeMs:Date.now()-loadStarted,message:await evaluate("document.querySelector('#volume-message').textContent"),performance:await evaluate("document.querySelector('#volume-performance').textContent"),centerVoxel,gpuError,screenshot,volumeScreenshot},null,2));
}finally{chrome.kill();server?.kill();for(const item of pending.values())clearTimeout(item.timer);}
