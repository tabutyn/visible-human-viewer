// Real-data static Pages/R2 emulation. Uses a fresh headless browser and local immutable objects.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtemp,readFile,stat,writeFile } from "node:fs/promises";
import { homedir,tmpdir } from "node:os";
import path from "node:path";
import { gzipSync } from "node:zlib";
import { acceptContentWarning } from "./browser_consent.mjs";

const viewerRoot=path.resolve(import.meta.dirname,"..");
const site=path.resolve(process.env.VISIBLE_HUMAN_PUBLIC_SITE||path.join(viewerRoot,"dist/public-v1"));
const dataRoot=path.resolve(process.env.VISIBLE_HUMAN_ROOT||path.join(homedir(),"Visible-Human-Project"));
const releaseRoot=path.resolve(process.env.VISIBLE_HUMAN_PUBLIC_RELEASE_ROOT||path.join(dataRoot,"PublicRelease/v1"));
const delivery=path.resolve(process.env.VISIBLE_HUMAN_DELIVERY_ROOT||path.join(dataRoot,"Delivery/v1"));
const sourceConfig=JSON.parse(await readFile(path.join(site,"release-config.json"),"utf8"));
const productionManifest=JSON.parse(await readFile(path.join(releaseRoot,new URL(sourceConfig.releaseManifestUrl).pathname),"utf8"));
const temporary=await mkdtemp(path.join(tmpdir(),"vh-static-public-"));
const requests=[];
const corruptOnce=process.env.VISIBLE_HUMAN_CORRUPT_ONCE==="1";
const interruptOnce=process.env.VISIBLE_HUMAN_INTERRUPT_ONCE==="1";
const noGpu=process.env.VISIBLE_HUMAN_NO_WEBGPU==="1";
const failureOnce=corruptOnce||interruptOnce;
let corruptTarget,faultPending=failureOnce;

function ordered(value){if(Array.isArray(value))return value.map(ordered);if(value&&typeof value==="object")return Object.fromEntries(Object.keys(value).sort().map(key=>[key,ordered(value[key])]));return value;}
function canonical(value){return Buffer.from(`${JSON.stringify(ordered(value))}\n`);}
function type(file){return file.endsWith(".html")?"text/html; charset=utf-8":file.endsWith(".mjs")?"text/javascript; charset=utf-8":file.endsWith(".css")?"text/css; charset=utf-8":file.endsWith(".json")?"application/json; charset=utf-8":file.endsWith(".md")?"text/markdown; charset=utf-8":"application/octet-stream";}

let base,manifestBytes,manifestPath,configBytes;
const server=createServer(async(req,res)=>{
  try{
    const pathname=new URL(req.url,base).pathname;requests.push({pathname,method:req.method,at:Date.now()});
    let payload,headers={};
    if(pathname==="/release-config.json"){payload=configBytes;headers={"content-type":"application/json; charset=utf-8","cache-control":"no-store"};}
    else if(pathname===manifestPath){payload=manifestBytes;headers={"content-type":"application/json; charset=utf-8","cache-control":"public, max-age=31536000, immutable, no-transform"};}
    else{
      const object=/^\/v1\/objects\/([0-9a-f]{64})\.gz$/.exec(pathname);
      if(object){
        if(faultPending&&object[1]===corruptTarget&&interruptOnce){
          const packed=await readFile(path.join(delivery,"blobs",`${object[1]}.gz`));faultPending=false;
          res.writeHead(200,{"content-type":"application/octet-stream","content-encoding":"gzip","cache-control":"no-store","content-length":packed.length});res.write(packed.subarray(0,Math.min(65536,packed.length-1)));setTimeout(()=>res.destroy(),5);return;
        }
        if(faultPending&&object[1]===corruptTarget&&corruptOnce){payload=gzipSync(Buffer.from("deliberately corrupt decoded brick"));faultPending=false;headers={"content-type":"application/octet-stream","content-encoding":"gzip","cache-control":"no-store"};}
        else{payload=await readFile(path.join(delivery,"blobs",`${object[1]}.gz`));headers={"content-type":"application/octet-stream","content-encoding":"gzip","cache-control":"public, max-age=31536000, immutable, no-transform","etag":`"${object[1]}"`};}
      }
      else{
        const relative=pathname==="/"?"index.html":pathname.slice(1);
        if(relative.includes(".."))throw Object.assign(new Error("unsafe"),{status:404});
        const file=path.resolve(site,relative);if(!file.startsWith(`${site}${path.sep}`))throw Object.assign(new Error("unsafe"),{status:404});
        payload=await readFile(file);headers={"content-type":type(file)};
        if(relative==="index.html")headers["content-security-policy"]="default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' blob:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
      }
    }
    res.writeHead(200,{...headers,"content-length":payload.length,"x-content-type-options":"nosniff"});if(req.method==="HEAD")res.end();else res.end(payload);
  }catch(error){res.writeHead(error.status||404,{"content-type":"text/plain"});res.end("not found");}
});
await new Promise((resolve,reject)=>{server.once("error",reject);server.listen(0,"127.0.0.1",resolve);});
base=`http://127.0.0.1:${server.address().port}`;
const localManifest=structuredClone(productionManifest);
for(const modality of ["ct","rgb"])for(const level of localManifest[modality].levels)for(const brick of level.bricks)brick.url=`${base}/${brick.object_key}`;
corruptTarget=localManifest.ct.levels.find(level=>level.level===2).bricks[0].transfer_sha256;
manifestBytes=canonical(localManifest);const manifestSha=createHash("sha256").update(manifestBytes).digest("hex");manifestPath=`/v1/releases/${manifestSha}/manifest.json`;
configBytes=Buffer.from(`${JSON.stringify({...sourceConfig,releaseManifestUrl:`${base}${manifestPath}`},null,2)}\n`);

const chrome=spawn(process.env.VISIBLE_HUMAN_CHROME||"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",[
  "--headless=new","--no-first-run","--no-default-browser-check","--disable-background-networking","--disable-component-update","--disable-sync","--enable-unsafe-webgpu","--use-angle=metal","--remote-debugging-pipe",`--user-data-dir=${temporary}/profile`,"about:blank",
],{stdio:["ignore","ignore","pipe","pipe","pipe"]});
let serial=0,buffer=Buffer.alloc(0),session,stderr="";const pending=new Map(),exceptions=[],gpuErrors=[];
chrome.stderr.on("data",chunk=>{stderr=(stderr+chunk).slice(-4000);});chrome.on("error",error=>{for(const item of pending.values())item.reject(error);});
chrome.stdio[4].on("data",chunk=>{buffer=Buffer.concat([buffer,chunk]);let boundary;while((boundary=buffer.indexOf(0))>=0){const message=JSON.parse(buffer.subarray(0,boundary));buffer=buffer.subarray(boundary+1);if(message.id){const item=pending.get(message.id);if(!item)continue;clearTimeout(item.timer);pending.delete(message.id);message.error?item.reject(new Error(JSON.stringify(message.error))):item.resolve(message.result);}else if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails);else if(message.method==="Runtime.consoleAPICalled"){const value=message.params.args.map(arg=>arg.value||arg.description).join(" ");if(/WebGPU:|shader compilation failed|Invalid CommandBuffer/.test(value))gpuErrors.push(value);}}});
function send(method,params={},sessionId=session){const id=++serial;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`${method} timeout: ${stderr}`));},240000);pending.set(id,{resolve,reject,timer});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+"\0");});}
async function evaluate(expression){const value=await send("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));return value.result.value;}
async function waitFor(condition,timeout=220000){const end=Date.now()+timeout;while(Date.now()<end){if(await evaluate(`Boolean(${condition})`))return;const error=await evaluate("document.querySelector('#medical-workspace')?.dataset.gpuError||''");if(error)throw new Error(error);await new Promise(resolve=>setTimeout(resolve,100));}throw new Error(`Timed out: ${condition}; ${await evaluate("document.querySelector('#volume-message')?.textContent")}`);}
async function screenshot(name){const file=path.join(temporary,name),capture=await send("Page.captureScreenshot",{format:"png"});await writeFile(file,Buffer.from(capture.data,"base64"));assert.ok((await stat(file)).size>10000);return file;}
try{
  const target=await send("Target.createTarget",{url:"about:blank"});session=(await send("Target.attachToTarget",{targetId:target.targetId,flatten:true})).sessionId;for(const domain of ["Page","Runtime"])await send(`${domain}.enable`);if(noGpu)await send("Page.addScriptToEvaluateOnNewDocument",{source:"Object.defineProperty(Navigator.prototype,'gpu',{configurable:true,get:()=>undefined})"});await send("Emulation.setDeviceMetricsOverride",{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await send("Page.navigate",{url:base});await waitFor("document.querySelector('#content-consent')?.dataset.ready==='true'");await new Promise(resolve=>setTimeout(resolve,500));
  assert.equal(requests.filter(item=>item.pathname.startsWith("/v1/")).length,0,"No release metadata or objects before consent");
  assert.ok(!requests.some(item=>/medical-renderer|release-data|volume-math|volume-alignment|viewer-math/.test(item.pathname)),"No renderer or GPU-support module before consent");
  await acceptContentWarning(evaluate,waitFor);
  if(noGpu){
    await waitFor("document.querySelector('#volume-retry')?.hidden===false");
    assert.match(await evaluate("document.querySelector('#volume-message').textContent"),/WebGPU is unavailable/);
    assert.equal(requests.filter(item=>item.pathname.startsWith("/v1/objects/")).length,0);
    console.log(JSON.stringify({passed:true,noWebGpu:true,objectRequests:0},null,2));
  }else{
  let retryShown=false;
  if(failureOnce){
    await waitFor("document.querySelector('#volume-retry')?.hidden===false||document.querySelector('#medical-workspace')?.dataset.renderedRgbLevel==='2'",30000);
    retryShown=await evaluate("document.querySelector('#volume-retry').hidden===false");
    if(retryShown){assert.match(await evaluate("document.querySelector('#volume-message').textContent"),/expected|mismatch|fetch|network/i);assert.notEqual(await evaluate("document.querySelector('#medical-workspace')?.dataset.renderedRgbLevel"),"2");await evaluate("document.querySelector('#volume-retry').click()");}
  }
  await waitFor("document.querySelector('#medical-workspace')?.dataset.renderedRgbLevel==='2'");await new Promise(resolve=>setTimeout(resolve,800));
  const l2Requests=requests.filter(item=>/^\/v1\/objects\//.test(item.pathname));if(!failureOnce){assert.equal(l2Requests.length,8);assert.match(await evaluate("document.querySelector('#volume-more-detail').textContent"),/147 MiB/);}const l2=await screenshot("public-l2.png");
  if(failureOnce){assert.equal(await evaluate("document.querySelector('#volume-retry').hidden"),true);assert.deepEqual(exceptions,[]);assert.deepEqual(gpuErrors,[]);console.log(JSON.stringify({passed:true,corruptRetry:corruptOnce&&retryShown,interruptedRetry:interruptOnce&&retryShown,transportAutoRetry:interruptOnce&&!retryShown,objectRequests:l2Requests.length,screenshot:l2},null,2));process.exitCode=0;}
  else{
  await evaluate("document.querySelector('#volume-more-detail').click()");await waitFor("document.querySelector('#medical-workspace')?.dataset.renderedRgbLevel==='1'");await new Promise(resolve=>setTimeout(resolve,800));assert.equal(requests.filter(item=>/^\/v1\/objects\//.test(item.pathname)).length,72);assert.match(await evaluate("document.querySelector('#volume-more-detail').textContent"),/1\.12 GiB/);const l1=await screenshot("public-l1.png");
  await evaluate("document.querySelector('#volume-more-detail').click()");await waitFor("document.querySelector('#medical-workspace')?.dataset.renderedRgbLevel==='0'");await new Promise(resolve=>setTimeout(resolve,800));const l0=await screenshot("public-l0.png");
  assert.deepEqual(exceptions,[]);assert.deepEqual(gpuErrors,[]);assert.ok(!requests.some(item=>/layers|review/.test(item.pathname)));assert.ok(requests.every(item=>["GET","HEAD"].includes(item.method)));
  console.log(JSON.stringify({passed:true,manifestSha,objectRequests:requests.filter(item=>item.pathname.startsWith("/v1/objects/")).length,performance:await evaluate("document.querySelector('#volume-performance').textContent"),screenshots:{l2,l1,l0}},null,2));
  }
  }
}finally{chrome.kill();server.close();for(const item of pending.values())clearTimeout(item.timer);}
