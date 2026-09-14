import { alignedDensityFrame, alignmentAt, bracketKnots, bracketLayers, colourDepthForDensity, glMatrixFromInverseUv, IDENTITY_UV, inverseUvToDisplayParameters } from "/viewer-math.mjs";
import { AlignmentReview, estimatedDensityFrame } from "/review-workflow.mjs";

const canvas = document.querySelector("#viewer-canvas");
const subjectSelect = document.querySelector("#subject-select");
const depthSlider = document.querySelector("#depth-slider");
const depthIndex = document.querySelector("#depth-index");
const depthTotal = document.querySelector("#depth-total");
const interpolationTicks = document.querySelector("#interpolation-ticks");
const blendSlider = document.querySelector("#blend-slider");
const windowSelect = document.querySelector("#window-preset");
const compareMode = document.querySelector("#compare-mode");
const registrationToggle = document.querySelector("#registration-toggle");
const registrationProfile = document.querySelector("#registration-profile");
const status = document.querySelector("#loading-status");
const reviewBoundary = document.querySelector("#review-boundary");
const reviewMetrics = document.querySelector("#review-metrics");
const flicker = document.querySelector("#raw-flicker");
const calibrationAnchor = document.querySelector("#calibration-anchor");
const calibrationCount = document.querySelector("#calibration-count");
const calibrationStatus = document.querySelector("#calibration-status");
const calibrationPreview = document.querySelector("#calibration-preview");

const gl = canvas.getContext("webgl2", { alpha: false, antialias: true });
const state = { subject: null, manifest: null, activeRegistration: null, depth: 0, blend: 0, window: [0, 2000], compare: "blend", registrationEnabled: true, cache: new Map(), textures: null, reviewRaw: false, flickerTimer: null, renderSerial: 0, interpolationKnots: [], calibration: { anchors: [] }, calibrationDraft: null, calibrationMaxFrame: 0 };
const presets = { full: [0, 2000], soft: [40, 400], lung: [-600, 1500], bone: [500, 2000] };
function message(value) { status.textContent = value; }
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
const alignmentReview = new AlignmentReview({ state, canvas, render, setCompare(value) {
  state.compare = value; compareMode.value = value; resize();
} });
let previewSignature = "", previewProfile = null, previewManifest = null;
function alignmentSnapshot() {
  const draft = alignmentReview.atDraft() ? alignmentReview.draft : null;
  const unsaved = Boolean(draft && (Math.abs(draft.ct_frame - draft.original.ct_frame) > 1e-6
    || Object.keys(draft.adjustment).some((key) => Math.abs(draft.adjustment[key] - draft.original.adjustment[key]) > 1e-6)));
  return { subject: state.subject, profile: state.activeRegistration, manifest: state.manifest, knots: workingKnots(),
    draft: draft ? { color_frame: draft.color_frame, color_depth: draft.color_depth, ct_frame: draft.ct_frame,
      inverse_uv: alignmentReview.currentMatrix(), origin: draft.origin, unsaved } : null };
}
function notifyAlignmentPreview() {
  const snapshot = alignmentSnapshot(), draft = snapshot.draft;
  const index = draft ? snapshot.knots.findIndex((item) => item.color_frame === draft.color_frame) : -1;
  const label = document.querySelector("#selected-knot-label");
  label.textContent = draft ? `RGB ${draft.color_frame} → CT ${draft.ct_frame.toFixed(2)}${index >= 0 ? ` · ${index + 1} / ${snapshot.knots.length}` : " · slice"}${draft.unsaved ? " · unsaved" : ""}` : "No knot selected";
  label.dataset.unsaved = String(Boolean(draft?.unsaved));
  const signature = JSON.stringify({ subject: snapshot.subject, knots: snapshot.knots, draft });
  if (signature === previewSignature && previewProfile === snapshot.profile && previewManifest === snapshot.manifest) return;
  previewSignature = signature; previewProfile = snapshot.profile; previewManifest = snapshot.manifest;
  window.dispatchEvent(new CustomEvent("alignment-preview-changed", { detail: snapshot }));
}
function ensureWorkingKnot() {
  if (!state.manifest || !workingKnots().length) return null;
  const current = alignmentReview.atDraft() && workingKnots().find((item) => item.color_frame === alignmentReview.draft.color_frame);
  if (!current) {
    const nearest = workingKnots().reduce((best, item) => Math.abs(item.color_depth - state.depth) < Math.abs(best.color_depth - state.depth) ? item : best);
    jumpToInterpolationKnot(nearest);
  }
  return alignmentSnapshot();
}
window.visibleHumanAlignment = Object.freeze({
  snapshot: alignmentSnapshot,
  previous: () => stepInterpolationKnot(-1),
  next: () => stepInterpolationKnot(1),
  select(frame) {
    const knot = workingKnots().find((item) => item.color_frame === Number(frame));
    if (!knot) return false;
    jumpToInterpolationKnot(knot); return true;
  },
  ensureKnot: ensureWorkingKnot,
});
function layerList(kind) { return state.manifest?.[kind]?.layers || []; }
function expectedStep(kind) { return kind === "rgb" ? 1 : state.subject === "female" ? 3 : 1; }
function bracket(kind, frame) {
  return bracketLayers(layerList(kind), frame, expectedStep(kind), kind === "density");
}
function sourceUrl(kind, frame, raw) { return `/api/subjects/${encodeURIComponent(state.subject)}/${kind}/layers/${frame}${raw ? "?source=raw" : ""}`; }
function cacheKey(kind, frame, raw) { return `${state.subject}:${kind}:${frame}:${raw ? "raw" : "corrected"}`; }
async function fetchLayer(kind, frame, raw = false) {
  const key = cacheKey(kind, frame, raw); if (state.cache.has(key)) return state.cache.get(key);
  const response = await fetch(sourceUrl(kind, frame, raw)); if (!response.ok) throw new Error(`${kind} ${frame}: HTTP ${response.status}`);
  const value = { bytes: new Uint8Array(await response.arrayBuffer()), format: response.headers.get("X-Visible-Human-Format"), width: Number(response.headers.get("X-Visible-Human-Width")), height: Number(response.headers.get("X-Visible-Human-Height")) };
  state.cache.set(key, value); if (state.cache.size > 24) state.cache.delete(state.cache.keys().next().value); return value;
}
function compile(type, source) { const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader); if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader)); return shader; }
function program() {
  const vertex = `#version 300 es
    in vec2 position; out vec2 uv; uniform vec2 displayScale; void main(){ uv=vec2((position.x+1.0)*.5,(1.0-position.y)*.5); gl_Position=vec4(position*displayScale,0,1); }`;
  const fragment = `#version 300 es
    precision highp float; precision highp usampler2D;
    in vec2 uv; out vec4 colour;
    uniform sampler2D rgb0,rgb1; uniform usampler2D ct0,ct1;
    uniform float rgbMix,ctMix,blend,center,width; uniform ivec2 ctSize;
    uniform mat3 ctUv0,ctUv1,ctSourceUv; uniform int compareMode; uniform bool ctAvailable;
    float inside(vec2 point) { return step(0.,point.x)*step(point.x,1.)*step(0.,point.y)*step(point.y,1.); }
    float ct(usampler2D tex, vec2 point) {
      vec2 p=point*vec2(ctSize)-.5; ivec2 lo=ivec2(floor(p)); vec2 f=fract(p);
      ivec2 hi=lo+ivec2(1); lo=clamp(lo,ivec2(0),ctSize-ivec2(1)); hi=clamp(hi,ivec2(0),ctSize-ivec2(1));
      float a=float(texelFetch(tex,lo,0).r), b=float(texelFetch(tex,ivec2(hi.x,lo.y),0).r);
      float c=float(texelFetch(tex,ivec2(lo.x,hi.y),0).r), d=float(texelFetch(tex,hi,0).r);
      return mix(mix(a,b,f.x),mix(c,d,f.x),f.y)-1024.;
    }
    void main(){
      if(compareMode==3){
        vec2 point=vec2(uv.x<.5?uv.x*2.:(uv.x-.5)*2.,uv.y);
        if(uv.x<.5){colour=vec4(mix(texture(rgb0,point).rgb,texture(rgb1,point).rgb,rgbMix),1.);return;}
        vec2 source=(ctSourceUv*vec3(point,1.)).xy;
        float hu=mix(ct(ct0,source),ct(ct1,source),ctMix);
        float density=clamp((hu-(center-width*.5))/width,0.,1.);
        colour=ctAvailable?vec4(vec3(density),1.):vec4(.12,.06,.03,1.);return;
      }
      vec3 rgb=mix(texture(rgb0,uv).rgb,texture(rgb1,uv).rgb,rgbMix);
      vec2 uv0=(ctUv0*vec3(uv,1.)).xy, uv1=(ctUv1*vec3(uv,1.)).xy;
      float valid0=inside(uv0), valid1=inside(uv1);
      float weight0=(1.-ctMix)*valid0, weight1=ctMix*valid1, coverage=weight0+weight1;
      float h=(ct(ct0,clamp(uv0,0.,1.))*weight0+ct(ct1,clamp(uv1,0.,1.))*weight1)/max(coverage,.0001);
      float d=clamp((h-(center-width*.5))/width,0.,1.);
      if(!ctAvailable || coverage<.001) {
        float hatch=step(.5,fract((gl_FragCoord.x+gl_FragCoord.y)/16.));
        colour=vec4(mix(rgb,vec3(.6,.24,.04)*(.45+.2*hatch),blend*.6),1.); return;
      }
      if(compareMode==1) {
        float luma=dot(rgb,vec3(.299,.587,.114));
        float rgbEdge=smoothstep(.025,.12,fwidth(luma)*2.5);
        float ctEdge=smoothstep(.018,.10,fwidth(d)*2.5);
        vec3 edges=rgb*.28+vec3(.05,.72,1.)*rgbEdge+vec3(1.,.18,.08)*ctEdge;
        colour=vec4(mix(rgb,edges,.45+.55*blend),1.); return;
      }
      if(compareMode==2) {
        float tile=mod(floor(gl_FragCoord.x/36.)+floor(gl_FragCoord.y/36.),2.);
        vec3 checker=mix(rgb,vec3(d),tile);
        colour=vec4(mix(rgb,checker,blend),1.); return;
      }
      colour=vec4(mix(rgb,vec3(d),blend*clamp(coverage,0.,1.)),1.);
    }`;
  const p = gl.createProgram(); gl.attachShader(p, compile(gl.VERTEX_SHADER, vertex)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p;
}
function createTexture(unit, integer = false) { const texture = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texture); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); if (integer) { gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); } return texture; }
function setupGl() {
  if (!gl) throw new Error("WebGL2 is required for trilinear slice rendering");
  const p = program(); gl.useProgram(p); const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW); const pos = gl.getAttribLocation(p, "position"); gl.enableVertexAttribArray(pos); gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
  state.textures = { p, rgb0: createTexture(0), rgb1: createTexture(1), ct0: createTexture(2, true), ct1: createTexture(3, true) };
  for (const [name, unit] of [["rgb0",0],["rgb1",1],["ct0",2],["ct1",3]]) gl.uniform1i(gl.getUniformLocation(p,name),unit);
}
function rawRgb(data) { const count = data.width * data.height, rgba = new Uint8Array(count * 4); for (let i=0;i<count;i++) { rgba[i*4]=data.bytes[i]; rgba[i*4+1]=data.bytes[count+i]; rgba[i*4+2]=data.bytes[count*2+i]; rgba[i*4+3]=255; } return rgba; }
function u16be(bytes) { const output = new Uint16Array(bytes.length / 2); for (let i=0;i<output.length;i++) output[i] = bytes[i*2]*256 + bytes[i*2+1]; return output; }
function uploadRgb(texture, data) {
  gl.bindTexture(gl.TEXTURE_2D, texture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  if(data.bitmap) { gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,data.bitmap); data.bitmap.close(); }
  else gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,data.width,data.height,0,gl.RGBA,gl.UNSIGNED_BYTE,rawRgb(data));
}
function uploadCt(texture, data) { gl.bindTexture(gl.TEXTURE_2D, texture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.texImage2D(gl.TEXTURE_2D,0,gl.R16UI,data.width,data.height,0,gl.RED_INTEGER,gl.UNSIGNED_SHORT,u16be(data.bytes)); }
function resize() { const ratio = devicePixelRatio || 1, width = Math.round(canvas.clientWidth * ratio), height = Math.round(canvas.clientHeight * ratio); if (canvas.width !== width || canvas.height !== height) { canvas.width=width; canvas.height=height; } gl.viewport(0,0,canvas.width,canvas.height); draw(); }
function draw() {
  alignmentReview.updatePoints();
  if (!state.textures) return;
  const p=state.textures.p; gl.useProgram(p); gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.uniform1f(gl.getUniformLocation(p,"rgbMix"),state._rgb?.mix||0); gl.uniform1f(gl.getUniformLocation(p,"ctMix"),state._ct?.mix||0);
  gl.uniform1f(gl.getUniformLocation(p,"blend"),state.blend); gl.uniform1f(gl.getUniformLocation(p,"center"),state.window[0]); gl.uniform1f(gl.getUniformLocation(p,"width"),state.window[1]);
  gl.uniform2i(gl.getUniformLocation(p,"ctSize"),...(state._ctSize||[1,1]));
  gl.uniformMatrix3fv(gl.getUniformLocation(p,"ctUv0"),false,state._ctMatrices?.[0]||glMatrixFromInverseUv()); gl.uniformMatrix3fv(gl.getUniformLocation(p,"ctUv1"),false,state._ctMatrices?.[1]||glMatrixFromInverseUv());
  gl.uniform1i(gl.getUniformLocation(p,"compareMode"),{blend:0,edges:1,checker:2,split:3}[state.compare]||0); gl.uniform1i(gl.getUniformLocation(p,"ctAvailable"),state._ct?1:0);
  gl.uniformMatrix3fv(gl.getUniformLocation(p,"ctSourceUv"),false,glMatrixFromInverseUv(alignmentReview.sourceMatrix));
  const imageAspect=(state.compare==="split"?2:1)*2048/1216,canvasAspect=canvas.width/Math.max(1,canvas.height); gl.uniform2f(gl.getUniformLocation(p,"displayScale"),canvasAspect>imageAspect?imageAspect/canvasAspect:1,canvasAspect>imageAspect?1:canvasAspect/imageAspect); gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
}
async function render() {
  notifyAlignmentPreview();
  if (!state.manifest) return;
  updateKnotContext();
  if (["medical", "align3d"].includes(document.body.dataset.activeTab)) return;
  const serial=++state.renderSerial,max=Math.max(0,state.manifest.colourPlanes-1),frame=state.depth*max; state._rgb=bracket("rgb",frame);
  const profile=state.registrationEnabled?state.activeRegistration:null;
  const reviewed=profile?alignmentReview.selection(state.depth):null;
  const verifiedDepth=profile?alignedDensityFrame(profile,state.depth):null;
  const densityFrame=reviewed?.ct_frame??(profile?estimatedDensityFrame(profile,state.depth,max):frame);
  const densityLayers=layerList("density");
  state._ct=Number.isFinite(densityFrame)&&densityFrame>=densityLayers[0]?.frame&&densityFrame<=densityLayers.at(-1)?.frame?bracket("density",densityFrame):null;
  const densityGap=state._ct?.missingGap;
  if(densityGap)state._ct=null;
  if(state._ct&&profile){
    const depthA=colourDepthForDensity(profile,state._ct.a.frame)??state.depth,depthB=colourDepthForDensity(profile,state._ct.b.frame)??state.depth;
    const matrixA=reviewed?.inverse_uv??alignmentAt(profile,depthA).inverseUv;
    const matrixB=reviewed?.inverse_uv??alignmentAt(profile,depthB).inverseUv;
    state._ctMatrices=[glMatrixFromInverseUv(matrixA),glMatrixFromInverseUv(matrixB)]; state._ctConfidence=alignmentAt(profile,state.depth).confidence;
  }else{state._ctMatrices=[glMatrixFromInverseUv(IDENTITY_UV),glMatrixFromInverseUv(IDENTITY_UV)];state._ctConfidence=null;}
  depthIndex.textContent=`${Math.round(frame)} / ${max}${Number.isFinite(densityFrame)?` · CT ${densityFrame.toFixed(1)}`:" · CT —"}`; depthSlider.value=String(Math.round(state.depth*10000));
  updateKnotContext();
  const rgbBracket=state._rgb,ctBracket=state._ct;
  try {
    const jobs=[];
    if(rgbBracket)jobs.push(fetchLayer("rgb",rgbBracket.a.frame,state.reviewRaw),fetchLayer("rgb",rgbBracket.b.frame,state.reviewRaw));
    if(ctBracket)jobs.push(fetchLayer("density",ctBracket.a.frame,state.reviewRaw),fetchLayer("density",ctBracket.b.frame,state.reviewRaw));
    const fetched=await Promise.all(jobs);
    const data=await Promise.all(fetched.map(async(item,index)=>{
      if(rgbBracket&&index<2&&["PNG","WEBP"].includes(item.format))return {...item,bitmap:await createImageBitmap(new Blob([item.bytes],{type:item.format==="WEBP"?"image/webp":"image/png"}))};
      return item;
    }));
    if(serial!==state.renderSerial){for(const item of data)item.bitmap?.close();return;}
    let i=0;
    if(rgbBracket){uploadRgb(state.textures.rgb0,data[i++]);uploadRgb(state.textures.rgb1,data[i++]);}
    if(ctBracket){const a=data[i++],b=data[i++];state._ctSize=[a.width,a.height];uploadCt(state.textures.ct0,a);uploadCt(state.textures.ct1,b);}
    else{state._ctSize=[1,1];uploadCt(state.textures.ct0,{bytes:new Uint8Array(2),width:1,height:1});uploadCt(state.textures.ct1,{bytes:new Uint8Array(2),width:1,height:1});}
    if(serial!==state.renderSerial)return; draw(); message(`${state.subject} · RGB ${Math.round(frame)} · ${Number.isFinite(densityFrame)?`CT ${densityFrame.toFixed(1)}`:"CT —"} · ${state.reviewRaw?"raw":"corrected"} · ${registrationProfile.value}${densityGap?" · density gap":!state._ct?" · no CT":""}`);
  } catch(error) { if(serial===state.renderSerial)message(`Layer unavailable: ${error.message}`); }
}
async function selectSubject(id, preferredProfile = "candidate") {
  const serial = state.subjectSerial = (state.subjectSerial || 0) + 1;
  state.renderSerial++; alignmentReview.loadSerial++; alignmentReview.draft=null;
  alignmentReview.anchors=[]; alignmentReview.legacy=[]; alignmentReview.matching=false;
  state.subject=id; state.manifest=null; state.cache.clear();
  const response=await fetch(`/api/subjects/${id}/manifest`); if(!response.ok) throw new Error(`manifest HTTP ${response.status}`);
  const manifest=await response.json(); if(serial!==state.subjectSerial)return;
  state.manifest=manifest; state.depth=0;
  const max=Math.max(0,state.manifest.colourPlanes-1); state.calibrationMaxFrame=max; depthTotal.textContent=String(max);
  registrationProfile.replaceChildren(new Option("Baseline", "baseline"), ...(manifest.registrationCandidate ? [new Option(`Candidate · ${String(manifest.registrationCandidateStatus).replaceAll("_", " ")}`, "candidate")] : []));
  registrationProfile.value=manifest.registrationCandidate&&preferredProfile!=="baseline"?"candidate":"baseline"; registrationProfile.disabled=!manifest.registrationCandidate;
  state.activeRegistration=registrationProfile.value==="candidate"?manifest.registrationCandidate:manifest.registration;
  registrationToggle.disabled=!state.activeRegistration; registrationToggle.title=state.activeRegistration?"Apply CT-to-color alignment":`Alignment profile ${state.manifest.registrationStatus||"missing"}`; state.registrationEnabled=Boolean(state.activeRegistration)&&registrationToggle.checked;
  refreshCandidateControls();
  reviewBoundary.replaceChildren(...state.manifest.boundaries.map((b)=>new Option(`${b.modality || "layer"} · ${b.left} → ${b.right}`,b.id || b.boundary)));
  await alignmentReview.load(); refreshInterpolationKnots(); await render();
  window.dispatchEvent(new CustomEvent("subject-selected",{detail:{subject:id}}));
}
async function loadSubjects() {
  const query=new URLSearchParams(location.search), requestedSubject=query.get("subject"), requestedFrame=query.get("frame");
  const response=await fetch("/api/subjects"), items=await response.json();
  subjectSelect.replaceChildren(...items.map((x)=>new Option(`${x.label}${x.processed?" · corrected":" · source"}${x.registrationStatus==="ready"?" · profile available":""}`,x.id)));
  if(!items.length){message("No Visible Human datasets found. Set VISIBLE_HUMAN_ROOT.");return;}
  const subject=items.find((x)=>x.id===requestedSubject)?.id||items[0].id;
  subjectSelect.value=subject; await selectSubject(subject,query.get("profile")||"candidate");
  if(requestedFrame!==null&&state.activeRegistration)alignmentReview.edit(Number(requestedFrame));
  if(document.body.dataset.activeTab==="align3d")ensureWorkingKnot();
}
async function loadReview() { const boundary=reviewBoundary.value; if(!boundary) return; const r=await fetch(`/api/review/${state.subject}/boundaries/${boundary}`); const x=await r.json(); for(const id of ["tx","ty","scale","rotation_deg"]) document.querySelector(`#review-${id}`).value=x.transform?.[id] ?? (id==="scale"?1:0); const max=Math.max(1,state.manifest.colourPlanes-1); if(Number.isFinite(x.right)){state.depth=clamp(x.right/max,0,1);await render();} reviewMetrics.textContent=JSON.stringify({accepted:x.accepted,improvement:x.improvement,overlap:x.overlap,raw:x.raw_residual,corrected:x.corrected_residual}); }
async function saveReview() { const boundary=reviewBoundary.value; if(!boundary) return; const transform=Object.fromEntries(["tx","ty","scale","rotation_deg"].map((id)=>[id,Number(document.querySelector(`#review-${id}`).value)])); await fetch(`/api/review/${state.subject}/boundaries/${boundary}`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({transform})}); const job=await (await fetch(`/api/review/${state.subject}/rebuild`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({boundary})})).json(); const poll=async()=>{const x=await (await fetch(`/api/review/jobs/${job.id}`)).json(); reviewMetrics.textContent=x.state; if(x.state==="queued"||x.state==="running") setTimeout(poll,600);else if(x.state==="complete")selectSubject(state.subject,registrationProfile.value);}; poll(); }
subjectSelect.addEventListener("change",()=>selectSubject(subjectSelect.value,"candidate").catch((e)=>message(e.message)));
depthSlider.addEventListener("input",()=>{state.depth=Number(depthSlider.value)/10000;render();});
blendSlider.addEventListener("input",()=>{state.blend=Number(blendSlider.value)/100;draw();});
windowSelect.addEventListener("change",()=>{state.window=presets[windowSelect.value];draw();});
compareMode.addEventListener("change",()=>{state.compare=compareMode.value;resize();});
registrationToggle.addEventListener("change",()=>{state.registrationEnabled=registrationToggle.checked&&Boolean(state.activeRegistration);render();});
registrationProfile.addEventListener("change",async()=>{state.activeRegistration=registrationProfile.value==="candidate"?state.manifest.registrationCandidate:state.manifest.registration;state.registrationEnabled=Boolean(state.activeRegistration)&&registrationToggle.checked;await alignmentReview.load();refreshInterpolationKnots();render();});
reviewBoundary.addEventListener("change",loadReview);document.querySelector("#save-override").addEventListener("click",saveReview);
flicker.addEventListener("change",()=>{clearInterval(state.flickerTimer);state.flickerTimer=null;state.reviewRaw=false;if(flicker.checked)state.flickerTimer=setInterval(()=>{state.reviewRaw=!state.reviewRaw;render();},450);render();});
window.addEventListener("resize",resize);
window.addEventListener("keydown",(event)=>{
  if(document.body.dataset.activeTab!=="alignment")return;
  if(event.target.matches("input,select,button"))return;
  if(event.key==="["||event.key==="]"){event.preventDefault();const max=Math.max(1,state.manifest?.colourPlanes-1||1);state.depth=clamp(state.depth+(event.key==="]"?1:-1)/max,0,1);render();}
});
window.addEventListener("alignment-tab-activated",()=>{resize();render();});
window.addEventListener("medical-tab-activated",()=>{state.renderSerial++;state.cache.clear();});
window.addEventListener("alignment3d-tab-activated",()=>{state.renderSerial++;state.cache.clear();ensureWorkingKnot();notifyAlignmentPreview();});
try { setupGl(); resize(); loadSubjects().catch((error)=>message(error.message)); } catch(error) { message(error.message); }

function humanAnchors(){return alignmentReview.anchors||[];}
function workingKnots(){return state.interpolationKnots||[];}
function selectedHumanAnchor(){return workingKnots().find(item=>String(item.color_frame)===document.querySelector("#human-anchor-audit").value);}
function refreshHumanAnchors(preferredFrame) {
  const knots=workingKnots(),select=document.querySelector("#human-anchor-audit"),previous=preferredFrame??state.currentReviewFrame??select.value;
  select.replaceChildren(...knots.map((item,index)=>new Option(`${index+1}. RGB ${item.color_frame} → CT ${Number(item.ct_frame).toFixed(1)}`,String(item.color_frame))));
  if(knots.some(item=>String(item.color_frame)===String(previous)))select.value=String(previous);
  else select.selectedIndex=-1;
  document.querySelector("#human-anchor-status").textContent=`${knots.length} knots`;
  const disabled=!knots.length;
  for(const id of ["human-anchor-previous","human-anchor-open","human-anchor-next"])document.querySelector(`#${id}`).disabled=disabled;
  updateHumanAnchorDetail();
}
function updateHumanAnchorDetail(){
  const item=selectedHumanAnchor(),detail=document.querySelector("#human-anchor-detail"),button=document.querySelector("#human-anchor-delete");
  detail.textContent=item?`RGB ${item.color_frame} · CT ${Number(item.ct_frame).toFixed(2)}`:"No working knot selected.";
  const editable=item&&state.activeRegistration===state.manifest?.registrationCandidate;
  button.textContent=item?`Delete RGB ${item.color_frame} knot`:"Nothing to delete";button.disabled=!editable;
}
function openHumanAnchor(item=selectedHumanAnchor()){if(item)jumpToInterpolationKnot(item);}
function stepHumanAnchor(direction){
  const knots=workingKnots(),select=document.querySelector("#human-anchor-audit");if(!knots.length)return;
  const current=knots.findIndex(item=>String(item.color_frame)===select.value),index=current<0?(direction>0?0:knots.length-1):(current+direction+knots.length)%knots.length;
  select.value=String(knots[index].color_frame);updateHumanAnchorDetail();jumpToInterpolationKnot(knots[index]);
}
document.querySelector("#human-anchor-audit").addEventListener("change",()=>{updateHumanAnchorDetail();openHumanAnchor();});
document.querySelector("#human-anchor-open").addEventListener("click",()=>openHumanAnchor());
document.querySelector("#human-anchor-previous").addEventListener("click",()=>stepHumanAnchor(-1));
document.querySelector("#human-anchor-next").addEventListener("click",()=>stepHumanAnchor(1));
document.querySelector("#human-anchor-delete").addEventListener("click",async()=>{
  const item=selectedHumanAnchor();if(!item||state.activeRegistration!==state.manifest?.registrationCandidate||!window.confirm(`Delete working knot RGB ${item.color_frame} → CT ${Number(item.ct_frame).toFixed(1)}?`))return;
  const index=workingKnots().indexOf(item);try{
    const response=await fetch(`/api/review/${state.subject}/candidate`,{method:"PUT",headers:{"content-type":"application/json"},body:JSON.stringify({action:"delete_knot",color_frame:item.color_frame})});
    const value=await response.json();if(!response.ok)throw new Error(value.error||`HTTP ${response.status}`);
    alignmentReview.anchors=value.anchors;const candidate=state.manifest.registrationCandidate;
    candidate.deleted_knot_frames=value.deleted_knot_frames;candidate.status="needs_rebuild";candidate.qc||={};candidate.qc.accepted=false;
    alignmentReview.refreshSaved();refreshInterpolationKnots();alignmentReview.reviewsChanged();
    const next=workingKnots()[Math.min(index,workingKnots().length-1)];if(next)jumpToInterpolationKnot(next);
  }catch(error){message(error.message);}
});
window.addEventListener("human-anchors-updated",()=>{refreshInterpolationKnots();});
window.addEventListener("review-frame-selected",(event)=>{
  const select=document.querySelector("#human-anchor-audit"),frame=String(event.detail?.color_frame??"");
  state.currentReviewFrame=Number(event.detail?.color_frame);
  if([...select.options].some(option=>option.value===frame))select.value=frame;
  else select.selectedIndex=-1;
  updateHumanAnchorDetail();
});

function interpolationKnotRecords() {
  const profile=state.activeRegistration,max=Math.max(1,state.manifest?.colourPlanes-1||1);
  if(!profile?.spatial_knots?.length)return[];
  const candidateProfile=profile===state.manifest?.registrationCandidate;
  const deletedFrames=new Set((candidateProfile?profile.deleted_knot_frames||[]:[]).map(Number));
  const currentHumanFrames=new Set(alignmentReview.anchors.filter(item=>!item.stale).map(item=>item.color_frame));
  const builtHumanFrames=new Set((candidateProfile?profile.local_solutions||[]:[]).filter(item=>item.reviewed_constraint===true).map(item=>item.color_frame));
  const parameters=profile.parameter_knots||[];
  const records=profile.spatial_knots.map((item)=>{
    const parameter=parameters.find(value=>Math.abs(value.color_depth-item.color_depth)<=1e-9);
    const color_frame=Number.isFinite(parameter?.color_frame)?parameter.color_frame:Math.round(item.color_depth*max);
    const ct_frame=Number.isFinite(parameter?.z_position)?parameter.z_position:alignedDensityFrame(profile,item.color_depth);
    return {...item,color_frame,ct_frame,source:candidateProfile?"automatic":"baseline",parameters:inverseUvToDisplayParameters(item.inverse_uv,profile.orientation)};
  }).filter(item=>!deletedFrames.has(item.color_frame)&&(!candidateProfile||!builtHumanFrames.has(item.color_frame)||currentHumanFrames.has(item.color_frame)));
  // Every saved human anchor is a real interpolation knot, even when it was
  // created between the candidate's original control points. Replace an
  // automatic knot at the same RGB frame with the saved absolute values.
  const byFrame=new Map(records.map(item=>[item.color_frame,item]));
  for(const anchor of alignmentReview.anchors.filter(item=>!item.stale&&!deletedFrames.has(item.color_frame))){
    const color_depth=Number.isFinite(anchor.color_depth)?anchor.color_depth:anchor.color_frame/max;
    byFrame.set(anchor.color_frame,{...anchor,color_depth,color_frame:anchor.color_frame,ct_frame:anchor.ct_frame,source:"human",parameters:inverseUvToDisplayParameters(anchor.inverse_uv,profile.orientation)});
  }
  return [...byFrame.values()].sort((a,b)=>a.color_depth-b.color_depth).map((item,index)=>({...item,_index:index}));
}
function knotName(item){return `KNOT RGB ${item.color_frame}${Number.isFinite(item.ct_frame)?`→CT ${Number(item.ct_frame).toFixed(1)}`:""}`;}
function knotProvenance(item){return item.source==="human"?"saved by you":item.source==="automatic"?"candidate generated":"baseline generated";}
function jumpToInterpolationKnot(item){
  if(!item)return;try{
    alignmentReview.edit(item.color_frame,false,item.source!=="human");
    // Human edit() already loaded the authoritative saved CT frame and
    // landmarks. Only automatic/profile knots need their profile Z applied.
    if(item.source!=="human"&&Number.isFinite(item.ct_frame)){
      alignmentReview.changePlane(item.ct_frame);
      alignmentReview.draft.original.ct_frame = item.ct_frame;
    }
    calibrationStatus.textContent="Knot loaded · absolute values";
    render();
  }catch(error){message(error.message);}
}
function refreshInterpolationKnots(){
  state.interpolationKnots=interpolationKnotRecords();
  interpolationTicks.replaceChildren(...state.interpolationKnots.map((item)=>{
    const tick=document.createElement("button");tick.type="button";tick.className=`knot-tick ${item.source}`;tick.style.left=`${100*item.color_depth}%`;
    tick.dataset.frame=String(item.color_frame);tick.dataset.source=item.source;
    tick.title=`${knotName(item)} · ${knotProvenance(item)} · click to open`;tick.setAttribute("aria-label",tick.title);tick.onclick=()=>jumpToInterpolationKnot(item);return tick;
  }));
  updateKnotContext();refreshHumanAnchors();notifyAlignmentPreview();
}
function updateKnotContext(){
  const knots=state.interpolationKnots||[],profile=state.activeRegistration;
  const context=document.querySelector("#knot-context"),values=document.querySelector("#knot-values");
  for(const tick of interpolationTicks.children)tick.classList.remove("bracket");
  const bracket=bracketKnots(knots,state.depth,"color_depth");
  if(!bracket){context.textContent="No spatial interpolation profile";values.textContent="";for(const id of ["previous-knot","next-knot","editor-previous-knot","editor-next-knot"])document.querySelector(`#${id}`).disabled=true;return;}
  const {left,right,mix}=bracket;
  interpolationTicks.children[left._index]?.classList.add("bracket");interpolationTicks.children[right._index]?.classList.add("bracket");
  const blocked=left!==right&&((left.segment&&right.segment&&left.segment!==right.segment)||(profile?.review_boundaries||[]).some(value=>value>left.color_depth&&value<=right.color_depth));
  context.textContent=left===right?`At ${knotName(left)}`:blocked?`Blocked boundary: ${knotName(left)} ↔ ${knotName(right)}`:`Spatial: ${knotName(left)} ↔ ${knotName(right)} · ${Math.round(mix*100)}%`;
  const z=bracketKnots(profile?.depth_knots,state.depth,"color_depth"),format=(number,digits=1)=>Number.isFinite(number)?Number(number).toFixed(digits):"—";
  const lp=left.parameters||{},rp=right.parameters||{};
  const spatial=left===right?`X ${format(lp.x_position)} · Y ${format(lp.y_position)} · scale ${format(lp.x_scale,3)}×${format(lp.y_scale,3)}`:
    `X ${format(lp.x_position)}→${format(rp.x_position)} · Y ${format(lp.y_position)}→${format(rp.y_position)} · scale ${format(lp.x_scale,3)}×${format(lp.y_scale,3)}→${format(rp.x_scale,3)}×${format(rp.y_scale,3)}`;
  const zText=z?(z.left===z.right?`Z knot RGB ${Math.round(z.left.color_depth*Math.max(1,state.manifest.colourPlanes-1))}→CT ${format(z.left.ct_frame)}`:
    `Z RGB ${Math.round(z.left.color_depth*Math.max(1,state.manifest.colourPlanes-1))}→CT ${format(z.left.ct_frame)} ↔ RGB ${Math.round(z.right.color_depth*Math.max(1,state.manifest.colourPlanes-1))}→CT ${format(z.right.ct_frame)}`):"Z unavailable";
  values.textContent=`${blocked?"No interpolation across this acquisition boundary · ":""}${spatial} · ${zText}`;
  values.title=values.textContent;
  const epsilon=.5/Math.max(1,state.manifest?.colourPlanes-1||1);
  document.querySelector("#previous-knot").disabled=!knots.some(item=>item.color_depth<state.depth-epsilon);
  document.querySelector("#next-knot").disabled=!knots.some(item=>item.color_depth>state.depth+epsilon);
  document.querySelector("#editor-previous-knot").disabled=document.querySelector("#previous-knot").disabled;
  document.querySelector("#editor-next-knot").disabled=document.querySelector("#next-knot").disabled;
}
function stepInterpolationKnot(direction){
  const epsilon=.5/Math.max(1,state.manifest?.colourPlanes-1||1),knots=state.interpolationKnots||[];
  const item=direction<0?[...knots].reverse().find(value=>value.color_depth<state.depth-epsilon):knots.find(value=>value.color_depth>state.depth+epsilon);
  jumpToInterpolationKnot(item);
}
document.querySelector("#previous-knot").addEventListener("click",()=>stepInterpolationKnot(-1));
document.querySelector("#next-knot").addEventListener("click",()=>stepInterpolationKnot(1));
document.querySelector("#editor-previous-knot").addEventListener("click",()=>stepInterpolationKnot(-1));
document.querySelector("#editor-next-knot").addEventListener("click",()=>stepInterpolationKnot(1));

function refreshCandidateControls() {
  const candidate=state.manifest?.registrationCandidate, queue=candidate?.control_review_queue||candidate?.z_review_queue||[], select=document.querySelector("#z-suggestion"),previous=select.value;
  select.replaceChildren(...queue.map((item,index)=>new Option(`${index+1}. RGB ${item.color_frame} → CT ${item.recommended_ct_frame} · ${item.status}`,String(item.color_frame))));
  const previousItem=queue.find(item=>String(item.color_frame)===previous),next=queue.find(item=>item.status==="pending");
  if(previousItem?.status==="pending")select.value=previous;else if(next)select.value=String(next.color_frame);
  const confirmed=queue.filter(x=>x.status==="confirmed").length,pending=queue.filter(x=>x.status==="pending").length;
  document.querySelector("#candidate-status").textContent=candidate?`${pending} remaining · ${confirmed} saved${candidate.status==="needs_rebuild"?" · rebuild required":""}`:"Not built";
  document.querySelector("#promote-candidate").disabled=!candidate?.qc?.accepted;
  updateZDetail();
}
function controlQueue(){const candidate=state.manifest?.registrationCandidate;return candidate?.control_review_queue||candidate?.z_review_queue||[];}
function selectedControl(){return controlQueue().find(x=>String(x.color_frame)===document.querySelector("#z-suggestion").value);}
function updateZDetail(){const item=selectedControl();if(!item){document.querySelector("#z-suggestion-detail").textContent="";return;}const alternatives=(item.alternatives||[]).filter(x=>x.ct_frame!==item.recommended_ct_frame).map(x=>x.ct_frame).slice(0,3).join(", ");document.querySelector("#z-suggestion-detail").textContent=`CT ${Number(item.recommended_ct_frame).toFixed(1)} · ${item.kind||"control point"}${alternatives?` · other CT ${alternatives}`:""}`;}
function openControl(item=selectedControl()){if(!item)return;alignmentReview.edit(item.color_frame);alignmentReview.changePlane(item.recommended_ct_frame);}
function stepControl(direction){const select=document.querySelector("#z-suggestion"),length=select.options.length;if(!length)return;select.selectedIndex=(select.selectedIndex+direction+length)%length;updateZDetail();openControl();}
document.querySelector("#z-suggestion").addEventListener("change",updateZDetail);
window.addEventListener("candidate-review-updated",refreshCandidateControls);
document.querySelector("#use-z-suggestion").addEventListener("click",()=>openControl());
document.querySelector("#control-previous").addEventListener("click",()=>stepControl(-1));
document.querySelector("#control-next").addEventListener("click",()=>stepControl(1));
document.querySelector("#reject-z-suggestion").addEventListener("click",async()=>{const frame=Number(document.querySelector("#z-suggestion").value);if(!Number.isFinite(frame))return;const response=await fetch(`/api/review/${state.subject}/candidate`,{method:"PUT",headers:{"content-type":"application/json"},body:JSON.stringify({color_frame:frame,status:"rejected"})});const value=await response.json();if(!response.ok)message(value.error);else{const item=controlQueue().find(x=>x.color_frame===frame);if(item)item.status="skipped";refreshCandidateControls();}});
document.querySelector("#promote-candidate").addEventListener("click",async()=>{const response=await fetch(`/api/review/${state.subject}/candidate`,{method:"PUT",headers:{"content-type":"application/json"},body:JSON.stringify({action:"promote"})});const value=await response.json();if(!response.ok)message(value.error);else await selectSubject(state.subject,"baseline");});
document.querySelector("#optimize-alignment").addEventListener("click",async()=>{const button=document.querySelector("#optimize-alignment");button.disabled=true;try{const response=await fetch(`/api/review/${state.subject}/optimize`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({memory_limit_gib:4})});const job=await response.json();if(!response.ok)throw new Error(job.error);while(true){await new Promise(resolve=>setTimeout(resolve,1200));const current=await(await fetch(`/api/review/jobs/${job.id}`)).json();document.querySelector("#candidate-status").textContent=current.log?.at(-1)?.trim().slice(-220)||current.state;if(current.state==="failed")throw new Error(current.error||current.log?.at(-1)||"Optimizer failed");if(current.state==="complete"){await selectSubject(state.subject,"candidate");break;}}}catch(error){message(error.message);}finally{button.disabled=false;}});
