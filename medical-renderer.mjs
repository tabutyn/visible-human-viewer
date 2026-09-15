import {
  GIB,
  HARD_MEMORY_LIMIT,
  TARGET_MEMORY_LIMIT,
  MAX_VOLUME_BYTES,
  brickDensityMask,
  brickOpacity,
  chooseVolumeLevel,
  clamp,
  displayBytes,
  planeAnglesFromNormal,
  planeNormalFromAngles,
  transferLut,
  volumeBytes,
} from "/volume-math.mjs";
import { buildVolumeAlignmentLut } from "/volume-alignment.mjs";

const alignmentWorkspace = document.querySelector("#alignment-workspace");
const medicalWorkspace = document.querySelector("#medical-workspace");
const alignmentControls = document.querySelector("#alignment-controls");
const medicalControls = document.querySelector("#medical-controls");
const alignmentTab = document.querySelector("#alignment-tab");
const medicalTab = document.querySelector("#medical-tab");
const align3dTab = document.querySelector("#align3d-tab");
const knotEditor = document.querySelector(".calibration-panel");
const knotEditorHome = knotEditor.parentElement;
const knotEditorNext = knotEditor.nextSibling;
const knotEditorHost = document.querySelector("#medical-knot-editor-host");
const knotBlend = document.querySelector("#knot-blend");
const subjectSelect = document.querySelector("#subject-select");
const presetSelect = document.querySelector("#volume-preset");
const buildButton = document.querySelector("#volume-build");
const moreDetailButton = document.querySelector("#volume-more-detail");
const retryButton = document.querySelector("#volume-retry");
const messageOutput = document.querySelector("#volume-message");
const performanceOutput = document.querySelector("#volume-performance");
const transferCanvas = document.querySelector("#transfer-canvas");
const colorList = document.querySelector("#transfer-colors");
const transferSelection = document.querySelector("#transfer-selection");
const windowCenterInput = document.querySelector("#volume-window-center");
const windowWidthInput = document.querySelector("#volume-window-width");
const customDensityControl = document.querySelector("#custom-density-control");
const customDensityInput = document.querySelector("#custom-density");
const customDensityValue = document.querySelector("#custom-density-value");
const planeAzimuthInput = document.querySelector("#plane-azimuth");
const planeInclinationInput = document.querySelector("#plane-inclination");
const planeDepthInput = document.querySelector("#plane-depth");
const planeAzimuthValue = document.querySelector("#plane-azimuth-value");
const planeInclinationValue = document.querySelector("#plane-inclination-value");
const planeDepthValue = document.querySelector("#plane-depth-value");
const medicalInfoButton = document.querySelector("#medical-info-toggle");
const medicalControlsGuide = document.querySelector("#medical-controls-guide");
const canvases = { volume: document.querySelector("#volume-canvas") };
const CUSTOM_DENSITY_MIN=-64,CUSTOM_DENSITY_MAX=1870,CUSTOM_DENSITY_DEFAULT=-64,CUSTOM_DENSITY_VERSION=2;

const PRESETS = {
  bone: {
    opacity: [{ hu: -1024, opacity: 0 }, { hu: 180, opacity: 0 }, { hu: 350, opacity: .06 }, { hu: 700, opacity: .42 }, { hu: 1600, opacity: .92 }, { hu: 3071, opacity: 1 }],
    colors: [{ hu: -1024, color: [0, 0, 0] }, { hu: 250, color: [.42, .2, .12] }, { hu: 700, color: [.92, .76, .55] }, { hu: 1800, color: [1, .98, .9] }, { hu: 3071, color: [1, 1, 1] }],
    window: [600, 2000], mip: false,
  },
  soft: {
    opacity: [{ hu: -1024, opacity: 0 }, { hu: -180, opacity: 0 }, { hu: -40, opacity: .03 }, { hu: 45, opacity: .2 }, { hu: 110, opacity: .34 }, { hu: 300, opacity: .06 }, { hu: 3071, opacity: 0 }],
    colors: [{ hu: -1024, color: [0, 0, 0] }, { hu: -80, color: [.35, .08, .05] }, { hu: 45, color: [.88, .36, .25] }, { hu: 140, color: [1, .72, .57] }, { hu: 3071, color: [1, .95, .85] }],
    window: [40, 400], mip: false,
  },
  lung: {
    opacity: [{ hu: -1024, opacity: 0 }, { hu: -900, opacity: .02 }, { hu: -760, opacity: .2 }, { hu: -420, opacity: .08 }, { hu: -120, opacity: 0 }, { hu: 500, opacity: .2 }, { hu: 1800, opacity: .7 }],
    colors: [{ hu: -1024, color: [0, 0, 0] }, { hu: -850, color: [.1, .25, .44] }, { hu: -500, color: [.35, .66, .82] }, { hu: 200, color: [.85, .55, .46] }, { hu: 1800, color: [1, 1, .9] }],
    window: [-600, 1500], mip: false,
  },
  body: {
    opacity: [{ hu: -1024, opacity: 0 }, { hu: -650, opacity: 0 }, { hu: -300, opacity: .015 }, { hu: -80, opacity: .06 }, { hu: 40, opacity: .12 }, { hu: 250, opacity: .08 }, { hu: 700, opacity: .32 }, { hu: 1800, opacity: .75 }],
    colors: [{ hu: -1024, color: [0, 0, 0] }, { hu: -250, color: [.25, .08, .04] }, { hu: 40, color: [.85, .3, .2] }, { hu: 350, color: [.96, .62, .42] }, { hu: 1500, color: [1, .96, .82] }],
    window: [250, 1800], mip: false,
  },
  mip: {
    opacity: [{ hu: -1024, opacity: 0 }, { hu: 3071, opacity: 1 }],
    colors: [{ hu: -1024, color: [0, 0, 0] }, { hu: 3071, color: [1, 1, 1] }],
    window: [500, 2500], mip: true,
  },
};

const state = {
  active: false,
  requestedDetail: null,
  detailSubject: null,
  nextDetail: null,
  liveAlignment: false,
  savedAlignmentRemap: false,
  alignmentSnapshot: null,
  alignmentDirty: true,
  alignmentRevision: 0,
  alignmentTable: null,
  alignmentBuffer: null,
  alignmentBufferBytes: 0,
  knotTexture: null,
  knotTextureKey: null,
  knotRequestedKey: null,
  knotTextureBytes: 0,
  knotRequest: null,
  focusedKnotKey: null,
  frameInFlight: false,
  initialized: false,
  loadingSerial: 0,
  manifest: null,
  level: null,
  adapter: null,
  device: null,
  format: null,
  pipeline: null,
  volumeTexture: null,
  rgbTextures: [],
  rgbSampler: null,
  opacityTexture: null,
  transferTexture: null,
  contexts: new Map(),
  uniforms: new Map(),
  bindGroups: new Map(),
  textureBytes: 0,
  rgbTextureBytes: 0,
  stagingBytes: 0,
  yaw: -Math.PI/2,
  pitch: 0,
  zoom: 1.15,
  cameraPan: [0, 0],
  cameraKeys: new Set(),
  cameraVelocity: [0, 0, 0],
  cameraMotionTime: 0,
  interactingUntil: 0,
  renderPending: false,
  renderAgain: false,
  queueScopeActive: false,
  gpuFault: null,
  gpuPhase: "initialization",
  loadTail: Promise.resolve(),
  lastFrame: 0,
  fps: 0,
  opacityPoints: [],
  colorPoints: [],
  mip: false,
  rgbReady: false,
  rgbManifest: null,
  rgbLevel: null,
  planeOffset: 0,
  planeRotateX: 0,
  planeRotateY: 0,
  planeNormal: [0, 0, 1],
  customDensity: CUSTOM_DENSITY_DEFAULT,
  selectedOpacity: -1,
  transferDragging: false,
  window: [600, 2000],
};
let viewSaveTimer=0;
const dataFetch=(input,init)=>window.visibleHumanReleaseData?.fetch(input,init)??fetch(input,init);

function currentMedicalView() {
  return {preset:presetSelect.value,yaw:state.yaw,pitch:state.pitch,zoom:state.zoom,cameraPan:[...state.cameraPan],planeOffset:state.planeOffset,planeNormal:[...state.planeNormal]};
}

async function restoreMedicalView(subject=subjectSelect.value||"male") {
  try{
    const response=await dataFetch(`/api/subjects/${encodeURIComponent(subject)}/medical-view`);if(!response.ok)throw new Error(`HTTP ${response.status}`);const value=await response.json();
    if(Number.isFinite(value.yaw))state.yaw=value.yaw;if(Number.isFinite(value.pitch))state.pitch=clamp(value.pitch,-1.45,1.45);if(Number.isFinite(value.zoom))state.zoom=clamp(value.zoom,.3,8);
    if(Array.isArray(value.cameraPan)&&value.cameraPan.length===2&&value.cameraPan.every(Number.isFinite))state.cameraPan=[...value.cameraPan];
    if(Number.isFinite(value.planeOffset))state.planeOffset=clamp(value.planeOffset,-1,1);
    if(Array.isArray(value.planeNormal)&&value.planeNormal.length===3&&value.planeNormal.every(Number.isFinite)&&Math.hypot(...value.planeNormal)>.0001){const length=Math.hypot(...value.planeNormal);state.planeNormal=value.planeNormal.map((item)=>item/length);}
    return ["bone","soft","lung","body","mip","custom"].includes(value.preset)?value.preset:"lung";
  }catch{return "lung";}
}

function saveMedicalView() {
  // Alignment inspection must not replace the user's saved rendering setup.
  if (!state.active || state.liveAlignment || window.visibleHumanConfig?.readOnly) return;
  clearTimeout(viewSaveTimer);viewSaveTimer=setTimeout(()=>dataFetch(`/api/subjects/${encodeURIComponent(subjectSelect.value||"male")}/medical-view`,{method:"PUT",headers:{"content-type":"application/json"},body:JSON.stringify(currentMedicalView())}).catch(()=>{}),120);
}

const SHADER = /* wgsl */`
struct Params {
  dimensions: vec4f,
  halfExtents: vec4f,
  cameraRight: vec4f,
  cameraUp: vec4f,
  cameraForward: vec4f,
  custom: vec4f,
  viewWindow: vec4f,
  render: vec4f,
  brickGrid: vec4f,
  clipPlane: vec4f,
  clipOptions: vec4f,
  remapInfo: vec4f,
  knot: vec4f,
  knotInverse0: vec4f,
  knotInverse1: vec4f,
};
@group(0) @binding(0) var volumeTexture: texture_3d<u32>;
@group(0) @binding(1) var transferTexture: texture_2d<f32>;
@group(0) @binding(2) var opacityTexture: texture_3d<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var rgbRedTexture: texture_3d<f32>;
@group(0) @binding(5) var rgbGreenTexture: texture_3d<f32>;
@group(0) @binding(6) var rgbBlueTexture: texture_3d<f32>;
@group(0) @binding(7) var rgbSampler: sampler;
struct AlignmentRow { first: vec4f, second: vec4f };
@group(0) @binding(8) var<storage, read> alignmentRows: array<AlignmentRow>;
@group(0) @binding(9) var knotTexture: texture_2d<f32>;

struct VertexOut { @builtin(position) position: vec4f, @location(0) uv: vec2f };
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOut {
  var positions = array<vec2f, 3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var output: VertexOut;
  output.position = vec4f(positions[index], 0.0, 1.0);
  output.uv = positions[index] * vec2f(.5,-.5) + vec2f(.5,.5);
  return output;
}
fn storedValue(uvValue: vec3f) -> f32 {
  let maximum = max(params.dimensions.xyz - vec3f(1.0), vec3f(1.0));
  let point = clamp(uvValue, vec3f(0.0), vec3f(1.0)) * maximum;
  let low = vec3i(floor(point));
  let high = min(low + vec3i(1), vec3i(params.dimensions.xyz) - vec3i(1));
  let fraction = fract(point);
  let a = f32(textureLoad(volumeTexture, low, 0).r);
  let b = f32(textureLoad(volumeTexture, vec3i(high.x, low.y, low.z), 0).r);
  let c = f32(textureLoad(volumeTexture, vec3i(low.x, high.y, low.z), 0).r);
  let d = f32(textureLoad(volumeTexture, vec3i(high.x, high.y, low.z), 0).r);
  let e = f32(textureLoad(volumeTexture, vec3i(low.x, low.y, high.z), 0).r);
  let f = f32(textureLoad(volumeTexture, vec3i(high.x, low.y, high.z), 0).r);
  let g = f32(textureLoad(volumeTexture, vec3i(low.x, high.y, high.z), 0).r);
  let h = f32(textureLoad(volumeTexture, high, 0).r);
  return mix(mix(mix(a,b,fraction.x),mix(c,d,fraction.x),fraction.y),mix(mix(e,f,fraction.x),mix(g,h,fraction.x),fraction.y),fraction.z);
}
fn transfer(value: f32) -> vec4f {
  return textureLoad(transferTexture, vec2i(i32(clamp(round(value), 0.0, 4095.0)), 0), 0);
}
fn gradient(uvValue: vec3f) -> vec3f {
  let delta = 1.0 / max(params.dimensions.xyz - vec3f(1.0), vec3f(1.0));
  return vec3f(
    storedValue(uvValue + vec3f(delta.x,0,0)) - storedValue(uvValue - vec3f(delta.x,0,0)),
    storedValue(uvValue + vec3f(0,delta.y,0)) - storedValue(uvValue - vec3f(0,delta.y,0)),
    storedValue(uvValue + vec3f(0,0,delta.z)) - storedValue(uvValue - vec3f(0,0,delta.z))
  );
}
fn rayBox(origin: vec3f, direction: vec3f, halfSize: vec3f) -> vec2f {
  let safe = select(vec3f(1e-6), direction, abs(direction) > vec3f(1e-6));
  let first = (-halfSize - origin) / safe;
  let second = (halfSize - origin) / safe;
  let nearValue = max(max(min(first.x,second.x),min(first.y,second.y)),min(first.z,second.z));
  let farValue = min(min(max(first.x,second.x),max(first.y,second.y)),max(first.z,second.z));
  return vec2f(nearValue,farValue);
}
fn rgbAtWorld(world: vec3f) -> vec4f {
  var uvValue=world/(params.halfExtents.xyz*2.0)+vec3f(.5);
  if (any(uvValue<vec3f(0.0)) || any(uvValue>vec3f(1.0))) { return vec4f(0.0); }
  if (params.remapInfo.y>.5) {
    let position=uvValue.z*max(params.remapInfo.x-1.0,0.0);
    let low=u32(floor(position));let high=min(low+1u,u32(params.remapInfo.x)-1u);
    let left=alignmentRows[low];let right=alignmentRows[high];
    let fraction=fract(position);
    if ((fraction<.0001 && left.second.w<.5) || (fraction>=.0001 && (left.second.w<.5 || right.second.w<.5 || left.second.w!=right.second.w))) { return vec4f(0.0); }
    let first=mix(left.first,right.first,fract(position));let second=mix(left.second,right.second,fract(position));
    let xy1=vec3f(uvValue.xy,1.0);
    uvValue=vec3f(dot(first.xyz,xy1),dot(second.xyz,xy1),first.w);
    if (any(uvValue<vec3f(0.0)) || any(uvValue>vec3f(1.0))) { return vec4f(0.0); }
  }
  let size=vec3f(textureDimensions(rgbRedTexture));
  let sampleUv=(uvValue*(size-vec3f(1.0))+vec3f(.5))/size;
  return vec4f(textureSampleLevel(rgbRedTexture,rgbSampler,sampleUv,0.0).r,textureSampleLevel(rgbGreenTexture,rgbSampler,sampleUv,0.0).r,textureSampleLevel(rgbBlueTexture,rgbSampler,sampleUv,0.0).r,1.0);
}
fn planeColour(world: vec3f) -> vec4f {
  let uvValue=world/(params.halfExtents.xyz*2.0)+vec3f(.5);
  if (any(uvValue<vec3f(0.0)) || any(uvValue>vec3f(1.0))) { return vec4f(0.0); }
  let source=rgbAtWorld(world);
  let tissue=smoothstep(250.0,550.0,storedValue(uvValue));
  let opacity=tissue*.98*source.a;
  return vec4f(source.rgb*opacity,opacity);
}
fn volumeFragment(uvValue: vec2f) -> vec4f {
  let screen = (uvValue * 2.0 - vec2f(1.0)) * vec2f(params.viewWindow.x, 1.0) / params.halfExtents.w + vec2f(params.clipOptions.y,params.clipOptions.w);
  let forward = normalize(params.cameraForward.xyz);
  let origin = -forward * 1.6 + params.cameraRight.xyz * screen.x + params.cameraUp.xyz * screen.y;
  let bounds = rayBox(origin, forward, params.halfExtents.xyz);
  if (bounds.y <= max(bounds.x,0.0)) { return vec4f(0,0,0,1); }
  var nearDistance=max(bounds.x,0.0);
  var farDistance=bounds.y;
  var planeAtFront=false;
  var planeAtBack=false;
  var planeDistance=0.0;
  let clipNormal=normalize(params.clipPlane.xyz);
  if (params.clipOptions.x>.5) {
    let nearSide=dot(origin+forward*nearDistance,clipNormal)-params.clipPlane.w;
    let farSide=dot(origin+forward*farDistance,clipNormal)-params.clipPlane.w;
    if (nearSide>0.0 && farSide>0.0) { return vec4f(0,0,0,1); }
    let denominator=dot(forward,clipNormal);
    if (abs(denominator)>1e-6) {
      planeDistance=(params.clipPlane.w-dot(origin,clipNormal))/denominator;
      if (nearSide>0.0 && farSide<=0.0) { nearDistance=max(nearDistance,planeDistance); planeAtFront=true; }
      if (nearSide<=0.0 && farSide>0.0) { farDistance=min(farDistance,planeDistance); planeAtBack=true; }
    }
  }
  var distance=nearDistance;
  var accumulated = vec4f(0.0);
  let customMode=params.custom.w>.5;
  if (planeAtFront && params.clipOptions.z>.5) {
    let plane=planeColour(origin+forward*planeDistance);
    if (customMode && plane.a>.01) { return vec4f(plane.rgb/max(plane.a,.001),1); }
    accumulated=accumulated+(1.0-accumulated.a)*plane;
  }
  var maximum = 0.0;
  var previousValue = 0.0;
  var previousDistance = 0.0;
  var hasPrevious = false;
  let voxelStep = (params.halfExtents.xyz * 2.0) / max(params.dimensions.xyz-vec3f(1.0),vec3f(1.0));
  let referenceStep = max(min(min(voxelStep.x,voxelStep.y),voxelStep.z), 1e-6);
  for (var sampleIndex=0u; sampleIndex<2048u; sampleIndex++) {
    if (sampleIndex >= u32(params.render.y) || distance > farDistance || accumulated.a > .985) { break; }
    let world = origin + forward * distance;
    let volumeUv = world / (params.halfExtents.xyz * 2.0) + vec3f(.5);
    let brick = min(vec3i(floor(volumeUv*params.brickGrid.xyz)),vec3i(params.brickGrid.xyz)-vec3i(1));
    if (textureLoad(opacityTexture,max(brick,vec3i(0)),0).r == 0u) { hasPrevious=false; distance += params.render.x * 8.0; continue; }
    let value = storedValue(volumeUv);
    if (customMode) {
      if (hasPrevious) {
        let before=previousValue-params.custom.z;
        let after=value-params.custom.z;
        let crossed=(before<0.0 && after>=0.0) || (before>0.0 && after<=0.0);
        if (crossed && abs(value-previousValue)>.25 && params.clipOptions.z>.5) {
          let fraction=clamp((params.custom.z-previousValue)/(value-previousValue),0.0,1.0);
          let hitWorld=origin+forward*mix(previousDistance,distance,fraction);
          let rgb=rgbAtWorld(hitWorld);
          if (rgb.a>.5) { return vec4f(rgb.rgb,1); }
        }
      }
      previousValue=value;previousDistance=distance;hasPrevious=true;
    } else if (params.render.w > .5) { maximum = max(maximum,value); }
    else {
      let rawColour=transfer(value);
      let opacity=1.0-pow(max(0.0,1.0-rawColour.a),max(params.render.x/referenceStep,.1));
      var rgb=rawColour.rgb;
      if (opacity > .015 && params.render.z > .5) {
        let normal = normalize(gradient(volumeUv)+vec3f(1e-5));
        rgb=rgb*(.34+.66*abs(dot(normal,normalize(vec3f(.4,-.5,.8)))));
      }
      let colour=vec4f(rgb*opacity,opacity);
      accumulated=accumulated+(1.0-accumulated.a)*colour;
    }
    distance += params.render.x;
  }
  if (planeAtBack && params.clipOptions.z>.5 && params.render.w<.5) {
    let plane=planeColour(origin+forward*planeDistance);
    accumulated=accumulated+(1.0-accumulated.a)*plane;
  }
  if (customMode) { return vec4f(accumulated.rgb,1); }
  if (params.render.w > .5) { let value=clamp((maximum-1024.0-(params.viewWindow.z-params.viewWindow.w*.5))/params.viewWindow.w,0.0,1.0); return vec4f(vec3f(value),1); }
  return vec4f(accumulated.rgb,1);
}
@fragment fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  let base=volumeFragment(input.uv);
  if (params.knot.y<.5 || params.knot.z<.5 || abs(params.cameraForward.z)<1e-5) { return base; }
  let screen=(input.uv*2.0-vec2f(1.0))*vec2f(params.viewWindow.x,1.0)/params.halfExtents.w+vec2f(params.clipOptions.y,params.clipOptions.w);
  let forward=normalize(params.cameraForward.xyz);
  let origin=-forward*1.6+params.cameraRight.xyz*screen.x+params.cameraUp.xyz*screen.y;
  let distance=(params.knot.x-origin.z)/forward.z;
  if (distance<0.0) { return base; }
  let world=origin+forward*distance;
  let ctUv=world/(params.halfExtents.xyz*2.0)+vec3f(.5);
  if (any(ctUv<vec3f(0.0)) || any(ctUv>vec3f(1.0))) { return base; }
  let rgbUv=vec2f(dot(params.knotInverse0.xyz,vec3f(ctUv.xy,1.0)),dot(params.knotInverse1.xyz,vec3f(ctUv.xy,1.0)));
  var rgb=vec3f(0.0);
  if (all(rgbUv>=vec2f(0.0)) && all(rgbUv<=vec2f(1.0))) { rgb=textureSampleLevel(knotTexture,rgbSampler,rgbUv,0.0).rgb; }
  let stored=storedValue(ctUv);
  let gray=clamp((stored-1024.0-(params.viewWindow.z-params.viewWindow.w*.5))/params.viewWindow.w,0.0,1.0);
  let colour=mix(vec3f(gray),rgb,params.knot.w);
  // Use the actual source slice, including displaced anatomy outside the bake.
  let opacity=select(.28,.94,stored>400.0 || max(rgb.r,max(rgb.g,rgb.b))>.12);
  return vec4f(mix(base.rgb,colour,opacity),1.0);
}`;

function clonePreset(value) {
  return {
    opacity: value.opacity.map((point) => ({ ...point })),
    colors: value.colors.map((point) => ({ hu: point.hu, color: [...point.color] })),
    window: [...value.window],
    mip: value.mip,
  };
}

function setMessage(value) { messageOutput.textContent = value; }
function storageKey() { return `visible-human-transfer-v1:${subjectSelect.value || "male"}`; }

function syncCustomDensityControl() {
  customDensityControl.hidden=presetSelect.value!=="custom";
  customDensityInput.value=String(Math.round(state.customDensity));
  customDensityValue.textContent=`${Math.round(state.customDensity)} HU`;
}

function applyPreset(name, allowSaved = true) {
  let value = PRESETS[name] ? clonePreset(PRESETS[name]) : null;
  if (name === "custom" && allowSaved) {
    try { value = JSON.parse(localStorage.getItem(storageKey())); } catch {}
  }
  if (!value) value = clonePreset(PRESETS.bone);
  if(name==="custom")state.customDensity=value.densityRangeVersion===CUSTOM_DENSITY_VERSION&&Number.isFinite(Number(value.density))?clamp(Number(value.density),CUSTOM_DENSITY_MIN,CUSTOM_DENSITY_MAX):CUSTOM_DENSITY_DEFAULT;
  state.opacityPoints = value.opacity;
  state.colorPoints = value.colors;
  state.window = value.window;
  state.mip = Boolean(value.mip);
  if(name==="custom")state.mip=false;
  windowCenterInput.value = String(state.window[0]);
  windowWidthInput.value = String(state.window[1]);
  syncCustomDensityControl();
  refreshColorRows();
  drawTransferEditor();
  uploadTransfer();
}

function saveCustom() {
  presetSelect.value = "custom";
  syncCustomDensityControl();
  localStorage.setItem(storageKey(), JSON.stringify({ opacity: state.opacityPoints, colors: state.colorPoints, window: state.window, mip: state.mip, density: state.customDensity, densityRangeVersion:CUSTOM_DENSITY_VERSION }));
  saveMedicalView();
}

function hex(color) { return `#${color.map((value) => Math.round(clamp(value,0,1)*255).toString(16).padStart(2,"0")).join("")}`; }
function colorFromHex(value) { return [1,3,5].map((start) => parseInt(value.slice(start,start+2),16)/255); }

function refreshColorRows() {
  state.colorPoints.sort((left,right) => left.hu-right.hu);
  colorList.replaceChildren(...state.colorPoints.map((point,index) => {
    const row=document.createElement("div"); row.className="transfer-color-row";
    const hu=document.createElement("input"); hu.type="number"; hu.min="-1024"; hu.max="3071"; hu.step="1"; hu.value=String(Math.round(point.hu)); hu.setAttribute("aria-label","Colour stop HU");
    const color=document.createElement("input"); color.type="color"; color.value=hex(point.color); color.setAttribute("aria-label",`Colour at ${point.hu} HU`);
    const remove=document.createElement("button"); remove.type="button"; remove.className="danger"; remove.textContent="×"; remove.title="Remove colour stop"; remove.disabled=state.colorPoints.length<=2;
    hu.onchange=()=>{point.hu=clamp(Number(hu.value),-1024,3071);saveCustom();refreshColorRows();uploadTransfer();};
    color.oninput=()=>{point.color=colorFromHex(color.value);saveCustom();uploadTransfer();drawTransferEditor();};
    remove.onclick=()=>{state.colorPoints.splice(index,1);saveCustom();refreshColorRows();uploadTransfer();};
    row.append(hu,color,remove);return row;
  }));
}

function transferPoint(event) {
  const rect=transferCanvas.getBoundingClientRect();
  return {
    x:(event.clientX-rect.left)/rect.width,
    y:(event.clientY-rect.top)/rect.height,
    hu:clamp(-1024+(event.clientX-rect.left)/rect.width*4095,-1024,3071),
    opacity:clamp(1-(event.clientY-rect.top)/rect.height,0,1),
  };
}

function drawTransferEditor() {
  const ratio=devicePixelRatio||1,width=Math.max(1,transferCanvas.clientWidth),height=Math.max(1,transferCanvas.clientHeight);
  transferCanvas.width=Math.round(width*ratio);transferCanvas.height=Math.round(height*ratio);
  const context=transferCanvas.getContext("2d");context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,width,height);
  const bins=state.manifest?.histogram?.bins||[], maximum=Math.max(1,...bins);
  context.fillStyle="#38414a";
  for(let index=0;index<bins.length;index+=8){const value=Math.max(...bins.slice(index,index+8)),bar=Math.log1p(value)/Math.log1p(maximum)*height;context.fillRect(index/bins.length*width,height-bar,Math.max(1,8/bins.length*width),bar);}
  const points=[...state.opacityPoints].sort((a,b)=>a.hu-b.hu);context.beginPath();
  points.forEach((point,index)=>{const x=(point.hu+1024)/4095*width,y=(1-point.opacity)*height;if(index)context.lineTo(x,y);else context.moveTo(x,y);});
  context.strokeStyle="#adf05c";context.lineWidth=2;context.stroke();
  points.forEach((point)=>{const original=state.opacityPoints.indexOf(point),x=(point.hu+1024)/4095*width,y=(1-point.opacity)*height;context.beginPath();context.arc(x,y,original===state.selectedOpacity?6:4,0,Math.PI*2);context.fillStyle=original===state.selectedOpacity?"#fff":"#adf05c";context.fill();});
  context.fillStyle="#98a3af";context.font="10px system-ui";context.fillText("−1024 HU",5,height-6);context.fillText("3071",width-32,height-6);
}

function makeLut() { return transferLut(state.opacityPoints,state.colorPoints); }
function dilatedBrickMask(level,values) {
  const [gx,gy,gz]=level.grid,raw=new Uint8Array(gx*gy*gz);
  level.bricks.forEach((brick,index)=>{raw[brick.x+gx*(brick.y+gy*brick.z)]=values[index];});
  const output=raw.slice();
  for(let z=0;z<gz;z++)for(let y=0;y<gy;y++)for(let x=0;x<gx;x++)if(raw[x+gx*(y+gy*z)])for(let dz=-1;dz<=1;dz++)for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    const nx=x+dx,ny=y+dy,nz=z+dz;if(nx>=0&&ny>=0&&nz>=0&&nx<gx&&ny<gy&&nz<gz)output[nx+gx*(ny+gy*nz)]=255;
  }
  return output;
}

function activeBrickMask(level) {
  if(presetSelect.value==="custom")return dilatedBrickMask(level,brickDensityMask(level.bricks,state.customDensity));
  return dilatedBrickMask(level,brickOpacity(level.bricks,makeLut()));
}

function packedRows(values,width,height,depth) {
  const bytesPerRow=Math.ceil(width/256)*256,output=new Uint8Array(bytesPerRow*height*depth);
  for(let z=0;z<depth;z++)for(let y=0;y<height;y++)output.set(values.subarray(width*(y+height*z),width*(y+height*z+1)),bytesPerRow*(y+height*z));
  return {bytes:output,bytesPerRow};
}

function uploadTransfer() {
  drawTransferEditor();
  if(!state.device||!state.transferTexture)return scheduleRender();
  state.gpuPhase="transfer-function upload";
  const lut=makeLut();state.device.queue.writeTexture({texture:state.transferTexture},lut,{bytesPerRow:4096*4},{width:4096,height:1});
  if(state.opacityTexture&&state.level){
    const [x,y,z]=state.level.grid,packed=packedRows(activeBrickMask(state.level),x,y,z);
    state.device.queue.writeTexture({texture:state.opacityTexture},packed.bytes,{bytesPerRow:packed.bytesPerRow,rowsPerImage:y},{width:x,height:y,depthOrArrayLayers:z});
  }
  scheduleRender();
}

function clearDeviceResources({ destroy = true } = {}) {
  state.knotRequest?.abort();state.knotRequest=null;state.knotRequestedKey=null;state.knotTextureKey=null;
  if(destroy){
    state.volumeTexture?.destroy();
    for(const texture of state.rgbTextures)texture.destroy();
    state.opacityTexture?.destroy();
    state.transferTexture?.destroy();
    state.alignmentBuffer?.destroy();state.knotTexture?.destroy();
    for(const buffer of state.uniforms.values())buffer.destroy();
  }
  state.volumeTexture=null;state.rgbTextures=[];state.rgbSampler=null;state.rgbReady=false;state.rgbManifest=null;state.rgbLevel=null;state.opacityTexture=null;state.transferTexture=null;state.pipeline=null;
  delete medicalWorkspace.dataset.rgbVolume;delete medicalWorkspace.dataset.rgbRendered;delete medicalWorkspace.dataset.rgbLevel;delete medicalWorkspace.dataset.firstRgbLevel;delete medicalWorkspace.dataset.renderedRgbLevel;delete medicalWorkspace.dataset.firstRenderedRgbLevel;delete medicalWorkspace.dataset.progressiveLevel;
  state.contexts.clear();state.uniforms.clear();state.bindGroups.clear();state.level=null;
  state.alignmentBuffer=null;state.alignmentBufferBytes=0;state.alignmentTable=null;state.alignmentDirty=true;state.knotTexture=null;state.knotTextureBytes=0;state.frameInFlight=false;
  state.textureBytes=0;state.rgbTextureBytes=0;state.stagingBytes=0;state.renderPending=false;state.renderAgain=false;state.queueScopeActive=false;state.lastFrame=0;
  updatePerformance();
}

function reportGpuFault(message,device=state.device) {
  if(device!==state.device||state.gpuFault)return;
  state.gpuFault=message;state.loadingSerial++;state.renderPending=false;state.renderAgain=false;
  medicalWorkspace.dataset.gpuError=message;
  setMessage(`WebGPU ${message}. Reopen Medical Rendering to retry.`);
}

function discardFaultedDevice() {
  const device=state.device;
  clearDeviceResources();state.device=null;state.adapter=null;state.initialized=false;state.gpuFault=null;
  delete medicalWorkspace.dataset.gpuError;
  try{device?.destroy();}catch{}
}

function emptyRgbTextures(device) {
  const textures=["red","green","blue"].map((channel)=>device.createTexture({label:`Empty RGB ${channel}`,size:[1,1,1],dimension:"3d",format:"r8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}));
  for(const texture of textures)device.queue.writeTexture({texture},new Uint8Array([0]),{},{width:1,height:1,depthOrArrayLayers:1});
  return textures;
}

async function initializeGpu() {
  if(state.gpuFault)discardFaultedDevice();
  if(state.initialized)return;
  if(!navigator.gpu)throw new Error("WebGPU is unavailable. Open this local app in current Chrome.");
  state.adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
  if(!state.adapter)throw new Error("Chrome could not obtain a WebGPU adapter.");
  // Dawn clears a newly allocated 3D texture through an internal buffer as
  // large as the texture. Request the adapter's supported limit explicitly;
  // the allocation ledger below still counts that transient second copy.
  const maxBufferSize=Math.min(MAX_VOLUME_BYTES,state.adapter.limits.maxBufferSize);
  const device=await state.adapter.requestDevice({requiredLimits:{maxBufferSize}});state.device=device;state.format=navigator.gpu.getPreferredCanvasFormat();
  state.gpuFault=null;delete medicalWorkspace.dataset.gpuError;
  device.addEventListener("uncapturederror",(event)=>{event.preventDefault();reportGpuFault(`${state.gpuPhase} failed: ${event.error.message}`,device);});
  device.lost.then((info)=>{
    if(state.device!==device)return;
    const detail=state.gpuFault||`device lost: ${info.message||info.reason}`;
    clearDeviceResources({destroy:false});state.initialized=false;state.device=null;state.adapter=null;state.loadingSerial++;
    state.gpuFault=detail;medicalWorkspace.dataset.gpuError=detail;setMessage(`WebGPU ${detail}. Reopen Medical Rendering to retry.`);
  });
  try{
    state.gpuPhase="shader compilation";
    const module=device.createShaderModule({label:"CT volume shader",code:SHADER});
    if(module.getCompilationInfo){
      const info=await module.getCompilationInfo(),errors=info.messages.filter((item)=>item.type==="error");
      if(errors.length)throw new Error(`CT shader compilation failed: ${errors.map((item)=>`${item.lineNum}:${item.linePos} ${item.message}`).join("; ")}`);
    }
    const descriptor={label:"CT volume renderer",layout:"auto",vertex:{module,entryPoint:"vertexMain"},fragment:{module,entryPoint:"fragmentMain",targets:[{format:state.format}]},primitive:{topology:"triangle-list"}};
    state.gpuPhase="pipeline creation";
    state.pipeline=device.createRenderPipelineAsync?await device.createRenderPipelineAsync(descriptor):device.createRenderPipeline(descriptor);
    state.gpuPhase="canvas configuration";
    for(const [name,canvas] of Object.entries(canvases)){
      const context=canvas.getContext("webgpu");context.configure({device,format:state.format,alphaMode:"opaque"});state.contexts.set(name,context);
      state.uniforms.set(name,device.createBuffer({label:`${name} uniforms`,size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));
    }
    state.transferTexture=device.createTexture({label:"CT transfer function",size:[4096,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    state.rgbTextures=emptyRgbTextures(device);
    state.rgbSampler=device.createSampler({magFilter:"linear",minFilter:"linear"});
    state.alignmentBuffer=device.createBuffer({label:"Live RGB alignment",size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});state.alignmentBufferBytes=32;
    state.knotTexture=device.createTexture({label:"Selected source RGB",size:[1,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});
    device.queue.writeTexture({texture:state.knotTexture},new Uint8Array(4),{},{width:1,height:1});
    state.initialized=true;uploadTransfer();
  }catch(error){
    if(state.device===device){clearDeviceResources();state.device=null;state.adapter=null;state.initialized=false;try{device.destroy();}catch{}}
    throw error;
  }
}

function makeOpacityTexture(level) {
  state.gpuPhase="opacity-grid upload";
  const texture=state.device.createTexture({label:"CT brick opacity",size:level.grid,dimension:"3d",format:"r8uint",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const [x,y,z]=level.grid,packed=packedRows(activeBrickMask(level),x,y,z);
  state.device.queue.writeTexture({texture},packed.bytes,{bytesPerRow:packed.bytesPerRow,rowsPerImage:y},{width:x,height:y,depthOrArrayLayers:z});
  return texture;
}

function createBindGroups() {
  if(!state.volumeTexture||!state.opacityTexture||state.rgbTextures.length!==3||!state.rgbSampler)return;
  state.gpuPhase="resource binding";
  for(const name of Object.keys(canvases))state.bindGroups.set(name,state.device.createBindGroup({layout:state.pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:state.volumeTexture.createView()},
    {binding:1,resource:state.transferTexture.createView()},
    {binding:2,resource:state.opacityTexture.createView()},
    {binding:3,resource:{buffer:state.uniforms.get(name)}},
    {binding:4,resource:state.rgbTextures[0].createView()},
    {binding:5,resource:state.rgbTextures[1].createView()},
    {binding:6,resource:state.rgbTextures[2].createView()},
    {binding:7,resource:state.rgbSampler},
    {binding:8,resource:{buffer:state.alignmentBuffer}},
    {binding:9,resource:state.knotTexture.createView()},
  ]}));
}

async function verifyReleaseBrick(bytes,brick,label){
  if(!window.visibleHumanReleaseData)return;
  if(bytes.byteLength!==brick.bytes)throw new Error(`${label}: received ${bytes.byteLength} bytes; expected ${brick.bytes}`);
  const digest=new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),actual=[...digest].map(value=>value.toString(16).padStart(2,"0")).join("");
  if(actual!==brick.sha256)throw new Error(`${label}: decoded SHA-256 mismatch`);
}

async function fetchBrick(subject,level,brick,serial) {
  const version=state.manifest.inputs?.ct_sha256||"current";
  const response=await dataFetch(`/api/subjects/${encodeURIComponent(subject)}/volume/bricks/${level.level}/${brick.x}/${brick.y}/${brick.z}?v=${version}`);
  if(!response.ok)throw new Error(`volume brick ${level.level}/${brick.x}/${brick.y}/${brick.z}: HTTP ${response.status}`);
  const bytes=new Uint8Array(await response.arrayBuffer());if(serial!==state.loadingSerial)return null;await verifyReleaseBrick(bytes,brick,`CT L${level.level}/${brick.x}/${brick.y}/${brick.z}`);
  const sourceRow=state.manifest.brick_size*2,bytesPerRow=Math.ceil(sourceRow/256)*256;
  if(bytesPerRow===sourceRow)return {brick,bytes,bytesPerRow};
  const padded=new Uint8Array(bytesPerRow*state.manifest.brick_size*brick.extent[2]);
  for(let z=0;z<brick.extent[2];z++)for(let y=0;y<state.manifest.brick_size;y++)padded.set(bytes.subarray(sourceRow*(y+state.manifest.brick_size*z),sourceRow*(y+state.manifest.brick_size*z+1)),bytesPerRow*(y+state.manifest.brick_size*z));
  return {brick,bytes:padded,bytesPerRow};
}

async function fetchRgbBrick(subject,level,brick,serial,manifest) {
  const version=manifest.inputs?.reviewed_sha256||manifest.inputs?.candidate_sha256||"current";
  const response=await dataFetch(`/api/subjects/${encodeURIComponent(subject)}/volume/rgb/bricks/${level.level}/${brick.x}/${brick.y}/${brick.z}?v=${version}`);
  if(!response.ok)throw new Error(`RGB volume brick ${level.level}/${brick.x}/${brick.y}/${brick.z}: HTTP ${response.status}`);
  const bytes=new Uint8Array(await response.arrayBuffer());if(serial!==state.loadingSerial)return null;await verifyReleaseBrick(bytes,brick,`RGB L${level.level}/${brick.x}/${brick.y}/${brick.z}`);
  const planeBytes=manifest.brick_size*manifest.brick_size*brick.extent[2];
  if(bytes.byteLength!==planeBytes*3)throw new Error(`RGB brick ${brick.x}/${brick.y}/${brick.z} has ${bytes.byteLength} bytes; expected ${planeBytes*3}`);
  const channels=[];
  for(let channel=0;channel<3;channel++)channels.push(packedRows(bytes.subarray(channel*planeBytes,(channel+1)*planeBytes),manifest.brick_size,manifest.brick_size,brick.extent[2]));
  return {brick,channels};
}

async function uploadRgbLevel(subject,manifest,level,serial) {
  const device=state.device,required=volumeBytes(level.dimensions,3),channelBytes=volumeBytes(level.dimensions,1);
  if(required+channelBytes+state.textureBytes+state.rgbTextureBytes>TARGET_MEMORY_LIMIT)throw new Error(`CT + RGB allocation would exceed the ${displayBytes(TARGET_MEMORY_LIMIT)} working target.`);
  const pending=["red","green","blue"].map((channel)=>device.createTexture({label:`${subject} RGB ${channel} level ${level.level}`,size:level.dimensions,dimension:"3d",format:"r8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}));
  let completed=0,adopted=false;
  try{
    for(let start=0;start<level.bricks.length&&serial===state.loadingSerial;start+=4){
      const batch=(await Promise.all(level.bricks.slice(start,start+4).map((brick)=>fetchRgbBrick(subject,level,brick,serial,manifest)))).filter(Boolean);
      if(!batch.length)break;
      state.stagingBytes=batch.reduce((total,item)=>total+item.channels.reduce((sum,channel)=>sum+channel.bytes.byteLength,0),0);updatePerformance();
      state.gpuPhase="RGB volume upload";state.queueScopeActive=true;device.pushErrorScope("validation");
      let staging=null,scopePopped=false;
      try{
        staging=device.createBuffer({size:state.stagingBytes,usage:GPUBufferUsage.COPY_SRC,mappedAtCreation:true});
        const mapped=new Uint8Array(staging.getMappedRange());let offset=0;
        for(const item of batch)for(let channel=0;channel<3;channel++){
          const packed=item.channels[channel];mapped.set(packed.bytes,offset);packed.offset=offset;offset+=packed.bytes.byteLength;
        }
        staging.unmap();const encoder=device.createCommandEncoder();
        for(const item of batch)for(let channel=0;channel<3;channel++)encoder.copyBufferToTexture(
          {buffer:staging,offset:item.channels[channel].offset,bytesPerRow:item.channels[channel].bytesPerRow,rowsPerImage:manifest.brick_size},
          {texture:pending[channel],origin:[item.brick.x*manifest.brick_size,item.brick.y*manifest.brick_size,item.brick.z*manifest.brick_size]},
          {width:item.brick.extent[0],height:item.brick.extent[1],depthOrArrayLayers:item.brick.extent[2]},
        );
        device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();const validation=await device.popErrorScope();scopePopped=true;
        if(validation)throw new Error(`RGB volume upload validation failed: ${validation.message}`);
      }finally{
        if(!scopePopped)await device.popErrorScope().catch(()=>null);
        staging?.destroy();state.stagingBytes=0;state.queueScopeActive=false;
        if(state.renderAgain){state.renderAgain=false;scheduleRender();}
      }
      completed+=batch.length;setMessage(`Loading registered RGB · ${completed}/${level.bricks.length} bricks`);
    }
    if(serial!==state.loadingSerial)return false;
    const old=state.rgbTextures;state.rgbTextures=pending;state.rgbTextureBytes=required;state.rgbManifest=manifest;state.rgbLevel=level;state.rgbReady=true;medicalWorkspace.dataset.rgbVolume="ready";medicalWorkspace.dataset.rgbLevel=String(level.level);if(!medicalWorkspace.dataset.firstRgbLevel)medicalWorkspace.dataset.firstRgbLevel=String(level.level);adopted=true;
    state.alignmentDirty=true;
    createBindGroups();state.renderPending=false;scheduleRender();await device.queue.onSubmittedWorkDone();for(const texture of old)texture.destroy();state.renderPending=false;scheduleRender();
    return true;
  }finally{
    if(!adopted)for(const texture of pending)texture.destroy();
  }
}

async function readTextureCenter(texture,dimensions) {
  const device=state.device,buffer=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  state.gpuPhase="volume verification";state.queueScopeActive=true;device.pushErrorScope("validation");
  let scopePopped=false;
  try{
    const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer(
      {texture,origin:dimensions.map((value)=>Math.floor(value/2))},
      {buffer,bytesPerRow:256,rowsPerImage:1},
      {width:1,height:1,depthOrArrayLayers:1},
    );device.queue.submit([encoder.finish()]);
    const validation=await device.popErrorScope();scopePopped=true;if(validation)throw new Error(`Volume verification failed: ${validation.message}`);
    await buffer.mapAsync(GPUMapMode.READ);return new DataView(buffer.getMappedRange()).getUint16(0,true);
  }finally{
    if(!scopePopped)await device.popErrorScope().catch(()=>null);
    if(buffer.mapState==="mapped")buffer.unmap();buffer.destroy();state.queueScopeActive=false;
    if(state.renderAgain){state.renderAgain=false;scheduleRender();}
  }
}

async function uploadLevel(subject,level,serial) {
  const device=state.device;
  const required=volumeBytes(level.dimensions),current=state.volumeTexture?state.textureBytes:0;
  if(required>MAX_VOLUME_BYTES)throw new Error(`Level ${level.level} needs ${displayBytes(required)}, above the 4 GiB voxel budget.`);
  if(required*2+current+state.rgbTextureBytes>TARGET_MEMORY_LIMIT)throw new Error(`Loading this level and Chrome's initialization buffer would exceed the ${displayBytes(TARGET_MEMORY_LIMIT)} working target.`);
  const pendingTexture=device.createTexture({label:`${subject} CT level ${level.level}`,size:level.dimensions,dimension:"3d",format:"r16uint",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});
  let completed=0,adopted=false;
  try{
  // Dawn may coalesce every queue.writeTexture staging allocation until the
  // next flush. Keep each batch well below the default 256 MiB buffer limit
  // instead of requesting a stack-sized temporary buffer from unified RAM.
  for(let start=0;start<level.bricks.length&&serial===state.loadingSerial;start+=6){
    const batch=(await Promise.all(level.bricks.slice(start,start+6).map((brick)=>fetchBrick(subject,level,brick,serial)))).filter(Boolean);
    if(!batch.length)break;
    state.stagingBytes=batch.reduce((total,item)=>total+item.bytes.byteLength,0);updatePerformance();
    state.gpuPhase="volume upload";state.queueScopeActive=true;device.pushErrorScope("validation");
    let staging=null,scopePopped=false;
    try{
      staging=device.createBuffer({size:state.stagingBytes,usage:GPUBufferUsage.COPY_SRC,mappedAtCreation:true});
      const mapped=new Uint8Array(staging.getMappedRange());let offset=0;for(const item of batch){mapped.set(item.bytes,offset);item.offset=offset;offset+=item.bytes.byteLength;}staging.unmap();
      const encoder=device.createCommandEncoder();
      for(const {brick,offset:brickOffset,bytesPerRow} of batch)encoder.copyBufferToTexture(
        {buffer:staging,offset:brickOffset,bytesPerRow,rowsPerImage:state.manifest.brick_size},
        {texture:pendingTexture,origin:[brick.x*state.manifest.brick_size,brick.y*state.manifest.brick_size,brick.z*state.manifest.brick_size]},
        {width:brick.extent[0],height:brick.extent[1],depthOrArrayLayers:brick.extent[2]},
      );
      device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();const validation=await device.popErrorScope();scopePopped=true;if(validation)throw new Error(`Volume upload validation failed: ${validation.message}`);
    }finally{
      if(!scopePopped)await device.popErrorScope().catch(()=>null);
      staging?.destroy();state.stagingBytes=0;state.queueScopeActive=false;
      if(state.renderAgain){state.renderAgain=false;scheduleRender();}
    }
    completed+=batch.length;
    setMessage(`Loading ${subject} CT · level ${level.level} · ${completed}/${level.bricks.length} bricks`);
  }
  if(serial!==state.loadingSerial||device!==state.device||!state.active)return false;
  const center=await readTextureCenter(pendingTexture,level.dimensions);
  if(serial!==state.loadingSerial||device!==state.device||!state.active)return false;
  medicalWorkspace.dataset.centerVoxel=String(center);
  const oldVolume=state.volumeTexture,oldOpacity=state.opacityTexture;
  const nextOpacity=makeOpacityTexture(level);
  state.volumeTexture=pendingTexture;state.opacityTexture=nextOpacity;state.level=level;state.textureBytes=required;adopted=true;
  createBindGroups();scheduleRender();await device.queue.onSubmittedWorkDone();oldVolume?.destroy();oldOpacity?.destroy();
  return true;
  }finally{if(!adopted)pendingTexture.destroy();}
}

async function loadVolume(subject=subjectSelect.value) {
  if(!state.active||!subject)return;
  if(state.detailSubject!==subject){state.requestedDetail=null;state.detailSubject=subject;}
  moreDetailButton.disabled=true;moreDetailButton.hidden=true;retryButton.hidden=true;retryButton.disabled=true;
  const serial=++state.loadingSerial;setMessage(`Opening ${subject} medical volume…`);buildButton.disabled=true;
  const previous=state.loadTail;let releaseLoad;state.loadTail=new Promise((resolve)=>{releaseLoad=resolve;});
  await previous.catch(()=>null);
  try{
    if(serial!==state.loadingSerial||!state.active)return;
    await initializeGpu();
    await state.device.queue.onSubmittedWorkDone();
    if(serial!==state.loadingSerial||!state.active)return;
    // A rejected/missing RGB manifest must not leave the prior subject or
    // alignment's textures attached to a newly loaded CT volume.
    releaseVolume({cancel:false});
    delete medicalWorkspace.dataset.firstRgbLevel;delete medicalWorkspace.dataset.firstRenderedRgbLevel;delete medicalWorkspace.dataset.renderedRgbLevel;delete medicalWorkspace.dataset.rgbLevel;delete medicalWorkspace.dataset.rgbRendered;
    const [response,initialRgbResponse]=await Promise.all([dataFetch(`/api/subjects/${encodeURIComponent(subject)}/volume/manifest`),dataFetch(`/api/subjects/${encodeURIComponent(subject)}/volume/rgb/manifest${state.liveAlignment?"?live=1":""}`)]);
    if(response.status===404){state.manifest=null;setMessage(`${subject} medical volume is not built.`);buildButton.disabled=false;return;}
    if(!response.ok){const failure=await response.json().catch(()=>null);throw new Error(failure?.error||`volume manifest HTTP ${response.status}`);}
    const manifest=await response.json();if(serial!==state.loadingSerial)return;
    let rgbResponse=initialRgbResponse,rgbManifest=rgbResponse.ok?await rgbResponse.json():null,savedAlignmentRemap=false;
    // Saving alignment knots changes registration hashes, but does not change
    // the source voxels. Normal viewing can use the verified bake plus the
    // saved correction curve, with no editor draft or selected-slice overlay.
    if(!state.liveAlignment&&rgbResponse.status===409){
      const liveResponse=await dataFetch(`/api/subjects/${encodeURIComponent(subject)}/volume/rgb/manifest?live=1`);
      if(liveResponse.ok){
        const liveManifest=await liveResponse.json();
        if(liveManifest.live_alignment?.available===true){rgbManifest=liveManifest;rgbResponse=liveResponse;savedAlignmentRemap=true;}
      }else rgbResponse=liveResponse;
    }
    if(serial!==state.loadingSerial)return;
    state.savedAlignmentRemap=savedAlignmentRemap;
    const maxDimension=state.device.limits.maxTextureDimension3D;
    // Editing stays at the half-resolution level; the selected photograph is
    // loaded separately. This avoids native-stack uploads during interaction.
    const preview=window.visibleHumanConfig?.automaticVolume!=="native";
    const minimum=state.liveAlignment?1:state.requestedDetail??(preview?Math.max(...manifest.levels.map(item=>item.level)):0);
    const editingLevels=manifest.levels.filter((item)=>item.level>=minimum);
    const target=chooseVolumeLevel(editingLevels.length?editingLevels:manifest.levels,MAX_VOLUME_BYTES,maxDimension);
    if(!target)throw new Error(`No volume level fits Chrome's ${maxDimension}³ texture limit and 4 GiB voxel budget.`);
    state.manifest=manifest;state.alignmentDirty=true;
    if(state.savedAlignmentRemap&&window.visibleHumanAlignment)acceptAlignmentSnapshot(window.visibleHumanAlignment.snapshot());
    drawTransferEditor();
    const path=[...manifest.levels].filter((level)=>level.level>=target.level).sort((a,b)=>b.level-a.level);
    let rgbLoaded=false;
    for(const level of path){
      if(!await uploadLevel(subject,level,serial))return;
      const rgbLevel=rgbManifest?.levels?.find((item)=>item.dimensions.every((value,index)=>value===level.dimensions[index]));
      if(rgbLevel){rgbLoaded=await uploadRgbLevel(subject,rgbManifest,rgbLevel,serial)||rgbLoaded;if(serial!==state.loadingSerial)return;}
      medicalWorkspace.dataset.progressiveLevel=String(level.level);
      setMessage(`${subject} · ${level.dimensions.join("×")} · ${rgbLevel?"CT + registered RGB":"CT preview"}${level.level===target.level?"":" · refining…"}`);
      state.renderPending=false;drawFrame();await state.device.queue.onSubmittedWorkDone();
    }
    const rgbFailure=!rgbResponse.ok?await rgbResponse.json().catch(()=>null):null;
    setMessage(`${subject} · ${state.level.dimensions.join("×")} · ${rgbLoaded?(state.liveAlignment?"Live CT + RGB":"CT + registered RGB"):"CT only"}${rgbFailure?.error?` · ${rgbFailure.error}`:""}`);
    buildButton.textContent="Rebuild volume";buildButton.disabled=false;scheduleRender();
    const next=manifest.levels.filter(item=>item.level<target.level).sort((a,b)=>b.level-a.level)[0];
    state.nextDetail=next?.level??null;
    if(next&&!state.liveAlignment){
      const rgbNext=rgbManifest?.levels?.find(item=>item.level===next.level);
      const raw=item=>(item?.bricks||[]).reduce((sum,b)=>sum+(b.bytes||0),0);
      const bytes=(next.transfer_bytes||raw(next))+(rgbNext?.transfer_bytes||raw(rgbNext));
      const exact=next.transfer_size_exact&&rgbNext?.transfer_size_exact;
      moreDetailButton.textContent=`More detail · ${exact?"":"up to "}${displayBytes(bytes)}`;
      moreDetailButton.hidden=false;moreDetailButton.disabled=false;
    }
  }catch(error){if(serial===state.loadingSerial){setMessage(`${error.message} · Retry when your connection is ready.`);buildButton.disabled=false;retryButton.hidden=false;retryButton.disabled=false;}}
  finally{releaseLoad();}
}

function cameraBasis() {
  const cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),sy=Math.sin(state.yaw),cy=Math.cos(state.yaw);
  const forward=[cp*cy,cp*sy,sp],right=[-sy,cy,0],up=[-sp*cy,-sp*sy,cp];return{forward,right,up};
}

const knotOutline=document.createElement("canvas");
knotOutline.id="medical-knot-outline";
knotOutline.style.cssText="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2";
canvases.volume.parentElement.append(knotOutline);
const liveAlignmentStatus=document.createElement("output");
liveAlignmentStatus.id="live-alignment-status";liveAlignmentStatus.className="live-only";
document.querySelector("#selected-knot-label").after(liveAlignmentStatus);

function acceptAlignmentSnapshot(snapshot) {
  state.alignmentSnapshot=snapshot;state.alignmentDirty=true;state.alignmentRevision++;
  medicalWorkspace.dataset.alignmentRevision=String(state.alignmentRevision);
  if(state.liveAlignment){loadKnotTexture();scheduleRender(true);}
  else if(state.savedAlignmentRemap&&state.active)scheduleRender();
}

async function loadKnotTexture() {
  const snapshot=state.alignmentSnapshot,frame=snapshot?.draft?.color_frame;
  if(!state.liveAlignment||!state.device||!Number.isFinite(frame)||snapshot.subject!==subjectSelect.value)return;
  const key=`${snapshot.subject}:${frame}`;
  if(key===state.knotRequestedKey)return;
  state.knotRequest?.abort();const controller=new AbortController();state.knotRequest=controller;state.knotRequestedKey=key;
  medicalWorkspace.dataset.knotTexture="loading";delete medicalWorkspace.dataset.knotTextureFrame;
  const device=state.device;
  try{
    const response=await dataFetch(`/api/subjects/${encodeURIComponent(snapshot.subject)}/rgb/layers/${frame}`,{signal:controller.signal});
    if(!response.ok)throw new Error(`RGB slice ${frame}: HTTP ${response.status}`);
    const bitmap=await createImageBitmap(await response.blob(),{resizeWidth:1024,resizeHeight:608,resizeQuality:"high"});
    if(controller.signal.aborted||device!==state.device||state.knotRequestedKey!==key){bitmap.close();return;}
    const texture=device.createTexture({label:`Selected RGB ${frame}`,size:[bitmap.width,bitmap.height],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});
    device.queue.copyExternalImageToTexture({source:bitmap},{texture},{width:bitmap.width,height:bitmap.height});
    state.knotTextureBytes=bitmap.width*bitmap.height*4;bitmap.close();
    const old=state.knotTexture;state.knotTexture=texture;state.knotTextureKey=key;createBindGroups();
    medicalWorkspace.dataset.knotTexture="ready";medicalWorkspace.dataset.knotTextureFrame=String(frame);
    scheduleRender(true);await device.queue.onSubmittedWorkDone();old?.destroy();
  }catch(error){if(controller.signal.aborted)return;state.knotRequestedKey=null;medicalWorkspace.dataset.knotTexture="failed";setMessage(error.message);}
}

function updateLiveAlignment() {
  if((!state.liveAlignment&&!state.savedAlignmentRemap)||!state.alignmentDirty||!state.manifest)return;
  const snapshot=state.alignmentSnapshot;
  if(!snapshot||snapshot.subject!==subjectSelect.value)return;
  const table=buildVolumeAlignmentLut({volumeManifest:state.manifest,rgbManifest:state.rgbManifest||{},knots:snapshot.knots||[],draft:state.liveAlignment?snapshot.draft:null,profile:snapshot.profile||{},sourceManifest:snapshot.manifest||{},orientation:"flip_y"});
  state.alignmentTable=table;state.alignmentDirty=false;
  const bytes=Math.max(32,table.data.byteLength);
  if(state.alignmentBufferBytes!==bytes){
    const old=state.alignmentBuffer;
    state.alignmentBuffer=state.device.createBuffer({label:"Live RGB alignment",size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});state.alignmentBufferBytes=bytes;
    createBindGroups();old?.destroy();
  }
  state.device.queue.writeBuffer(state.alignmentBuffer,0,table.data.byteLength?table.data:new Float32Array(8));
  medicalWorkspace.dataset.liveAlignment=table.enabled?"ready":table.reason||"unavailable";
  medicalWorkspace.dataset.savedAlignment=state.savedAlignmentRemap?(table.enabled?"ready":table.reason||"unavailable"):"off";
  medicalWorkspace.dataset.liveValidSlices=String(table.validCount||0);
  medicalWorkspace.dataset.selectedKnot=String(table.selected?.colorFrame??"");
  medicalWorkspace.dataset.selectedCt=String(table.selected?.ctFrame??"");
  medicalWorkspace.dataset.selectedZ=String(table.selected?.z??"");
  liveAlignmentStatus.textContent=!state.liveAlignment?"":!table.selected?.valid?"Preview only · Z ordering or coverage invalid":!table.enabled?"RGB volume remapping unavailable":table.blockedCorrectionCount?"Preview limited at an acquisition boundary or long knot interval":"";
  if(table.selected?.index!==null&&Number.isFinite(table.selected?.z)&&state.level){
    const key=`${snapshot.subject}:${table.selected.colorFrame}`;
    if(state.focusedKnotKey!==key){
      const z=(table.selected.z-.5)*2*sceneGeometry().half[2],basis=cameraBasis();
      state.cameraPan=[z*basis.right[2],z*basis.up[2]];state.focusedKnotKey=key;syncInteractionState();
    }
  }
  loadKnotTexture();
}

function drawKnotOutline() {
  const ratio=devicePixelRatio||1,width=canvases.volume.clientWidth,height=canvases.volume.clientHeight;
  if(knotOutline.width!==Math.round(width*ratio)||knotOutline.height!==Math.round(height*ratio)){knotOutline.width=Math.round(width*ratio);knotOutline.height=Math.round(height*ratio);}
  const context=knotOutline.getContext("2d");context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,width,height);
  const selected=state.alignmentTable?.selected;
  if(!state.liveAlignment||!Number.isFinite(selected?.z)||!selected?.inverseUv||!state.level)return;
  const {half}=sceneGeometry(),basis=cameraBasis(),z=(selected.z-.5)*2*half[2];
  const project=(point)=>{const dot=(axis)=>point.reduce((sum,item,index)=>sum+item*axis[index],0);return [(1+(dot(basis.right)-state.cameraPan[0])*state.zoom/(width/height))*width/2,(1+(dot(basis.up)-state.cameraPan[1])*state.zoom)*height/2];};
  const corners=[[-half[0],-half[1],z],[half[0],-half[1],z],[half[0],half[1],z],[-half[0],half[1],z]].map(project);
  context.beginPath();corners.forEach(([x,y],index)=>index?context.lineTo(x,y):context.moveTo(x,y));context.closePath();
  const colour=selected.valid?"#b6f46b":"#ffb454";
  context.strokeStyle=colour;context.lineWidth=2;context.shadowColor="#000";context.shadowBlur=3;context.stroke();
  const [x,y]=corners.reduce((best,item)=>item[0]<best[0]?item:best,corners[0]);
  context.font="12px system-ui";context.fillStyle=colour;context.fillText(`RGB ${selected.colorFrame} · CT ${selected.ctFrame.toFixed(2)}`,Math.max(8,x),Math.max(18,y-8));
}

function rotateAroundAxis(vector,axis,angle) {
  const cosine=Math.cos(angle),sine=Math.sin(angle),dot=vector[0]*axis[0]+vector[1]*axis[1]+vector[2]*axis[2];
  const cross=[axis[1]*vector[2]-axis[2]*vector[1],axis[2]*vector[0]-axis[0]*vector[2],axis[0]*vector[1]-axis[1]*vector[0]];
  const rotated=vector.map((value,index)=>value*cosine+cross[index]*sine+axis[index]*dot*(1-cosine)),length=Math.hypot(...rotated)||1;
  return rotated.map((value)=>value/length);
}

function updateCameraMotion(now) {
  const elapsed=state.cameraMotionTime?Math.min((now-state.cameraMotionTime)/1000,.05):0;
  state.cameraMotionTime=now;
  if(!elapsed)return state.cameraKeys.size>0;
  const axis=(positive,negative)=>(state.cameraKeys.has(positive)?1:0)-(state.cameraKeys.has(negative)?1:0);
  const target=[axis("d","a"),axis("q","e"),axis("w","s")],response=1-Math.exp(-elapsed*14);
  for(let index=0;index<3;index++)state.cameraVelocity[index]+=(target[index]-state.cameraVelocity[index])*response;
  const [x,y,z]=state.cameraVelocity,panSpeed=.42/Math.max(state.zoom,.45),zoomSpeed=1.05;
  state.cameraPan[0]+=x*panSpeed*elapsed;state.cameraPan[1]+=y*panSpeed*elapsed;
  state.zoom=clamp(state.zoom*Math.exp(z*zoomSpeed*elapsed),.3,8);
  const moving=state.cameraKeys.size>0||state.cameraVelocity.some((value)=>Math.abs(value)>.003);
  if(!moving)state.cameraVelocity.fill(0);
  if(x||y||z)syncInteractionState();
  return moving;
}

function sceneGeometry() {
  const dimensions=state.level.dimensions,baseSpacing=state.manifest.spacing_mm.map((value)=>value*state.level.factor),physical=dimensions.map((value,index)=>value*baseSpacing[index]),maximum=Math.max(...physical);
  const half=physical.map((value)=>value/maximum*.5),normal=state.planeNormal;
  const support=Math.abs(normal[0])*half[0]+Math.abs(normal[1])*half[1]+Math.abs(normal[2])*half[2];
  return{dimensions,baseSpacing,half,normal,support,offset:state.planeOffset*support};
}

function resizeCanvas(canvas,qualityScale=1) {
  const ratio=(devicePixelRatio||1)*qualityScale,width=Math.max(1,Math.round(canvas.clientWidth*ratio)),height=Math.max(1,Math.round(canvas.clientHeight*ratio));
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}return width/height;
}

function uniformValues(name,aspect,interactive) {
  const {dimensions,baseSpacing,half,normal,offset}=sceneGeometry(),maximum=Math.max(...dimensions.map((value,index)=>value*baseSpacing[index]));
  const basis=cameraBasis(),scale=[aspect,1],coarse=interactive,stepVoxels=coarse?3.2:1.15,step=stepVoxels*Math.min(...baseSpacing)/maximum;
  const values=new Float32Array(60);
  values.set([...dimensions,0],0);values.set([...half,state.zoom],4);values.set([...basis.right,0],8);values.set([...basis.up,0],12);values.set([...basis.forward,0],16);
  values.set([0,0,state.customDensity+1024,presetSelect.value==="custom"?1:0],20);values.set([...scale,...state.window],24);values.set([step,2048,coarse?0:1,state.mip?1:0],28);values.set([...state.level.grid,0],32);
  values.set([...normal,offset],36);values.set([state.liveAlignment?0:1,state.cameraPan[0],state.rgbReady?1:0,state.cameraPan[1]],40);
  const table=state.alignmentTable,selected=table?.selected,valid=state.liveAlignment&&Number.isFinite(selected?.z)&&selected?.inverseUv;
  values.set([Math.max(1,table?.depth||1),state.liveAlignment||state.savedAlignmentRemap?1:0,0,0],44);
  values.set([valid?(selected.z-.5)*2*half[2]:0,valid?1:0,state.knotTextureKey===`${state.alignmentSnapshot?.subject}:${selected?.colorFrame}`?1:0,Number(knotBlend.value)/100],48);
  const inverse=selected?.inverseUv||[1,0,0,0,1,0];values.set([...inverse.slice(0,3),0],52);values.set([...inverse.slice(3,6),0],56);
  return values;
}

function drawFrame() {
  state.renderPending=false;if(!state.active||!state.device||!state.volumeTexture||state.gpuFault)return;
  if(state.queueScopeActive||state.frameInFlight){state.renderAgain=true;return;}
  const device=state.device;state.gpuPhase="render";
  const now=performance.now(),cameraMoving=updateCameraMotion(now),interactive=cameraMoving||now<state.interactingUntil;
  const revision=state.alignmentRevision;
  try{
    updateLiveAlignment();
    const encoder=device.createCommandEncoder({label:"CT frame"});
    for(const [name,canvas] of Object.entries(canvases)){
      const renderScale=name==="volume"&&interactive ? .55 : state.liveAlignment ? .8 : 1,aspect=resizeCanvas(canvas,renderScale);
      device.queue.writeBuffer(state.uniforms.get(name),0,uniformValues(name,aspect,interactive));
      const pass=encoder.beginRenderPass({label:`${name} pass`,colorAttachments:[{view:state.contexts.get(name).getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});
      pass.setPipeline(state.pipeline);pass.setBindGroup(0,state.bindGroups.get(name));pass.draw(3);pass.end();
    }
    device.queue.submit([encoder.finish()]);state.frameInFlight=true;
    const rgbLevel=state.rgbLevel?.level,rgbReady=state.rgbReady;
    device.queue.onSubmittedWorkDone().then(()=>{
      if(device!==state.device)return;state.frameInFlight=false;
      medicalWorkspace.dataset.rgbRendered=rgbReady?"true":"false";
      if(rgbReady&&rgbLevel!==undefined){medicalWorkspace.dataset.renderedRgbLevel=String(rgbLevel);if(!medicalWorkspace.dataset.firstRenderedRgbLevel)medicalWorkspace.dataset.firstRenderedRgbLevel=String(rgbLevel);}
      medicalWorkspace.dataset.renderedAlignmentRevision=String(revision);drawKnotOutline();
      const elapsed=Math.max(1,performance.now()-now),instant=1000/elapsed;state.fps=state.fps?state.fps*.82+instant*.18:instant;updatePerformance();
      if(state.renderAgain||interactive){state.renderAgain=false;scheduleRender(cameraMoving);}
    }).catch((error)=>{state.frameInFlight=false;reportGpuFault(`render completion failed: ${error.message}`,device);});
  }catch(error){reportGpuFault(`render failed: ${error.message}`,device);return;}
}

function scheduleRender(interacting=false) {
  if(interacting)state.interactingUntil=performance.now()+180;
  if(state.queueScopeActive||state.frameInFlight){state.renderAgain=true;return;}
  if(!state.renderPending){state.renderPending=true;requestAnimationFrame(drawFrame);}
}

function updatePerformance() {
  const total=state.textureBytes+state.rgbTextureBytes+state.stagingBytes+state.knotTextureBytes+state.alignmentBufferBytes+(state.transferTexture?4096*4:0);
  performanceOutput.textContent=`${state.fps.toFixed(1)} FPS · ${displayBytes(total)} · ${state.level?`L${state.level.level}`:"—"}`;
  performanceOutput.title=`Renderer-owned allocations; hard limit ${displayBytes(HARD_MEMORY_LIMIT)}`;
}

async function buildVolume() {
  const subject=subjectSelect.value;if(!subject)return;buildButton.disabled=true;setMessage(`Building ${subject} volume…`);
  try{
    const response=await dataFetch(`/api/subjects/${encodeURIComponent(subject)}/volume/build`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({memory_limit_gib:8})});const job=await response.json();if(!response.ok)throw new Error(job.error||`HTTP ${response.status}`);
    while(true){await new Promise((resolve)=>setTimeout(resolve,1000));const current=await(await dataFetch(`/api/review/jobs/${job.id}`)).json();setMessage(`Building ${subject} volume · ${Math.round((current.progress||0)*100)}%`);if(current.state==="failed")throw new Error(current.error||current.log?.at(-1)||"Volume build failed");if(current.state==="complete")break;}
    await loadVolume(subject);
  }catch(error){setMessage(error.message);buildButton.disabled=false;}
}

function releaseVolume({cancel=true}={}) {
  if(cancel)state.loadingSerial++;state.volumeTexture?.destroy();state.opacityTexture?.destroy();for(const texture of state.rgbTextures)texture.destroy();
  state.volumeTexture=null;state.opacityTexture=null;state.rgbTextures=state.device?emptyRgbTextures(state.device):[];state.rgbReady=false;delete medicalWorkspace.dataset.rgbVolume;delete medicalWorkspace.dataset.rgbRendered;delete medicalWorkspace.dataset.rgbLevel;delete medicalWorkspace.dataset.firstRgbLevel;delete medicalWorkspace.dataset.renderedRgbLevel;delete medicalWorkspace.dataset.firstRenderedRgbLevel;delete medicalWorkspace.dataset.progressiveLevel;state.rgbManifest=null;state.rgbLevel=null;state.level=null;state.textureBytes=0;state.rgbTextureBytes=0;state.stagingBytes=0;state.bindGroups.clear();updatePerformance();
  state.alignmentTable=null;state.alignmentDirty=true;
  state.savedAlignmentRemap=false;delete medicalWorkspace.dataset.savedAlignment;
}

async function switchTab(name) {
  if(window.visibleHumanConfig?.readOnly && name!=="medical")return;
  const wasLive=state.liveAlignment;
  const medical=name==="medical"||name==="align3d";state.active=medical;state.liveAlignment=name==="align3d";state.alignmentDirty=true;document.body.dataset.activeTab=name;
  state.focusedKnotKey=null;
  clearTimeout(viewSaveTimer);
  knotEditorHost.hidden=!state.liveAlignment;
  if(state.liveAlignment)knotEditorHost.append(knotEditor);else knotEditorHome.insertBefore(knotEditor,knotEditorNext);
  alignmentWorkspace.hidden=medical;medicalWorkspace.hidden=!medical;alignmentControls.hidden=medical;medicalControls.hidden=!medical;
  for(const [tab,value] of [[alignmentTab,"alignment"],[medicalTab,"medical"],[align3dTab,"align3d"]]){tab.classList.toggle("active",name===value);tab.setAttribute("aria-selected",String(name===value));}
  history.replaceState(null,"",`${location.pathname}${location.search}#${name}`);
  if(medical){
    if(wasLive&&!state.liveAlignment){
      const preset=await restoreMedicalView();
      if(document.body.dataset.activeTab!==name)return;
      presetSelect.value=preset;applyPreset(preset);syncInteractionState();
    }
    window.dispatchEvent(new Event(state.liveAlignment?"alignment3d-tab-activated":"medical-tab-activated"));
    if(state.liveAlignment&&window.visibleHumanAlignment)acceptAlignmentSnapshot(window.visibleHumanAlignment.snapshot());
    loadVolume();
  }else{releaseVolume();drawKnotOutline();window.dispatchEvent(new Event("alignment-tab-activated"));}
}

alignmentTab.addEventListener("click",()=>switchTab("alignment"));medicalTab.addEventListener("click",()=>switchTab("medical"));
align3dTab.addEventListener("click",()=>switchTab("align3d"));
moreDetailButton.addEventListener("click",()=>{
  if(state.nextDetail===null)return;
  state.requestedDetail=state.nextDetail;loadVolume();
});
retryButton.addEventListener("click",()=>loadVolume());
window.addEventListener("alignment-preview-changed",(event)=>acceptAlignmentSnapshot(event.detail));
knotBlend.addEventListener("input",()=>scheduleRender(true));
subjectSelect.addEventListener("change",async()=>{if(state.active){const preset=await restoreMedicalView();presetSelect.value=preset;applyPreset(preset);syncInteractionState();}});
window.addEventListener("subject-selected",(event)=>{if(state.active)loadVolume(event.detail?.subject||subjectSelect.value);});
presetSelect.addEventListener("change",()=>{applyPreset(presetSelect.value);saveMedicalView();});
function syncInteractionState(){
  const [azimuth,inclination]=planeAnglesFromNormal(state.planeNormal);
  planeAzimuthInput.value=azimuth.toFixed(1);planeInclinationInput.value=inclination.toFixed(1);planeDepthInput.value=state.planeOffset.toFixed(3);
  planeAzimuthValue.textContent=`${azimuth.toFixed(1)}°`;planeInclinationValue.textContent=`${inclination.toFixed(1)}°`;planeDepthValue.textContent=state.planeOffset.toFixed(3);
  medicalWorkspace.dataset.planeOffset=state.planeOffset.toFixed(6);medicalWorkspace.dataset.planeRotateX=state.planeRotateX.toFixed(6);medicalWorkspace.dataset.planeRotateY=state.planeRotateY.toFixed(6);medicalWorkspace.dataset.planeNormal=state.planeNormal.map((value)=>value.toFixed(6)).join(",");medicalWorkspace.dataset.cameraPan=state.cameraPan.map((value)=>value.toFixed(6)).join(",");medicalWorkspace.dataset.cameraZoom=state.zoom.toFixed(6);medicalWorkspace.dataset.cameraYaw=state.yaw.toFixed(6);medicalWorkspace.dataset.cameraPitch=state.pitch.toFixed(6);
}
for(const control of [planeAzimuthInput,planeInclinationInput])control.addEventListener("input",()=>{state.planeNormal=planeNormalFromAngles(Number(planeAzimuthInput.value),Number(planeInclinationInput.value));syncInteractionState();saveMedicalView();scheduleRender(true);});
planeDepthInput.addEventListener("input",()=>{state.planeOffset=clamp(Number(planeDepthInput.value),-1,1);syncInteractionState();saveMedicalView();scheduleRender(true);});
function toggleMedicalGuide(show){medicalControlsGuide.hidden=!show;medicalInfoButton.setAttribute("aria-expanded",String(show));medicalInfoButton.setAttribute("aria-label",show?"Hide viewer controls":"Show viewer controls");}
medicalInfoButton.addEventListener("click",()=>toggleMedicalGuide(medicalControlsGuide.hidden));
document.addEventListener("pointerdown",(event)=>{if(!medicalControlsGuide.hidden&&!medicalControlsGuide.contains(event.target)&&event.target!==medicalInfoButton)toggleMedicalGuide(false);});
document.addEventListener("keydown",(event)=>{if(event.key==="Escape"&&!medicalControlsGuide.hidden)toggleMedicalGuide(false);});
customDensityInput.addEventListener("input",()=>{state.customDensity=clamp(Number(customDensityInput.value),CUSTOM_DENSITY_MIN,CUSTOM_DENSITY_MAX);saveCustom();uploadTransfer();});
buildButton.addEventListener("click",buildVolume);
document.querySelector("#transfer-reset").addEventListener("click",()=>applyPreset(presetSelect.value==="custom"?"bone":presetSelect.value,false));
document.querySelector("#transfer-add-color").addEventListener("click",()=>{state.colorPoints.push({hu:400,color:[1,.75,.55]});saveCustom();refreshColorRows();uploadTransfer();});
windowCenterInput.addEventListener("input",()=>{state.window[0]=Number(windowCenterInput.value);saveCustom();scheduleRender();});windowWidthInput.addEventListener("input",()=>{state.window[1]=Math.max(1,Number(windowWidthInput.value));saveCustom();scheduleRender();});

transferCanvas.addEventListener("pointerdown",(event)=>{const value=transferPoint(event),rect=transferCanvas.getBoundingClientRect();let best=-1,distance=Infinity;state.opacityPoints.forEach((point,index)=>{const x=(point.hu+1024)/4095,y=1-point.opacity,d=Math.hypot((x-value.x)*rect.width,(y-value.y)*rect.height);if(d<distance){best=index;distance=d;}});if(distance>13){state.opacityPoints.push({hu:value.hu,opacity:value.opacity});best=state.opacityPoints.length-1;}state.selectedOpacity=best;state.transferDragging=true;transferCanvas.setPointerCapture(event.pointerId);transferSelection.textContent=`${Math.round(state.opacityPoints[best].hu)} HU`;drawTransferEditor();});
transferCanvas.addEventListener("pointermove",(event)=>{if(!state.transferDragging||state.selectedOpacity<0)return;const value=transferPoint(event),point=state.opacityPoints[state.selectedOpacity];point.hu=value.hu;point.opacity=value.opacity;transferSelection.textContent=`${Math.round(point.hu)} HU · ${point.opacity.toFixed(2)}`;uploadTransfer();});
transferCanvas.addEventListener("pointerup",(event)=>{state.transferDragging=false;transferCanvas.releasePointerCapture(event.pointerId);state.opacityPoints.sort((a,b)=>a.hu-b.hu);state.selectedOpacity=-1;saveCustom();drawTransferEditor();});
transferCanvas.addEventListener("dblclick",(event)=>{if(state.opacityPoints.length<=2)return;const value=transferPoint(event),rect=transferCanvas.getBoundingClientRect();let best=-1,distance=Infinity;state.opacityPoints.forEach((point,index)=>{const d=Math.hypot(((point.hu+1024)/4095-value.x)*rect.width,(1-point.opacity-value.y)*rect.height);if(d<distance){best=index;distance=d;}});if(distance<15){state.opacityPoints.splice(best,1);saveCustom();uploadTransfer();}});

let drag=null,pinch=null;
const touchPointers=new Map();
function makeDrag(event){return{id:event.pointerId,x:event.clientX,y:event.clientY,yaw:state.yaw,pitch:state.pitch,pan:[...state.cameraPan],planeX:state.planeRotateX,planeY:state.planeRotateY,planeNormal:[...state.planeNormal],camera:cameraBasis(),planeOffset:state.planeOffset,mode:event.metaKey?"position":event.ctrlKey?"plane":event.shiftKey?"pan":"camera"};}
function rebaseTouches(){
  const touches=[...touchPointers.values()];drag=null;pinch=null;
  if(touches.length===1){drag=makeDrag({...touches[0],metaKey:false,ctrlKey:false,shiftKey:false});return;}
  if(touches.length<2)return;
  const [first,second]=touches;
  pinch={x:(first.clientX+second.clientX)/2,y:(first.clientY+second.clientY)/2,distance:Math.max(4,Math.hypot(first.clientX-second.clientX,first.clientY-second.clientY)),zoom:state.zoom,pan:[...state.cameraPan]};
}
function moveDrag(event){
  if(!drag||drag.id!==event.pointerId)return;
  const deltaX=event.clientX-drag.x,deltaY=event.clientY-drag.y;
  if(drag.mode==="position")state.planeOffset=clamp(drag.planeOffset+deltaY*.003,-1,1);
  else if(drag.mode==="plane"){const radians=Math.PI/180,aroundUp=rotateAroundAxis(drag.planeNormal,drag.camera.up,deltaX*.35*radians);state.planeNormal=rotateAroundAxis(aroundUp,drag.camera.right,deltaY*.35*radians);state.planeRotateY=drag.planeY+deltaX*.35;state.planeRotateX=clamp(drag.planeX+deltaY*.35,-180,180);}
  else if(drag.mode==="pan"){const speed=.002/Math.max(state.zoom,.3);state.cameraPan[0]=drag.pan[0]-deltaX*speed;state.cameraPan[1]=drag.pan[1]-deltaY*speed;}
  else{state.yaw=drag.yaw+deltaX*.008;state.pitch=clamp(drag.pitch+deltaY*.008,-1.45,1.45);}
  syncInteractionState();scheduleRender(true);
}
function movePinch(){
  if(!pinch)return;
  const [first,second]=[...touchPointers.values()];if(!second)return;
  const x=(first.clientX+second.clientX)/2,y=(first.clientY+second.clientY)/2;
  const distance=Math.max(4,Math.hypot(first.clientX-second.clientX,first.clientY-second.clientY));
  const speed=.002/Math.max(pinch.zoom,.3);
  state.cameraPan[0]=pinch.pan[0]-(x-pinch.x)*speed;state.cameraPan[1]=pinch.pan[1]-(y-pinch.y)*speed;
  state.zoom=clamp(pinch.zoom*distance/pinch.distance,.3,8);
  syncInteractionState();scheduleRender(true);
}
canvases.volume.addEventListener("pointerdown",(event)=>{
  if(event.pointerType==="touch"){touchPointers.set(event.pointerId,{pointerId:event.pointerId,clientX:event.clientX,clientY:event.clientY});rebaseTouches();}
  else if(!touchPointers.size)drag=makeDrag(event);
  try{canvases.volume.setPointerCapture(event.pointerId);}catch{}
});
canvases.volume.addEventListener("pointermove",(event)=>{
  if(event.pointerType==="touch"){
    if(!touchPointers.has(event.pointerId))return;
    touchPointers.set(event.pointerId,{pointerId:event.pointerId,clientX:event.clientX,clientY:event.clientY});
    if(touchPointers.size>=2)movePinch();else moveDrag(event);
  }else if(!touchPointers.size)moveDrag(event);
});
function finishVolumePointer(event){
  if(event.pointerType==="touch"){
    if(!touchPointers.delete(event.pointerId))return;
    rebaseTouches();if(!touchPointers.size){saveMedicalView();scheduleRender();}
  }else if(drag?.id===event.pointerId){drag=null;saveMedicalView();scheduleRender();}
  try{if(canvases.volume.hasPointerCapture(event.pointerId))canvases.volume.releasePointerCapture(event.pointerId);}catch{}
}
canvases.volume.addEventListener("pointerup",finishVolumePointer);
canvases.volume.addEventListener("pointercancel",finishVolumePointer);
canvases.volume.addEventListener("wheel",(event)=>{event.preventDefault();state.zoom=clamp(state.zoom*Math.exp(-event.deltaY*.001),.3,8);syncInteractionState();saveMedicalView();scheduleRender(true);},{passive:false});
canvases.volume.addEventListener("contextmenu",(event)=>event.preventDefault());
window.addEventListener("keydown",(event)=>{if(!state.active||event.metaKey||event.ctrlKey||event.altKey||event.target?.matches?.("input, select, button, textarea"))return;const key=event.key.toLowerCase();if(!"wasdqe".includes(key))return;event.preventDefault();state.cameraKeys.add(key);state.cameraMotionTime=performance.now();scheduleRender(true);});
window.addEventListener("keyup",(event)=>{const key=event.key.toLowerCase();if(!"wasdqe".includes(key))return;state.cameraKeys.delete(key);state.cameraMotionTime=performance.now();saveMedicalView();scheduleRender(true);});
window.addEventListener("blur",()=>{state.cameraKeys.clear();state.cameraMotionTime=performance.now();saveMedicalView();scheduleRender(true);});
window.addEventListener("resize",()=>{drawTransferEditor();scheduleRender();});

async function startMedicalRenderer(){const preset=await restoreMedicalView();presetSelect.value=preset;syncInteractionState();applyPreset(preset);if(["#medical","#align3d"].includes(location.hash))switchTab(location.hash.slice(1));}
startMedicalRenderer();
