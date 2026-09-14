import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  GIB,
  brickOpacity,
  canvasPlaneScale,
  chooseVolumeLevel,
  transferLut,
  volumeBytes,
} from "../volume-math.mjs";

test("16-bit volume sizing and level selection", () => {
  assert.equal(volumeBytes([512,512,16384]),8*GIB);
  assert.equal(volumeBytes([512,512,1877]),984088576);
  const levels=[{level:0,dimensions:[4096,512,512]},{level:1,dimensions:[2048,256,256]},{level:2,dimensions:[1024,128,128]}];
  assert.equal(chooseVolumeLevel(levels,4*GIB,2048).level,1);
});

test("transfer LUT interpolates HU opacity and colour", () => {
  const lut=transferLut(
    [{hu:-1024,opacity:0},{hu:0,opacity:.5},{hu:3071,opacity:1}],
    [{hu:-1024,color:[0,0,0]},{hu:3071,color:[1,.5,.25]}],
  );
  assert.equal(lut.length,4096*4);
  assert.ok(lut[1024*4+3]>=127&&lut[1024*4+3]<=128);
  assert.deepEqual([...lut.slice(4095*4,4095*4+4)],[255,128,64,255]);
  assert.deepEqual([...brickOpacity([{min_hu:-1024,max_hu:-1024},{min_hu:0,max_hu:100}],lut)],[0,Math.max(...Array.from({length:101},(_,i)=>lut[(1024+i)*4+3]))]);
});

test("MPR plane scale preserves physical aspect", () => {
  assert.deepEqual(canvasPlaneScale(2,[512,512,1000],[1,1,1],1),[1,1]);
  assert.deepEqual(canvasPlaneScale(1,[512,512,1024],[1,1,1],1),[.5,1]);
});

test("WGSL avoids non-portable writes to vector components", async () => {
  const source=await readFile(new URL("../medical-renderer.mjs",import.meta.url),"utf8");
  const shader=source.match(/const SHADER = \/\* wgsl \*\/`([\s\S]*?)`;\n/)[1];
  assert.doesNotMatch(shader,/\.(?:[rgba]{1,4}|[xyzw]{1,4})\s*(?:[+*/-]=|=(?!=))/);
  assert.match(source,/getCompilationInfo/);
  assert.match(source,/createRenderPipelineAsync/);
});

test("custom density rendering returns opaque registered RGB at the first ray crossing", async () => {
  const source=await readFile(new URL("../medical-renderer.mjs",import.meta.url),"utf8");
  const shader=source.match(/const SHADER = \/\* wgsl \*\/`([\s\S]*?)`;\n/)[1];
  assert.match(shader,/let crossed=.*before<0\.0.*after>=0\.0/);
  assert.match(shader,/let rgb=rgbAtWorld\(hitWorld\);\s*if \(rgb\.a>\.5\) \{ return vec4f\(rgb\.rgb,1\); \}/);
  assert.doesNotMatch(source,/#plane-(?:scale|rgb)/);
  assert.match(source,/mode:event\.metaKey\?"position":event\.ctrlKey\?"plane":event\.shiftKey\?"pan":"camera"/);
  assert.match(source,/axis\("q","e"\)/);
  assert.match(source,/state\.planeRotateX=clamp\(drag\.planeX\+deltaY\*\.35/);
  assert.match(source,/rotateAroundAxis\(drag\.planeNormal,drag\.camera\.up/);
  assert.match(source,/rotateAroundAxis\(aroundUp,drag\.camera\.right/);
  assert.doesNotMatch(source,/Custom density.*first RGB hit/);
  assert.match(source,/function currentMedicalView\(\)/);
  assert.match(source,/function restoreMedicalView\(/);
  assert.match(source,/function saveMedicalView\(\)/);
  assert.match(source,/CUSTOM_DENSITY_MIN=-64,CUSTOM_DENSITY_MAX=1870,CUSTOM_DENSITY_DEFAULT=-64/);
  assert.doesNotMatch(source,/medical-back/);
  assert.match(source,/"wasdqe"\.includes\(key\)/);
  assert.match(source,/cameraKeys: new Set\(\)/);
  assert.match(source,/function updateCameraMotion\(now\)/);
  assert.match(source,/window\.addEventListener\("keyup"/);
});
