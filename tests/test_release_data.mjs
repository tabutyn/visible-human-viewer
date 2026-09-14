import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { createReleaseData, validateReleaseManifest } from "../release-data.mjs";

const digest=value=>createHash("sha256").update(String(value)).digest("hex");
function manifest(){
  const counts={0:240,1:32,2:4};
  const volume=(modality)=>({subject:"male",status:"ready",levels:[0,1,2].map(level=>({level,dimensions:[8,8,8],bricks:Array.from({length:counts[level]},(_,index)=>{const transfer=digest(`${modality}${level}${index}`);return{x:index,y:0,z:0,extent:[8,8,8],bytes:modality==="ct"?1024:1536,sha256:digest(modality==="ct"?"a":"b"),transfer_bytes:100,transfer_sha256:transfer,object_key:`v1/objects/${transfer}.gz`,url:`https://visiblehuman-data.ballrollergames.com/v1/objects/${transfer}.gz`};})}))});
  return {schema:"visible-human-public-release/v1",subject:"male",experimental:true,default_view:{yaw:0},ct:volume("ct"),rgb:volume("rgb")};
}

test("release manifest accepts only pinned male LOD object URLs",()=>{
  const value=manifest();assert.equal(validateReleaseManifest(value,"https://visiblehuman-data.ballrollergames.com/v1/releases/x/manifest.json"),value);
  assert.throws(()=>validateReleaseManifest({...value,subject:"female"},"https://visiblehuman-data.ballrollergames.com/x"),/identity/);
  const escaped=structuredClone(value);escaped.ct.levels[0].bricks[0].url="https://other.example/v1/objects/a.gz";
  assert.throws(()=>validateReleaseManifest(escaped,"https://visiblehuman-data.ballrollergames.com/x"),/outside/);
  assert.throws(()=>validateReleaseManifest({...value,editor:{}},"https://visiblehuman-data.ballrollergames.com/x"),/forbidden/);
});

test("release data maps only read-only volume endpoints",async()=>{
  const value=manifest(),calls=[],payload=Buffer.from(`${JSON.stringify(value)}\n`),releaseSha=createHash("sha256").update(payload).digest("hex");
  const fetchImpl=async(input)=>{calls.push(String(input));return String(input).includes("manifest.json")?new Response(payload):new Response(new Uint8Array([1,2,3]));};
  const data=await createReleaseData({releaseManifestUrl:`https://visiblehuman-data.ballrollergames.com/v1/releases/${releaseSha}/manifest.json`},{fetchImpl});
  assert.equal(data.subject.id,"male");
  assert.equal((await data.fetch("/api/subjects/male/volume/manifest")).status,200);
  assert.equal((await data.fetch("/api/subjects/male/volume/build",{method:"POST"})).status,403);
  assert.equal((await data.fetch("/api/subjects/male/rgb/layers/10")).status,404);
  const brick=await data.fetch("/api/subjects/male/volume/bricks/2/0/0/0?v=x");assert.equal(brick.status,200);
  assert.equal(calls.at(-1),value.ct.levels[2].bricks[0].url);
});

test("release data rejects a manifest whose bytes do not match its pinned URL",async()=>{
  const payload=Buffer.from(JSON.stringify(manifest()));
  await assert.rejects(createReleaseData({releaseManifestUrl:`https://visiblehuman-data.ballrollergames.com/v1/releases/${digest("0")}/manifest.json`},{fetchImpl:async()=>new Response(payload)}),/SHA-256 differs/);
});
