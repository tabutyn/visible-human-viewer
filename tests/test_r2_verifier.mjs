import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { verifyPublicRelease } from "../verify_visible_human_r2.mjs";

const cache="public, max-age=31536000, immutable, no-transform",viewer="https://visiblehuman.ballrollergames.com",data="https://visiblehuman-data.ballrollergames.com";
const sha=value=>createHash("sha256").update(value).digest("hex");
function fixture(){
  const bodies=new Map(),bricks=[];
  for(let index=0;index<8;index+=1){const body=Buffer.from(`decoded-${index}`),decoded=sha(body),packed=sha(`packed-${index}`),url=`${data}/v1/objects/${packed}.gz`;bodies.set(url,body);bricks.push({x:index,y:0,z:0,extent:[1,1,1],bytes:body.length,sha256:decoded,transfer_bytes:100+index,transfer_sha256:packed,object_key:`v1/objects/${packed}.gz`,url});}
  const manifest={schema:"visible-human-public-release/v1",subject:"male",ct:{levels:[{level:2,bricks:bricks.slice(0,4)}]},rgb:{levels:[{level:2,bricks:bricks.slice(4)}]}};
  const payload=Buffer.from(`${JSON.stringify(manifest)}\n`),manifestSha=sha(payload),manifestUrl=`${data}/v1/releases/${manifestSha}/manifest.json`;
  const headers=extra=>({"access-control-allow-origin":viewer,"access-control-expose-headers":"Content-Length, Content-Encoding, ETag","cache-control":cache,...extra});
  const fetchImpl=async(input,init={})=>{
    const url=String(input);
    if(url===manifestUrl)return new Response(payload,{headers:headers({"content-type":"application/json; charset=utf-8","content-length":String(payload.length)})});
    if(init.headers?.Range)return new Response(Buffer.alloc(32),{status:206,headers:headers({"content-type":"application/octet-stream","content-encoding":"gzip","content-length":"32","content-range":"bytes 0-31/100","etag":"x"})});
    const body=bodies.get(url),brick=bricks.find(item=>item.url===url);return new Response(body,{headers:headers({"content-type":"application/octet-stream","content-encoding":"gzip","content-length":String(brick.transfer_bytes),etag:"x"})});
  };
  return{manifestUrl,fetchImpl,bodies};
}

test("public R2 verifier checks L2 CORS, metadata, hashes, and ranges",async()=>{
  const value=fixture(),result=await verifyPublicRelease({manifestUrl:value.manifestUrl,viewerOrigin:viewer,fetchImpl:value.fetchImpl});
  assert.deepEqual(result,{release_sha256:value.manifestUrl.split("/").at(-2),mode:"L2",brick_references:8,unique_objects:8,decoded_bytes:72,range_verified:true,cors_verified:true,https_verified:true});
});

test("public R2 verifier rejects decoded corruption",async()=>{
  const value=fixture(),original=value.fetchImpl;let changed=false;
  value.fetchImpl=async(input,init)=>{const response=await original(input,init);if(!changed&&String(input).includes("/objects/")&&!init?.headers?.Range){changed=true;return new Response(Buffer.from("wrong"),{headers:response.headers});}return response;};
  await assert.rejects(verifyPublicRelease({manifestUrl:value.manifestUrl,viewerOrigin:viewer,fetchImpl:value.fetchImpl}),/decoded bytes differ/);
});
