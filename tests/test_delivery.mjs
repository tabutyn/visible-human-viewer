import assert from "node:assert/strict";
import test from "node:test";
import http from "node:http";
import { mkdtemp,mkdir,readFile,writeFile,stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { gzipSync,gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { acceptsGzip,sendCompressedFile,transferManifest } from "../delivery.mjs";

test("HTTP gzip negotiation respects an explicit refusal",()=>{
  assert.equal(acceptsGzip("br, gzip"),true);
  assert.equal(acceptsGzip("gzip;q=0, *;q=1"),false);
  assert.equal(acceptsGzip("*;q=.5"),true);
  assert.equal(acceptsGzip(""),false);
});
test("compressed delivery is byte-exact, varies by encoding, and rejects stale copies",async()=>{
  const root=await mkdtemp(path.join(tmpdir(),"vhp-delivery-test-")),processedRoot=path.join(root,"processed"),deliveryRoot=path.join(root,"delivery");
  const source=path.join(processedRoot,"male","volume-v1","brick.u16le");
  await mkdir(path.dirname(source),{recursive:true});await mkdir(deliveryRoot);
  const raw=Buffer.alloc(32768);for(let i=0;i<raw.length;i+=2)raw.writeUInt16LE(i%4096,i);
  await writeFile(source,raw);const packed=gzipSync(raw),info=await stat(source,{bigint:true}),hash=createHash("sha256").update(raw).digest("hex");
  await writeFile(path.join(deliveryRoot,"brick.gz"),packed);
  await writeFile(path.join(deliveryRoot,"index.json"),JSON.stringify({version:1,files:{"male/volume-v1/brick.u16le":{encoding:"gzip",file:"brick.gz",bytes:packed.length,source_bytes:raw.length,source_mtime_ns:String(info.mtimeNs),source_sha256:hash}}}));
  const options={processedRoot,deliveryRoot};
  const server=http.createServer((req,res)=>sendCompressedFile(req,res,source,{"Content-Type":"application/octet-stream",ETag:`"${hash}"`},options));
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const get=encoding=>new Promise((resolve,reject)=>http.get({host:"127.0.0.1",port:server.address().port,headers:{"Accept-Encoding":encoding}},res=>{const chunks=[];res.on("data",c=>chunks.push(c));res.on("end",()=>resolve({headers:res.headers,body:Buffer.concat(chunks)}));}).on("error",reject));
  try{
    let value=await get("gzip");assert.equal(value.headers["content-encoding"],"gzip");assert.equal(Number(value.headers["content-length"]),packed.length);assert.equal(value.headers.vary,"Accept-Encoding");assert.deepEqual(gunzipSync(value.body),raw);
    value=await get("gzip;q=0");assert.equal(value.headers["content-encoding"],undefined);assert.deepEqual(value.body,raw);
    const manifest={levels:[{level:0,bricks:[{file:"brick.u16le",bytes:raw.length,sha256:hash}]}]};
    assert.equal((await transferManifest(manifest,"male","volume-v1",options)).levels[0].transfer_size_exact,true);
    assert.equal(manifest.levels[0].transfer_bytes,undefined);
    const changed=Buffer.from(raw);changed[0]=231;await writeFile(source,changed);
    value=await get("gzip");assert.deepEqual(gunzipSync(value.body),changed);assert.equal(value.headers["content-length"],undefined);
    assert.equal((await transferManifest(manifest,"male","volume-v1",options)).levels[0].transfer_size_exact,false);
  }finally{await new Promise(resolve=>server.close(resolve));}
});
