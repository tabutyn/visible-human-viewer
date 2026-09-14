import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { mkdtemp,mkdir,readFile,writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { publishRelease } from "../publish_visible_human_r2.mjs";

const sha=value=>createHash("sha256").update(value).digest("hex"),counts={0:240,1:32,2:4};
async function fixture(){
  const root=await mkdtemp(path.join(tmpdir(),"vh-r2-publish-"));
  const delivery=path.join(root,"delivery");
  const blobs=path.join(delivery,"blobs");
  await mkdir(blobs,{recursive:true});
  let serial=0;
  async function volume(modality){
    const levels=[];
    for(const level of [0,1,2]){
      const bricks=[];
      for(let index=0;index<counts[level];index+=1){
        const packed=Buffer.from(`${modality}-${level}-${index}-${serial++}`);
        const packedSha=sha(packed);
        const decodedSha=sha(`decoded-${serial}`);
        await writeFile(path.join(blobs,`${packedSha}.gz`),packed);
        bricks.push({
          x:index,y:0,z:0,extent:[1,1,1],bytes:2,sha256:decodedSha,
          transfer_bytes:packed.length,transfer_sha256:packedSha,
          object_key:`v1/objects/${packedSha}.gz`,
          url:`https://visiblehuman-data.ballrollergames.com/v1/objects/${packedSha}.gz`,
        });
      }
      levels.push({level,dimensions:[1,1,1],bricks});
    }
    return{levels};
  }
  const manifest={schema:"visible-human-public-release/v1",subject:"male",ct:await volume("ct"),rgb:await volume("rgb")},file=path.join(root,"manifest.json");await writeFile(file,JSON.stringify(manifest));return{root,delivery,file};
}
class MemoryR2{
  constructor(){this.values=new Map();}
  async head(key){return this.values.get(key)||null;}
  async put(object){this.values.set(object.key,{bytes:object.bytes,etag:`"${object.md5Hex}"`,contentType:object.contentType,contentEncoding:object.contentEncoding,cacheControl:"public, max-age=31536000, immutable, no-transform",metadata:{sha256:object.sha256,decodedsha256:object.decodedSha256,kind:object.kind}});}
}
test("publisher dry-runs, commits objects before the manifest, and resumes",async()=>{
  const f=await fixture(),storage=new MemoryR2(),dry=await publishRelease({manifestFile:f.file,deliveryRoot:f.delivery,storage});assert.equal(dry.brick_references,552);assert.equal(dry.objects,552);assert.equal(dry.missing_objects,552);assert.equal(dry.published,false);assert.equal(dry.cold_preview_gets,8);assert.equal(dry.lod.L2.objects,8);assert.equal(dry.estimated_apply_requests.class_a,553);assert.equal(storage.values.size,0);
  const applied=await publishRelease({manifestFile:f.file,deliveryRoot:f.delivery,storage,apply:true});assert.equal(applied.published,true);assert.equal(applied.manifest_exists,true);assert.equal(storage.values.size,553);assert.ok(storage.values.has(applied.manifest_key));
  const resumed=await publishRelease({manifestFile:f.file,deliveryRoot:f.delivery,storage});assert.equal(resumed.missing_objects,0);assert.equal(resumed.manifest_exists,true);
  const first=[...storage.values.keys()].find(key=>key.includes("/objects/"));storage.values.get(first).contentEncoding=null;
  await assert.rejects(publishRelease({manifestFile:f.file,deliveryRoot:f.delivery,storage}),/incorrect metadata/);
});

test("publisher preserves 552 brick references while deduplicating identical content",async()=>{
  const f=await fixture(),manifest=JSON.parse(await readFile(f.file,"utf8"));
  const first=manifest.ct.levels[0].bricks[0],second=manifest.ct.levels[0].bricks[1];
  manifest.ct.levels[0].bricks[1]={...first,x:second.x,y:second.y,z:second.z};
  await writeFile(f.file,JSON.stringify(manifest));
  const result=await publishRelease({manifestFile:f.file,deliveryRoot:f.delivery,storage:new MemoryR2()});
  assert.equal(result.brick_references,552);assert.equal(result.objects,551);assert.equal(result.deduplicated_bricks,1);
});
