#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { lstat, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const CACHE_CONTROL="public, max-age=31536000, immutable, no-transform";
const SHA=/^[0-9a-f]{64}$/;
const EXPECTED_BRICK_REFERENCES=552;
function fail(message){const error=new Error(message);error.code="release_invalid";throw error;}
function sha(bytes){return createHash("sha256").update(bytes).digest("hex");}
function md5(bytes){const digest=createHash("md5").update(bytes);return {hex:digest.copy().digest("hex"),base64:digest.digest("base64")};}
function inside(root,file){const relative=path.relative(root,file);return relative!==""&&relative!==".."&&!relative.startsWith(`..${path.sep}`)&&!path.isAbsolute(relative);}
async function mapLimit(values,limit,fn){
  const output=new Array(values.length);let next=0;
  await Promise.all(Array.from({length:Math.min(limit,values.length)},async()=>{while(true){const index=next++;if(index>=values.length)return;output[index]=await fn(values[index],index);}}));return output;
}
function headMatches(head,expected){
  if(!head)return false;const metadata=Object.fromEntries(Object.entries(head.metadata||{}).map(([key,value])=>[key.toLowerCase(),String(value)]));
  return head.bytes===expected.bytes&&head.contentType===expected.contentType&&head.contentEncoding===expected.contentEncoding&&head.cacheControl===CACHE_CONTROL&&String(head.etag||"").replaceAll('"',"")===expected.md5Hex&&metadata.sha256===expected.sha256&&metadata.decodedsha256===expected.decodedSha256&&metadata.kind===expected.kind;
}

export async function publicationObjects(manifestFile,deliveryRoot){
  const manifestPath=path.resolve(manifestFile),delivery=path.resolve(deliveryRoot),bytes=await readFile(manifestPath),manifest=JSON.parse(bytes);
  if(manifest.schema!=="visible-human-public-release/v1"||manifest.subject!=="male")fail("Only a male v1 release manifest may be published");
  const byKey=new Map();let brickReferences=0;
  for(const modality of ["ct","rgb"])for(const level of manifest[modality]?.levels||[])for(const brick of level.bricks||[]){
    brickReferences+=1;
    if(!SHA.test(brick.transfer_sha256)||!SHA.test(brick.sha256)||brick.object_key!==`v1/objects/${brick.transfer_sha256}.gz`)fail("Brick object identity is invalid");
    const file=path.resolve(delivery,"blobs",`${brick.transfer_sha256}.gz`),info=await lstat(file).catch(()=>null);
    if(!inside(delivery,file)||!info?.isFile()||info.isSymbolicLink()||info.size!==brick.transfer_bytes)fail(`Local delivery blob is missing or unsafe: ${brick.object_key}`);
    const packed=await readFile(file);if(sha(packed)!==brick.transfer_sha256)fail(`Local delivery blob hash differs: ${brick.object_key}`);const checksum=md5(packed);
    const record={key:brick.object_key,file,bytes:brick.transfer_bytes,sha256:brick.transfer_sha256,decodedSha256:brick.sha256,md5Hex:checksum.hex,contentMd5:checksum.base64,kind:"brick",contentType:"application/octet-stream",contentEncoding:"gzip"};
    const previous=byKey.get(record.key);if(previous&&JSON.stringify({...previous,file:null})!==JSON.stringify({...record,file:null}))fail(`Conflicting object identity: ${record.key}`);byKey.set(record.key,record);
  }
  if(brickReferences!==EXPECTED_BRICK_REFERENCES)fail(`Release contains ${brickReferences} brick references; expected ${EXPECTED_BRICK_REFERENCES}`);
  const manifestSha=sha(bytes),manifestKey=`v1/releases/${manifestSha}/manifest.json`;
  const manifestMd5=md5(bytes);
  return {manifest,manifestSha,brickReferences,manifestObject:{key:manifestKey,file:manifestPath,bytes:bytes.length,sha256:manifestSha,decodedSha256:manifestSha,md5Hex:manifestMd5.hex,contentMd5:manifestMd5.base64,kind:"manifest",contentType:"application/json; charset=utf-8",contentEncoding:null},bricks:[...byKey.values()].sort((a,b)=>a.key.localeCompare(b.key))};
}

export async function publishRelease({manifestFile,deliveryRoot,storage,apply=false,concurrency=12}){
  if(!storage||typeof storage.head!=="function"||typeof storage.put!=="function")fail("R2 storage adapter is required");
  const release=await publicationObjects(manifestFile,deliveryRoot),checked=await mapLimit(release.bricks,concurrency,async object=>{
    const remote=await storage.head(object.key);if(remote&&!headMatches(remote,object))fail(`Existing content-addressed object has incorrect metadata: ${object.key}`);return {object,exists:Boolean(remote)};
  });
  const missing=checked.filter(item=>!item.exists).map(item=>item.object);
  if(apply)await mapLimit(missing,Math.min(6,concurrency),async object=>{
    await storage.put(object);const remote=await storage.head(object.key);if(!headMatches(remote,object))fail(`Uploaded object did not verify: ${object.key}`);
  });
  const manifestRemote=await storage.head(release.manifestObject.key);
  if(manifestRemote&&!headMatches(manifestRemote,release.manifestObject))fail("Existing release manifest has incorrect metadata");
  if(apply&&!manifestRemote){await storage.put(release.manifestObject);const verified=await storage.head(release.manifestObject.key);if(!headMatches(verified,release.manifestObject))fail("Uploaded release manifest did not verify");}
  const transferBytes=release.bricks.reduce((sum,item)=>sum+item.bytes,0);
  const lod=Object.fromEntries([0,1,2].map(number=>{
    const entries=["ct","rgb"].flatMap(modality=>release.manifest[modality].levels.find(level=>level.level===number).bricks);
    return [`L${number}`,{objects:entries.length,transfer_bytes:entries.reduce((sum,item)=>sum+item.transfer_bytes,0)}];
  }));
  return {
    mode:apply?"apply":"dry-run",release_sha256:release.manifestSha,
    manifest_key:release.manifestObject.key,manifest_bytes:release.manifestObject.bytes,
    brick_references:release.brickReferences,objects:release.bricks.length,
    deduplicated_bricks:release.brickReferences-release.bricks.length,
    existing_objects:checked.length-missing.length,
    missing_objects:missing.length,transfer_bytes:transferBytes,
    upload_bytes:missing.reduce((sum,item)=>sum+item.bytes,0),lod,
    release_storage_bytes:transferBytes+release.manifestObject.bytes,
    estimated_new_storage_bytes:missing.reduce((sum,item)=>sum+item.bytes,0)+(manifestRemote?0:release.manifestObject.bytes),
    estimated_apply_requests:{class_a:missing.length+(manifestRemote?0:1),class_b:release.bricks.length+1+missing.length+(manifestRemote?0:1)},
    cold_preview_gets:lod.L2.objects,manifest_preexisting:Boolean(manifestRemote),
    manifest_exists:apply||Boolean(manifestRemote),published:apply,
  };
}

function required(name){const value=process.env[name];if(!value)fail(`${name} is required`);return value;}
export async function r2Storage(){
  const endpoint=new URL(required("VHP_R2_ENDPOINT"));if(endpoint.protocol!=="https:"||endpoint.pathname!=="/"||endpoint.search||endpoint.hash)fail("VHP_R2_ENDPOINT must be an HTTPS account origin without a bucket path");
  const bucket=process.env.VHP_R2_BUCKET||"visible-human-public";if(bucket!=="visible-human-public")fail("VHP_R2_BUCKET must be the dedicated visible-human-public bucket");
  const [{S3Client,HeadObjectCommand,PutObjectCommand},{NodeHttpHandler}]=await Promise.all([import("@aws-sdk/client-s3"),import("@smithy/node-http-handler")]);
  const client=new S3Client({region:"auto",endpoint:endpoint.origin,forcePathStyle:true,maxAttempts:3,requestHandler:new NodeHttpHandler({connectionTimeout:10_000,requestTimeout:180_000,socketTimeout:60_000}),credentials:{accessKeyId:required("VHP_R2_ACCESS_KEY_ID"),secretAccessKey:required("VHP_R2_SECRET_ACCESS_KEY")}});
  const notFound=error=>error?.$metadata?.httpStatusCode===404||["NotFound","NoSuchKey"].includes(error?.name);
  return {
    async head(key){try{const value=await client.send(new HeadObjectCommand({Bucket:bucket,Key:key}));return {bytes:Number(value.ContentLength),etag:value.ETag,contentType:value.ContentType,contentEncoding:value.ContentEncoding||null,cacheControl:value.CacheControl,metadata:value.Metadata||{}};}catch(error){if(notFound(error))return null;throw error;}},
    async put(object){await client.send(new PutObjectCommand({Bucket:bucket,Key:object.key,Body:createReadStream(object.file),ContentLength:object.bytes,ContentMD5:object.contentMd5,ContentType:object.contentType,ContentEncoding:object.contentEncoding||undefined,CacheControl:CACHE_CONTROL,StorageClass:"STANDARD",IfNoneMatch:"*",Metadata:{sha256:object.sha256,decodedsha256:object.decodedSha256,kind:object.kind}}));},
  };
}

async function verifySite(siteDirectory,manifestKey,publicOrigin){
  const site=path.resolve(siteDirectory),info=await lstat(site).catch(()=>null);if(!info?.isDirectory()||info.isSymbolicLink())fail("Static Pages site is missing or unsafe");
  const config=JSON.parse(await readFile(path.join(site,"release-config.json"),"utf8"));
  if(config.readOnly!==true||config.automaticVolume!=="preview"||config.releaseManifestUrl!==`${publicOrigin}/${manifestKey}`)fail("Static Pages site is not pinned to this release manifest");
  for(const required of ["index.html","entry.mjs","release-data.mjs","medical-renderer.mjs","_headers"]){const file=await lstat(path.join(site,required)).catch(()=>null);if(!file?.isFile()||file.isSymbolicLink())fail(`Static Pages site is missing ${required}`);}
  for(const forbidden of ["viewer.js","server.mjs","review-server.mjs"]){if(await lstat(path.join(site,forbidden)).catch(()=>null))fail(`Static Pages site contains forbidden ${forbidden}`);}
  return site;
}

async function deployPages(site,project){
  if(!/^[a-z0-9][a-z0-9-]{0,57}[a-z0-9]$/.test(project))fail("Pages project name is invalid");
  const wrangler=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"node_modules/.bin/wrangler");
  const info=await lstat(wrangler).catch(()=>null);if(!info?.isFile()&&!info?.isSymbolicLink())fail("Wrangler is not installed; run npm ci");
  await new Promise((resolve,reject)=>{
    const child=spawn(wrangler,["pages","deploy",site,"--project-name",project,"--branch","main","--commit-dirty=true"],{stdio:"inherit",env:process.env});
    child.once("error",reject);child.once("exit",code=>code===0?resolve():reject(new Error(`Wrangler Pages deploy exited ${code}`)));
  });
}

async function main(){
  const args=process.argv.slice(2),take=name=>{const index=args.indexOf(name);if(index<0||!args[index+1])fail(`${name} is required`);return args[index+1];};
  if(args.includes("--env-file"))process.loadEnvFile(path.resolve(take("--env-file")));
  if(args.includes("--apply")===args.includes("--dry-run"))fail("Choose exactly one of --dry-run or --apply");
  const manifest=take("--manifest"),delivery=take("--delivery-root"),site=take("--site"),project=process.env.VHP_PAGES_PROJECT||"visible-human",publicBase=new URL(required("VHP_R2_PUBLIC_BASE_URL"));if(publicBase.protocol!=="https:"||publicBase.pathname!=="/"||publicBase.username||publicBase.password||publicBase.search||publicBase.hash)fail("VHP_R2_PUBLIC_BASE_URL must be an HTTPS origin");
  const parsed=JSON.parse(await readFile(manifest,"utf8"));for(const modality of ["ct","rgb"])for(const level of parsed[modality]?.levels||[])for(const brick of level.bricks||[])if(new URL(brick.url).origin!==publicBase.origin)fail("Manifest object origin differs from VHP_R2_PUBLIC_BASE_URL");
  const manifestSha=sha(await readFile(manifest)),manifestKey=`v1/releases/${manifestSha}/manifest.json`,verifiedSite=await verifySite(site,manifestKey,publicBase.origin);
  const apply=args.includes("--apply"),result=await publishRelease({manifestFile:manifest,deliveryRoot:delivery,storage:await r2Storage(),apply});
  if(apply)await deployPages(verifiedSite,project);
  console.log(JSON.stringify({...result,pages_project:project,pages_deployed:apply},null,2));
}

if(path.resolve(process.argv[1]||"")===fileURLToPath(import.meta.url))main().catch(error=>{console.error(`R2 publication failed: ${error.message}`);process.exitCode=1;});
