#!/usr/bin/env node
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";

const SHA=/^[0-9a-f]{64}$/;
const CACHE_CONTROL="public, max-age=31536000, immutable, no-transform";
function fail(message){throw new Error(message);}
function sha(bytes){return createHash("sha256").update(bytes).digest("hex");}
function cors(response,origin,label){
  if(response.headers.get("access-control-allow-origin")!==origin)fail(`${label}: CORS origin is missing or incorrect`);
  const exposed=(response.headers.get("access-control-expose-headers")||"").toLowerCase();
  for(const header of ["content-length","content-encoding","etag"])if(!exposed.split(",").map(value=>value.trim()).includes(header))fail(`${label}: CORS does not expose ${header}`);
}
async function limited(values,limit,fn){let next=0;await Promise.all(Array.from({length:Math.min(limit,values.length)},async()=>{while(next<values.length){const index=next++;await fn(values[index],index);}}));}

export async function verifyPublicRelease({manifestUrl,viewerOrigin="https://visiblehuman.ballrollergames.com",all=false,fetchImpl=fetch}){
  const url=new URL(manifestUrl),origin=new URL(viewerOrigin).origin;
  if(url.protocol!=="https:"||new URL(viewerOrigin).protocol!=="https:")fail("Production manifest and viewer origins must use HTTPS");
  const identity=/\/v1\/releases\/([0-9a-f]{64})\/manifest\.json$/.exec(url.pathname);if(!identity)fail("Manifest URL is not content-addressed");
  const manifestResponse=await fetchImpl(url,{headers:{Origin:origin}});if(!manifestResponse.ok)fail(`Manifest HTTP ${manifestResponse.status}`);cors(manifestResponse,origin,"manifest");if(manifestResponse.headers.get("cache-control")!==CACHE_CONTROL)fail("Manifest cache policy is incorrect");
  const manifestBytes=new Uint8Array(await manifestResponse.arrayBuffer());if(sha(manifestBytes)!==identity[1])fail("Manifest bytes do not match the URL SHA-256");const manifest=JSON.parse(new TextDecoder().decode(manifestBytes));
  if(manifest.schema!=="visible-human-public-release/v1"||manifest.subject!=="male")fail("Manifest identity is invalid");
  const references=[];for(const modality of ["ct","rgb"])for(const level of manifest[modality]?.levels||[])if(all||level.level===2)for(const brick of level.bricks||[])references.push({modality,level:level.level,...brick});
  const unique=new Map();for(const brick of references){const previous=unique.get(brick.url);if(previous&&(previous.sha256!==brick.sha256||previous.transfer_sha256!==brick.transfer_sha256))fail("A shared object URL has conflicting hashes");unique.set(brick.url,brick);}
  await limited([...unique.values()],4,async brick=>{
    const objectUrl=new URL(brick.url);if(objectUrl.protocol!=="https:"||objectUrl.origin!==url.origin||objectUrl.pathname!==`/v1/objects/${brick.transfer_sha256}.gz`||!SHA.test(brick.sha256))fail("Brick URL or hash is invalid");
    const response=await fetchImpl(objectUrl,{headers:{Origin:origin}});if(!response.ok)fail(`${brick.modality} L${brick.level}: HTTP ${response.status}`);cors(response,origin,`${brick.modality} L${brick.level}`);
    if(response.headers.get("content-type")!=="application/octet-stream"||response.headers.get("content-encoding")!=="gzip"||response.headers.get("cache-control")!==CACHE_CONTROL)fail(`${brick.modality} L${brick.level}: object metadata is incorrect`);
    if(Number(response.headers.get("content-length"))!==brick.transfer_bytes)fail(`${brick.modality} L${brick.level}: compressed byte length differs`);
    const decoded=new Uint8Array(await response.arrayBuffer());if(decoded.byteLength!==brick.bytes||sha(decoded)!==brick.sha256)fail(`${brick.modality} L${brick.level}: decoded bytes differ`);
  });
  const sample=[...unique.values()][0];if(!sample)fail("No release bricks were selected");
  const range=await fetchImpl(sample.url,{headers:{Origin:origin,Range:"bytes=0-31"}});cors(range,origin,"range request");if(range.status!==206||!/^bytes 0-31\//.test(range.headers.get("content-range")||""))fail("R2 custom domain did not honor a byte range");await range.body?.cancel();
  return {release_sha256:identity[1],mode:all?"all":"L2",brick_references:references.length,unique_objects:unique.size,decoded_bytes:references.reduce((sum,item)=>sum+item.bytes,0),range_verified:true,cors_verified:true,https_verified:true};
}

async function main(){const args=process.argv.slice(2),take=name=>{const index=args.indexOf(name);if(index<0||!args[index+1])fail(`${name} is required`);return args[index+1];};console.log(JSON.stringify(await verifyPublicRelease({manifestUrl:take("--manifest"),viewerOrigin:args.includes("--viewer-origin")?take("--viewer-origin"):undefined,all:args.includes("--all")}),null,2));}
if(path.resolve(process.argv[1]||"")===fileURLToPath(import.meta.url))main().catch(error=>{console.error(`R2 verification failed: ${error.message}`);process.exitCode=1;});
