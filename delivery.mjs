import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { createGzip } from "node:zlib";
import { pipeline } from "node:stream/promises";
import path from "node:path";

const indexes=new Map();
function inside(root,file){const relative=path.relative(root,file);return relative!==""&&!relative.startsWith(`..${path.sep}`)&&relative!==".."&&!path.isAbsolute(relative);}
export function acceptsGzip(header=""){
  const values=String(header).toLowerCase().split(",").map(item=>item.trim().split(";"));
  const match=values.find(item=>item[0]==="gzip")||values.find(item=>item[0]==="*");
  if(!match)return false;
  const q=match.slice(1).find(item=>item.trim().startsWith("q="));return !q||Number(q.trim().slice(2))>0;
}
async function deliveryIndex(root){
  try{
    const file=path.join(root,"index.json"),info=await stat(file),stamp=`${info.mtimeMs}:${info.size}`;
    if(indexes.get(root)?.stamp===stamp)return indexes.get(root).value;
    const value=JSON.parse(await readFile(file,"utf8"));
    if(value.version!==1||!value.files)return null;
    indexes.set(root,{stamp,value});return value;
  }catch{return null;}
}
async function alternative(file,kind,{processedRoot,deliveryRoot},expectedSha=null){
  if(!inside(processedRoot,file))return null;
  const index=await deliveryIndex(deliveryRoot),entry=index?.files?.[path.relative(processedRoot,file).split(path.sep).join("/")];
  if(!entry||entry.encoding!==kind||(expectedSha&&entry.source_sha256!==expectedSha))return null;
  try{
    const source=await stat(file,{bigint:true});
    if(String(source.mtimeNs)!==entry.source_mtime_ns||Number(source.size)!==entry.source_bytes)return null;
    const target=path.resolve(deliveryRoot,entry.file);
    if(!inside(deliveryRoot,target))return null;
    const packed=await stat(target);
    if(packed.size!==entry.bytes)return null;
    return {...entry,file:target};
  }catch{return null;}
}
export const webpAlternative=(file,options)=>alternative(file,"webp-lossless",options);

export async function sendCompressedFile(req,res,file,headers,options){
  const info=await stat(file),offset=options.offset||0,gzip=options.compress!==false&&acceptsGzip(req.headers["accept-encoding"]);
  const packed=gzip&&!offset?await alternative(file,"gzip",options):null;
  const output={...headers,Vary:"Accept-Encoding"};
  // Weak validators identify the same decompressed bytes across encodings.
  if(output.ETag&&!output.ETag.startsWith("W/"))output.ETag=`W/${output.ETag}`;
  if(gzip){output["Content-Encoding"]="gzip";if(packed)output["Content-Length"]=packed.bytes;}
  else output["Content-Length"]=Math.max(0,info.size-offset);
  res.writeHead(200,output);
  const source=createReadStream(packed?.file||file,packed?{}:{start:offset});
  try{
    if(gzip&&!packed)await pipeline(source,createGzip({level:3}),res);
    else await pipeline(source,res);
  }catch(error){if(!res.destroyed)res.destroy(error);}
}

export async function transferManifest(manifest,subject,folder,options){
  // Never mutate/hash a serialized source manifest: hashes describe the bake.
  const levels=await Promise.all((manifest.levels||[]).map(async level=>{
    let bytes=0,complete=true;
    for(const brick of level.bricks||[]){
      const source=path.resolve(options.processedRoot,subject,folder,brick.file);
      const packed=await alternative(source,"gzip",options,brick.sha256);
      bytes+=packed?.bytes??brick.bytes??0;if(!packed)complete=false;
    }
    return {...level,transfer_bytes:bytes,transfer_size_exact:complete};
  }));
  return {...manifest,levels};
}
