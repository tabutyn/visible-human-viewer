const SCHEMA="visible-human-public-release/v1";
const SHA=/^[0-9a-f]{64}$/;
const EXPECTED_BRICKS={0:240,1:32,2:4};

function fail(message){throw new Error(`Public release: ${message}`);}
function json(value,status=200){return new Response(`${JSON.stringify(value)}\n`,{status,headers:{"Content-Type":"application/json"}});}
function safeManifestUrl(value){
  let url;try{url=new URL(value);}catch{fail("release manifest URL is invalid");}
  if(url.protocol!=="https:"&&!(["127.0.0.1","localhost","::1"].includes(url.hostname)&&url.protocol==="http:"))fail("release manifest must use HTTPS");
  return url;
}
function validateLevel(level,origin,modality){
  if(!Number.isInteger(level?.level)||!Array.isArray(level.dimensions)||level.dimensions.length!==3||!level.dimensions.every(Number.isInteger))fail(`${modality} level is invalid`);
  if(!Array.isArray(level.bricks)||level.bricks.length!==EXPECTED_BRICKS[level.level])fail(`${modality} L${level.level} has an unexpected brick count`);
  const coordinates=new Set();
  for(const brick of level.bricks){
    if(![brick.x,brick.y,brick.z].every(Number.isInteger)||!Array.isArray(brick.extent)||brick.extent.length!==3||!brick.extent.every(Number.isInteger))fail(`${modality} brick coordinates are invalid`);
    if(!Number.isSafeInteger(brick.bytes)||brick.bytes<1||!SHA.test(brick.sha256)||!Number.isSafeInteger(brick.transfer_bytes)||brick.transfer_bytes<1||!SHA.test(brick.transfer_sha256))fail(`${modality} brick integrity metadata is invalid`);
    const identity=`${brick.x}:${brick.y}:${brick.z}`;if(coordinates.has(identity))fail(`${modality} L${level.level} repeats a brick coordinate`);coordinates.add(identity);
    const objectKey=`v1/objects/${brick.transfer_sha256}.gz`;
    const url=safeManifestUrl(brick.url);if(brick.object_key!==objectKey||url.origin!==origin||url.pathname!==`/${objectKey}`||url.search||url.hash)fail(`${modality} brick URL is outside the release origin`);
  }
}

async function hashHex(bytes){
  const digest=await crypto.subtle.digest("SHA-256",bytes);
  return [...new Uint8Array(digest)].map(value=>value.toString(16).padStart(2,"0")).join("");
}

export function validateReleaseManifest(value,manifestUrl){
  const source=safeManifestUrl(manifestUrl);
  if(!value||value.schema!==SCHEMA||value.subject!=="male"||value.experimental!==true)fail("unsupported manifest identity");
  if(value.alignment_knots||value.editor||value.review)fail("editor data is forbidden");
  for(const modality of ["ct","rgb"]){
    const manifest=value[modality];
    if(!manifest||manifest.subject!=="male"||manifest.status!=="ready"||!Array.isArray(manifest.levels))fail(`${modality} manifest is unavailable`);
    for(const level of manifest.levels)validateLevel(level,source.origin,modality);
  }
  const ctLevels=new Map(value.ct.levels.map(level=>[level.level,level])),rgbLevels=new Map(value.rgb.levels.map(level=>[level.level,level]));
  if([...ctLevels.keys()].sort().join(",")!=="0,1,2"||[...rgbLevels.keys()].sort().join(",")!=="0,1,2")fail("L0, L1, and L2 are required");
  for(const [number,ct] of ctLevels){const rgb=rgbLevels.get(number);if(!rgb||ct.dimensions.join(",")!==rgb.dimensions.join(","))fail(`CT/RGB L${number} dimensions differ`);}
  if(!value.default_view||!Number.isFinite(value.default_view.yaw))fail("default view is missing");
  return value;
}

export async function createReleaseData(config,{fetchImpl=globalThis.fetch}={}){
  const manifestUrl=safeManifestUrl(config?.releaseManifestUrl);
  const match=/\/v1\/releases\/([0-9a-f]{64})\/manifest\.json$/.exec(manifestUrl.pathname);if(!match)fail("manifest URL is not content-addressed");
  const response=await fetchImpl(manifestUrl,{cache:"force-cache",credentials:"omit"});
  if(!response.ok)fail(`manifest HTTP ${response.status}`);
  const payload=await response.arrayBuffer();if(await hashHex(payload)!==match[1])fail("manifest SHA-256 differs from its URL");
  let decoded;try{decoded=JSON.parse(new TextDecoder().decode(payload));}catch{fail("manifest JSON is invalid");}
  const manifest=validateReleaseManifest(decoded,manifestUrl);
  const bricks=new Map();
  for(const [modality,volume] of [["ct",manifest.ct],["rgb",manifest.rgb]])for(const level of volume.levels)for(const brick of level.bricks)bricks.set(`${modality}:${level.level}:${brick.x}:${brick.y}:${brick.z}`,brick.url);
  async function request(input,init={}){
    const method=String(init.method||"GET").toUpperCase();if(!["GET","HEAD"].includes(method))return json({error:"Public release is read-only"},403);
    const url=new URL(String(input),globalThis.location?.origin||"https://viewer.invalid"),path=url.pathname;
    if(path==="/api/subjects/male/medical-view")return json(manifest.default_view);
    if(path==="/api/subjects/male/volume/manifest")return json(manifest.ct);
    if(path==="/api/subjects/male/volume/rgb/manifest")return json(manifest.rgb);
    const match=/^\/api\/subjects\/male\/volume\/(rgb\/)?bricks\/(\d+)\/(\d+)\/(\d+)\/(\d+)$/.exec(path);
    if(match){
      const key=`${match[1]?"rgb":"ct"}:${match[2]}:${match[3]}:${match[4]}:${match[5]}`,target=bricks.get(key);
      return target?fetchImpl(target,{...init,method,credentials:"omit"}):json({error:"Brick is not in the pinned release"},404);
    }
    return json({error:"Unavailable in the static public release"},404);
  }
  return Object.freeze({manifest,subject:Object.freeze({id:"male",label:"Male · public release"}),fetch:request});
}
