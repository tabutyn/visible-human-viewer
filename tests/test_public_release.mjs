import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { mkdtemp,readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SERVER = fileURLToPath(new URL("../server.mjs", import.meta.url));

test("public mode is read-only at the server and defaults to a consent-gated preview",async()=>{
  const root=await mkdtemp(path.join(tmpdir(),"vhp-public-test-")),port=48000+Math.floor(Math.random()*1000);
  const child=spawn(process.execPath,[SERVER],{env:{...process.env,PORT:String(port),HOST:"127.0.0.1",VISIBLE_HUMAN_ROOT:root,VISIBLE_HUMAN_PROCESSED_ROOT:root,VISIBLE_HUMAN_MODE:"public",VISIBLE_HUMAN_AUTO_DETAIL:"native"}});
  await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error("server timeout")),5000);child.stdout.once("data",()=>{clearTimeout(timer);resolve();});child.once("error",reject);});
  const base=`http://127.0.0.1:${port}`;
  try{
    const config=await(await fetch(`${base}/api/config`)).json();assert.equal(config.readOnly,true);assert.equal(config.automaticVolume,"preview");
    for(const [method,url] of [["PUT","subjects/male/medical-view"],["POST","subjects/male/volume/build"],["POST","subjects/male/volume/rgb/build"],["PUT","review/male/frames/0"],["DELETE","review/male/frames/0"],["POST","review/male/optimize"],["POST","review/male/refine/0"],["PUT","review/male/candidate"],["PUT","review/male/boundaries/a"],["POST","review/male/rebuild"]]){
      assert.equal((await fetch(`${base}/api/${url}`,{method,headers:{"content-type":"application/json"},body:"{}"})).status,403,`${method} ${url}`);
    }
    assert.equal((await fetch(`${base}/api/review/jobs/1`)).status,404);
    assert.equal((await fetch(`${base}/api/subjects/female/manifest`)).status,404);
    assert.deepEqual(await readdir(root),[]);
    const html=await(await fetch(base)).text();assert.match(html,/real human cadaver/);assert.match(html,/id="confirm-age" type="checkbox" required/);assert.match(html,/id="confirm-content" type="checkbox" required/);assert.match(html,/viewer-shell" hidden/);
    assert.doesNotMatch(html,/<script[^>]+src="\/(viewer\.js|medical-renderer\.mjs)"/);
    assert.equal((await fetch(`${base}/LICENSE`)).status,200);
    assert.match(await(await fetch(`${base}/NOTICE.md`)).text(),/not an NLM-endorsed/);
  }finally{child.kill();}
});
