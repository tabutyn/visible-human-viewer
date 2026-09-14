// No image, dataset, renderer, or GPU initialization runs before consent.
const gate=document.querySelector("#content-warning"),form=document.querySelector("#content-consent");
const age=document.querySelector("#confirm-age"),content=document.querySelector("#confirm-content");
const enter=document.querySelector("#enter-viewer"),status=document.querySelector("#consent-status");
const key="visible-human-content-consent-v1",lifetime=8*60*60*1000;
let started=false;
function accepted(){try{const value=JSON.parse(sessionStorage.getItem(key));const elapsed=Date.now()-value?.acceptedAt;return value?.version===1&&elapsed>=0&&elapsed<lifetime;}catch{return false;}}
function forget(){try{sessionStorage.removeItem(key);}catch{}}
async function start(){
  if(started)return;started=true;enter.disabled=true;status.textContent="Opening viewer…";
  try{
    let response=await fetch("/release-config.json",{cache:"no-store"});
    if(!response.ok)response=await fetch("/api/config");
    if(!response.ok)throw new Error("Unable to load application configuration.");
    const config=await response.json();
    window.visibleHumanConfig=Object.freeze(config);document.body.dataset.readOnly=String(config.readOnly===true);
    if(config.readOnly||!location.hash)history.replaceState(null,"",`${location.pathname}${location.search}#medical`);
    document.body.dataset.activeTab=["#medical","#align3d"].includes(location.hash)?location.hash.slice(1):"alignment";
    if(config.releaseManifestUrl){
      const {createReleaseData}=await import("/release-data.mjs"),data=await createReleaseData(config);
      window.visibleHumanReleaseData=data;document.body.dataset.releaseMode="static";
      const subjects=document.querySelector("#subject-select");subjects.replaceChildren(new Option(data.subject.label,data.subject.id));subjects.value=data.subject.id;
    }else await import("/viewer.js");
    await import("/medical-renderer.mjs");
    document.querySelector(".viewer-shell").hidden=false;gate.hidden=true;
    document.querySelector("#release-credit").hidden=false;
    window.dispatchEvent(new Event("resize"));
  }catch(error){started=false;forget();status.textContent=error.message;enter.disabled=!(age.checked&&content.checked);}
}
form.addEventListener("change",()=>{enter.disabled=!(age.checked&&content.checked);});
form.addEventListener("submit",(event)=>{
  event.preventDefault();if(!age.checked||!content.checked)return;
  try{sessionStorage.setItem(key,JSON.stringify({version:1,acceptedAt:Date.now()}));}catch{}
  start();
});
document.querySelector("#leave-viewer").addEventListener("click",()=>{
  forget();age.checked=false;content.checked=false;enter.disabled=true;
  if(started){location.reload();return;}
  status.textContent="No cadaver images were loaded. You can close this tab.";
});
document.querySelector("#hide-imagery").addEventListener("click",()=>{
  // Immediately cover the canvas, then dispose the document and GPU context.
  document.querySelector(".viewer-shell").hidden=true;gate.hidden=false;forget();location.reload();
});
// A fast click or browser-restored form state may precede module execution.
enter.disabled=!(age.checked&&content.checked);
form.dataset.ready="true";
if(accepted())start();
