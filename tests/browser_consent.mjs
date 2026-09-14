export async function acceptContentWarning(evaluate,waitFor){
  await waitFor("document.querySelector('#content-consent')?.dataset.ready==='true'");
  await evaluate(`(()=>{
    if(document.querySelector('#content-warning').hidden)return;
    for(const id of ['confirm-age','confirm-content'])document.getElementById(id).click();
    document.querySelector('#enter-viewer').click();
  })()`);
  await waitFor("document.querySelector('#content-warning').hidden");
}
export async function requestNativeDetail(evaluate,waitFor){
  for(let i=0;i<4;i++){
    await waitFor("document.querySelector('#medical-workspace')?.dataset.renderedRgbLevel==='0'||(!document.querySelector('#volume-more-detail').hidden&&!document.querySelector('#volume-more-detail').disabled)");
    if(await evaluate("document.querySelector('#medical-workspace').dataset.renderedRgbLevel==='0'"))return;
    await evaluate("document.querySelector('#volume-more-detail').click()");
  }
  throw new Error("Native volume did not load after explicit requests");
}
