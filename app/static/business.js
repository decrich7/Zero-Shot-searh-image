function $(id){ return document.getElementById(id); }
async function post(url, form){ return (await fetch(url,{method:"POST",body:form})).json(); }
function esc(s){return (s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}

let tab = "text";
let q0 = null, lang0 = "auto";
let heat = null;

function setTab(t, el){
  tab = t;
  document.querySelectorAll(".seg").forEach(x=>x.classList.remove("active"));
  el.classList.add("active");
  $("pane-text").hidden = t!=="text";
  $("pane-image").hidden = t!=="image";
  $("pane-combine").hidden = t!=="combine";
  $("results").innerHTML=""; $("status").textContent="";
  const combine = t==="combine";
  $("model").parentElement.style.display = combine ? "none" : "";
  const m=$("model");
  [...m.options].forEach(o=>o.disabled = (t!=="text" && (o.value==="ensemble3"||o.value==="blip2")));
  if(t!=="text" && (m.value==="ensemble3"||m.value==="blip2")) m.value="ensemble";
  $("preciseSwitch").style.opacity = t==="text" ? "1" : ".4";
  $("precise").disabled = t!=="text";
  onModel();
}
function onModel(){ $("weightField").hidden = tab==="combine" || $("model").value !== "ensemble"; }

function draw(data){
  if(data.error){ $("status").textContent="Ошибка: "+data.error; return; }
  if($("meta")) $("meta").textContent = data.translated
    ? ("перевод: "+data.translated+" · кликните результат для карты внимания")
    : "Кликните результат — покажем карту внимания модели.";
  const res=data.results||[];
  if(!res.length){ $("status").textContent="Ничего не найдено."; return; }
  $("status").textContent = data.precise ? "точный режим: пересортировано ITM-головой" : "";
  const click = tab==="text" && q0;
  $("results").innerHTML = res.map(r=>{
    const cap=(r.captions&&r.captions[0])?r.captions[0]:"";
    const mine=r.source==="user"?"<span class='b-mine'>каталог</span>":"";
    const lens=click?"<span class='b-hint'>🔍 внимание</span>":"";
    const on=click?`onclick="openHeat(${r.row})"`:"";
    return `<div class="tile" ${on}>${mine}
      <span class="b-score">${r.score}</span>${lens}
      <img src="/image/${r.row}" loading="lazy">
      <div class="cap">${esc(cap)}</div></div>`;
  }).join("");
}

async function runText(){
  const q=$("q").value.trim(); if(!q) return;
  q0=q; lang0=$("lang").value;
  $("status").textContent = $("precise").checked
    ? "точный режим: пересортировка ITM-головой (на CPU до минуты)…" : "ищем…";
  $("results").innerHTML="";
  const f=new FormData();
  f.append("q",q); f.append("lang",lang0);
  f.append("model",$("model").value); f.append("weight",$("weight").value);
  f.append("precise",$("precise").checked);
  draw(await post("/api/search_text", f));
}
async function runImage(){
  const file=$("imgfile").files[0]; if(!file){alert("выберите файл");return;}
  q0=null;
  $("status").textContent="ищем…"; $("results").innerHTML="";
  const f=new FormData();
  f.append("file",file); f.append("model",$("model").value);
  f.append("weight",$("weight").value);
  draw(await post("/api/search_image", f));
}
async function runCombine(){
  const files=$("combofiles").files;
  if(files.length<2){alert("выберите минимум 2 картинки");return;}
  q0=null;
  $("status").textContent="ищем кадры со всеми объектами…"; $("results").innerHTML="";
  const f=new FormData();
  for(const file of files) f.append("files",file);
  f.append("mode","and");
  draw(await post("/api/search_multi", f));
}

async function openHeat(row){
  if(!q0) return;
  heat=null;
  $("hmModal").hidden=false;
  $("hmToggle").textContent="Скрыть карту";
  $("hmInfo").textContent="считаем карту внимания…";
  const img=$("hmImg");
  img.onload=()=>{ if(heat) drawHeat(); };
  img.src="/image/"+row;
  const f=new FormData();
  f.append("q",q0); f.append("row",row); f.append("lang",lang0);
  const d=await post("/api/heatmap", f);
  if(d.error){ $("hmInfo").textContent="Ошибка: "+d.error; return; }
  heat=d;
  if(img.complete) drawHeat();
  $("hmInfo").textContent="Совпадение (ITM): "+Math.round(d.prob*100)
    +"% · ярче = сильнее внимание модели на слова запроса";
}
function drawHeat(){
  const img=$("hmImg"), cv=$("hmCanvas");
  if(!heat) return;
  const n=heat.size, g=heat.grid;
  const off=document.createElement("canvas"); off.width=n; off.height=n;
  const octx=off.getContext("2d");
  const im=octx.createImageData(n,n);
  for(let y=0;y<n;y++) for(let x=0;x<n;x++){
    const v=g[y][x], i=(y*n+x)*4;
    im.data[i]  =255;
    im.data[i+1]=Math.round(190-150*v);
    im.data[i+2]=Math.round(60-60*v);
    im.data[i+3]=Math.round(255*Math.min(0.85, Math.pow(v,1.2)*0.95));
  }
  octx.putImageData(im,0,0);
  cv.width=img.clientWidth; cv.height=img.clientHeight;
  const ctx=cv.getContext("2d");
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(off,0,0,cv.width,cv.height);
  cv.dataset.on="1";
}
function toggleHeat(){
  const cv=$("hmCanvas");
  if(cv.dataset.on==="1"){ cv.getContext("2d").clearRect(0,0,cv.width,cv.height); cv.dataset.on="0"; $("hmToggle").textContent="Показать карту"; }
  else { drawHeat(); $("hmToggle").textContent="Скрыть карту"; }
}
function closeHeat(){ $("hmModal").hidden=true; }
$("hmModal") && ($("hmModal").addEventListener("click",e=>{ if(e.target.id==="hmModal") closeHeat(); }));

async function uploadZip(){
  const file=$("zip").files[0]; if(!file) return;
  $("progress").hidden=false; $("ptext").textContent="Загружаем архив…";
  const f=new FormData(); f.append("file",file);
  const d=await post("/api/upload_archive", f);
  if(d.error){ $("ptext").textContent="Ошибка: "+d.error; return; }
  const timer=setInterval(async()=>{
    const j=await (await fetch("/api/upload_progress/"+d.job_id)).json();
    $("ptext").textContent="Обработано: "+j.done+" / "+j.total;
    if(j.finished){
      clearInterval(timer);
      $("ptext").textContent="Готово. Добавлено "+j.added+" изображений.";
      $("zip").value=""; refreshStats();
      setTimeout(()=>{$("progress").hidden=true;},4000);
    }
  },1000);
}
async function refreshStats(){
  const d=await (await fetch("/api/stats")).json();
  $("baseInfo").textContent="В индексе: "+d.total+" изображений"+(d.user?(" (загружено: "+d.user+")"):"");
}

onModel(); refreshStats();
