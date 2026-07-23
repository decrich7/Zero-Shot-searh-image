let scope = "user";
let pic = null;
let full = null;
let autoRun = false;

const CAL = { text: {lo:0.18, hi:0.55}, image: {lo:0.45, hi:0.90} };
function pct(score, mode){
  const c = CAL[mode] || CAL.text;
  let p = (score - c.lo) / (c.hi - c.lo) * 100;
  return Math.max(1, Math.min(99, Math.round(p)));
}

// Качественные лейблы похожести. Уровень берём из уже откалиброванного
// pct(), поэтому границы автоматически подстраиваются под режим (text/image).
const SIM_BASE = ["совсем не похоже", "немного похоже", "похоже", "сильно похоже"];
const SIM_TOP  = { image: "почти одинаково", text: "точное совпадение" };
function simLevel(score, mode){
  const p = pct(score, mode);
  if (p < 20) return 0;
  if (p < 40) return 1;
  if (p < 60) return 2;
  if (p < 80) return 3;
  return 4;
}
function simLabel(score, mode){
  const lvl = simLevel(score, mode);
  return lvl < 4 ? SIM_BASE[lvl] : (SIM_TOP[mode] || SIM_TOP.image);
}

function $(id){ return document.getElementById(id); }
async function post(url, form){
  const r = await fetch(url, {method:"POST", body:form});
  return r.json();
}
function esc(s){
  return (s || "").replace(/[&<>"]/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
function toast(msg){
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._t);
  t._t = setTimeout(() => { t.hidden = true; }, 2500);
}

function toggleRefine(){
  const p = $("refPanel");
  p.hidden = !p.hidden;
  $("refBtn").classList.toggle("on", !p.hidden);
}
function toggleExclude(force){
  const p = $("negPanel");
  p.hidden = (force === true) ? false : !p.hidden;
  $("negBtn").classList.toggle("on", !p.hidden);
  if(!p.hidden) $("neg").focus();
}

function paintScope(){
  document.querySelectorAll(".seg").forEach(s =>
    s.classList.toggle("active", s.dataset.scope === scope));
}
function setScope(v){
  scope = v;
  paintScope();
  if(hasQuery()) run();
}
function hasQuery(){ return $("query").value.trim() || pic; }

function photoSearch(){
  $("refPanel").hidden = false;
  $("refBtn").classList.add("on");
  autoRun = true;
  $("refFile").click();
}
function onRefFile(){
  const f = $("refFile").files[0];
  if(!f) return;
  full = f; pic = f;
  $("refImg").src = URL.createObjectURL(f);
  $("refEmpty").hidden = true;
  $("refLoaded").hidden = false;
  hideSel();
  $("cropReset").hidden = true;
  $("cropHint").textContent = "Потяните по фото, чтобы искать по фрагменту";
  if(autoRun){ autoRun = false; run(); }
}
function clearRef(){
  full = null; pic = null;
  $("refFile").value = "";
  $("refLoaded").hidden = true;
  $("refEmpty").hidden = false;
  hideSel();
}
function resetCrop(){
  pic = full;
  hideSel();
  $("cropReset").hidden = true;
  $("cropHint").textContent = "Потяните по фото, чтобы искать по фрагменту";
}
function hideSel(){ $("cropSel").hidden = true; }

(function(){
  const wrap = $("cropWrap"), sel = $("cropSel"), img = $("refImg");
  if(!wrap) return;
  let on = false, sx = 0, sy = 0;

  function at(e){
    const r = img.getBoundingClientRect();
    let x = (e.clientX ?? e.touches[0].clientX) - r.left;
    let y = (e.clientY ?? e.touches[0].clientY) - r.top;
    x = Math.max(0, Math.min(img.clientWidth,  x));
    y = Math.max(0, Math.min(img.clientHeight, y));
    return {x, y};
  }
  function place(a, b){
    const left = Math.min(a.x,b.x), top = Math.min(a.y,b.y);
    const w = Math.abs(a.x-b.x), h = Math.abs(a.y-b.y);
    sel.style.left = left+"px"; sel.style.top = top+"px";
    sel.style.width = w+"px";   sel.style.height = h+"px";
    sel.hidden = false;
    return {left, top, w, h};
  }
  wrap.addEventListener("mousedown", e => { e.preventDefault(); on = true; const p = at(e); sx = p.x; sy = p.y; place(p,p); });
  document.addEventListener("mousemove", e => { if(on) place({x:sx,y:sy}, at(e)); });
  document.addEventListener("mouseup", e => {
    if(!on) return; on = false;
    const r = place({x:sx,y:sy}, at(e));
    if(r.w < 10 || r.h < 10){ hideSel(); return; }
    cropTo(r);
  });
})();

function cropTo(r){
  const img = $("refImg");
  const kx = img.naturalWidth / img.clientWidth;
  const ky = img.naturalHeight / img.clientHeight;
  const cv = document.createElement("canvas");
  cv.width = Math.round(r.w*kx); cv.height = Math.round(r.h*ky);
  cv.getContext("2d").drawImage(img, r.left*kx, r.top*ky, r.w*kx, r.h*ky, 0, 0, r.w*kx, r.h*ky);
  cv.toBlob(b => { pic = new File([b], "crop.jpg", {type:"image/jpeg"}); }, "image/jpeg", 0.92);
  $("cropReset").hidden = false;
  $("cropHint").textContent = "Ищем по выделенному фрагменту";
}

async function run(){
  const q = $("query").value.trim();
  const neg = $("neg").value.trim();
  if(!q && !pic) return;

  $("status").textContent = "ищем…";
  $("results").innerHTML = "";
  $("shareBtn").hidden = false;
  syncUrl(q, neg);

  const f = new FormData();
  f.append("q", q);
  f.append("neg", neg);
  f.append("lang", "auto");
  f.append("scope", scope);
  if(pic) f.append("ref", pic);

  draw(await post("/api/search_smart", f), q ? "text" : "image");
}

async function similar(row){
  $("status").textContent = "ищем похожие…";
  $("results").innerHTML = "";
  const r = await fetch("/api/similar/" + row + "?scope=" + scope);
  draw(await r.json(), "image");
}

const LANGS = {
  ru:"русского", uk:"украинского", fr:"французского", de:"немецкого",
  es:"испанского", it:"итальянского", pt:"португальского", pl:"польского",
  tr:"турецкого", zh:"китайского", "zh-cn":"китайского", ja:"японского",
  ko:"корейского", ar:"арабского", nl:"нидерландского", cs:"чешского"
};
function langHint(code){
  const name = LANGS[(code||"").toLowerCase()];
  return name ? (" (перевели с " + name + ")") : "";
}

function draw(data, mode){
  if(data.error){ $("status").textContent = "Ошибка: " + data.error; return; }

  $("translated").textContent = data.translated
    ? ("ищем по: " + data.translated + langHint(data.detected)) : "";

  const res = data.results || [];
  if(res.length === 0){
    $("status").textContent = (scope === "user")
      ? "Среди ваших фото ничего не нашлось. Переключите на «Все» или загрузите архив."
      : "Ничего не нашлось. Попробуйте описать иначе.";
    return;
  }
  $("status").textContent = "";

  $("results").innerHTML = res.map(r => {
    const cap = (r.captions && r.captions[0]) ? r.captions[0] : "";
    const mine = r.source === "user" ? "<span class='b-mine'>моё</span>" : "";
    const lvl = simLevel(r.score, mode);
    return "<div class='tile' onclick='similar(" + r.row + ")'>" +
      mine +
      "<span class='b-score lvl" + lvl + "'>" + simLabel(r.score, mode) + "</span>" +
      "<span class='b-hint'>похожие →</span>" +
      "<img src='/image/" + r.row + "' loading='lazy'>" +
      "<div class='cap'>" + esc(cap) + "</div></div>";
  }).join("");
}

function syncUrl(q, neg){
  const p = new URLSearchParams();
  if(q) p.set("q", q);
  if(neg) p.set("neg", neg);
  p.set("scope", scope);
  history.replaceState(null, "", location.pathname + "?" + p.toString());
}
function share(){
  const q = $("query").value.trim();
  const neg = $("neg").value.trim();
  syncUrl(q, neg);
  const url = location.origin + location.pathname + location.search;
  const note = pic ? "Ссылка скопирована (без фото-примера)" : "Ссылка скопирована";
  if(navigator.clipboard){
    navigator.clipboard.writeText(url).then(() => toast(note), () => toast(url));
  } else {
    toast(url);
  }
}

async function uploadArchive(){
  const f = $("zipFile").files[0];
  if(!f) return;
  $("progress").hidden = false;
  $("progText").textContent = "Загружаем архив…";
  const form = new FormData();
  form.append("file", f);
  const data = await post("/api/upload_archive", form);
  if(data.error){ $("progText").textContent = "Ошибка: " + data.error; return; }
  poll(data.job_id);
}
function poll(jobId){
  const timer = setInterval(async () => {
    const j = await (await fetch("/api/upload_progress/" + jobId)).json();
    $("progText").textContent = "Обрабатываем фото: " + j.done + " из " + j.total;
    if(j.finished){
      clearInterval(timer);
      $("progText").textContent = "Готово! Добавлено " + j.added + " фото."
        + (j.errors ? (" Пропущено: " + j.errors + ".") : "");
      $("zipFile").value = "";
      refreshStats();
      scope = "user"; paintScope();
      setTimeout(() => { $("progress").hidden = true; }, 4000);
    }
  }, 1000);
}

async function refreshStats(){
  const d = await (await fetch("/api/stats")).json();
  const title = $("archiveBox").querySelector(".archive-title");
  if(d.user > 0){
    title.textContent = "В вашем архиве " + d.user + " фото";
    $("scopeBox").hidden = false;
  } else {
    title.textContent = "Свои фото ещё не загружены";
    $("scopeBox").hidden = true;
    scope = "all";
  }
  paintScope();
  return d;
}

async function init(){
  await refreshStats();
  const p = new URLSearchParams(location.search);
  if(p.get("scope")){ scope = p.get("scope"); paintScope(); }
  if(p.get("neg")){ $("neg").value = p.get("neg"); toggleExclude(true); }
  const q = p.get("q");
  if(q){ $("query").value = q; run(); }
}
init();
