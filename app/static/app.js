let tab = "text";

document.querySelectorAll(".nav-link").forEach(function (a) {
  a.onclick = function (e) {
    e.preventDefault();
    tab = a.dataset.tab;
    document.querySelectorAll(".nav-link").forEach(x => x.classList.remove("active"));
    a.classList.add("active");
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.add("d-none"));
    document.getElementById("tab-" + tab).classList.remove("d-none");
    document.getElementById("model-box").style.display = (tab === "add") ? "none" : "flex";
    document.getElementById("results").innerHTML = "";
    document.getElementById("status").innerHTML = "";
  };
});

document.getElementById("weight").oninput = function () {
  document.getElementById("wval").textContent = this.value;
};
document.getElementById("mixw").oninput = function () {
  document.getElementById("mixval").textContent = this.value;
};
document.getElementById("model").onchange = function () {
  document.getElementById("weight-box").style.display = (this.value === "ensemble") ? "block" : "none";
};

function model() { return document.getElementById("model").value; }
function weight() { return document.getElementById("weight").value; }
function clean() { return document.getElementById("clean").checked; }

function busy() {
  document.getElementById("status").innerHTML = "ищем...";
  document.getElementById("results").innerHTML = "";
}

function draw(data) {
  if (data.error) {
    document.getElementById("status").innerHTML = "Ошибка: " + data.error;
    return;
  }
  let extra = data.translated ? ("перевод: " + data.translated) : "";
  document.getElementById("status").innerHTML = extra;
  document.getElementById("translated").textContent = extra;

  let html = "";
  data.results.forEach(function (r) {
    let caps = r.captions.map(c => "<div class='small text-muted'>" + c + "</div>").join("");
    let tag = r.source === "user" ? "<span class='badge bg-success'>добавлено</span> " : "";
    html += "<div class='col-6 col-md-3'><div class='card h-100'>" +
      "<img src='/image/" + r.row + "' class='card-img-top'>" +
      "<div class='card-body p-2'>" + tag +
      "<div class='small'>score " + r.score + "</div>" + caps +
      "</div></div></div>";
  });
  document.getElementById("results").innerHTML = html;
}

async function post(url, form) {
  let resp = await fetch(url, { method: "POST", body: form });
  return await resp.json();
}

async function searchText() {
  let q = document.getElementById("query").value.trim();
  if (!q) return;
  busy();
  let f = new FormData();
  f.append("q", q);
  f.append("lang", document.getElementById("lang").value);
  f.append("model", model());
  f.append("weight", weight());
  f.append("clean", clean());
  draw(await post("/api/search_text", f));
}

async function searchImage() {
  let file = document.getElementById("imgfile").files[0];
  if (!file) return;
  busy();
  let f = new FormData();
  f.append("file", file);
  f.append("model", model());
  f.append("weight", weight());
  f.append("clean", clean());
  draw(await post("/api/search_image", f));
}

async function searchMulti() {
  let files = document.getElementById("multifiles").files;
  if (files.length < 2) { alert("нужно хотя бы 2 картинки"); return; }
  busy();
  let f = new FormData();
  for (let file of files) f.append("files", file);
  f.append("model", model());
  f.append("weight", document.getElementById("mixw").value);
  draw(await post("/api/search_multi", f));
}

async function addImage() {
  let file = document.getElementById("addfile").files[0];
  if (!file) { alert("выберите картинку"); return; }
  let f = new FormData();
  f.append("file", file);
  let inputs = document.querySelectorAll("#caps input");
  inputs.forEach(i => { if (i.value.trim()) f.append("captions", i.value.trim()); });
  document.getElementById("addmsg").textContent = "добавляем...";
  let data = await post("/api/add", f);
  if (data.error) {
    document.getElementById("addmsg").textContent = "Ошибка: " + data.error;
  } else {
    document.getElementById("addmsg").innerHTML = "<span class='text-success'>Готово! Теперь в базе " + data.total + " картинок.</span>";
    document.getElementById("total").textContent = data.total;
  }
}

(function () {
  let html = "";
  for (let i = 1; i <= 5; i++) {
    html += "<input class='form-control mb-1' placeholder='описание " + i + (i === 1 ? " (обязательно)" : " (необязательно)") + "' style='max-width:400px'>";
  }
  document.getElementById("caps").innerHTML = html;
})();

fetch("/api/stats").then(r => r.json()).then(d => {
  document.getElementById("total").textContent = d.total;
});