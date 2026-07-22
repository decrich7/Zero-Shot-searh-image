import io
import uuid
import zipfile
import tempfile
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from app.engine import Engine

app = FastAPI(title = "Поиск картинок")
HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory = HERE / "static"), name = "static")

# Расширения картинок, которые вытаскиваем из ZIP-архива.
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

engine = None

# Прогресс фоновой заливки архивов: job_id -> {done, total, added, errors, finished}
JOBS = {}



@app.on_event("startup")
def startup():
    global engine
    engine = Engine()



# ---------- страницы ----------

@app.get("/")
def index():
    # Пользовательская версия (B2C): личный фотоархив, без технических ручек.
    return FileResponse(HERE / "static" / "user.html")


@app.get("/business")
def business():
    # Технический интерфейс (будущий B2B): выбор модели, вес, метрики.
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/stats")
def stats():
    total = len(engine.meta)
    user = sum(1 for m in engine.meta if m["source"] == "user")
    return {"total": total, "user": user}


# ---------- поиск (B2C) ----------

@app.post("/api/search_smart")
def search_smart(q: str = Form(""),
                 neg: str = Form(""),
                 lang: str = Form("auto"),
                 scope: str = Form("all"),
                 ref_weight: float = Form(0.5),
                 neg_weight: float = Form(0.5),
                 ref: UploadFile | None = File(None)):
    ref_image = read_image(ref) if ref is not None else None
    return engine.search_smart(q.strip(), lang, neg, ref_image,
                               ref_weight, neg_weight, scope)


@app.get("/api/similar/{row}")
def similar(row: int, scope: str = "all"):
    if row < 0 or row >= len(engine.meta):
        return JSONResponse({"error": "нет такой картинки"}, status_code = 404)
    return engine.search_similar(row, scope)


# ---------- поиск (технический /business) ----------

@app.post("/api/search_text")
def search_text(q: str = Form(...), lang: str = Form("auto"),model: str = Form("ensemble"),weight: float = Form(0.75),clean: bool = Form(True),precise: bool = Form(False)):
    return engine.search_text(q.strip(), lang, model, weight, clean, precise)


@app.post("/api/heatmap")
def heatmap(q: str = Form(...), row: int = Form(...), lang: str = Form("auto")):
    text = engine.to_english(q.strip(), lang)["en"]
    if row < 0 or row >= len(engine.meta):
        return JSONResponse({"error": "нет такой картинки"}, status_code = 404)
    return engine.attention_map(text, row)

@app.post("/api/search_image")
def search_image(file: UploadFile = File(...),model: str = Form("ensemble"),weight: float = Form(0.75),clean: bool = Form(True)):
    image = read_image(file)
    return engine.search_image(image, model, weight, clean)

@app.post("/api/search_multi")
def search_multi(files: list[UploadFile] = File(...),model: str = Form("clip"),weight: float = Form(0.5),mode: str = Form("blend")):
    images = []
    for f in files:
        images.append(read_image(f))
    if len(images) < 2:
        return JSONResponse({"error": "нужно хотя бы 2 картинки"},
                            status_code = 400)

    return engine.search_multi(images, model, weight, mode)


# ---------- добавление в базу ----------

@app.post("/api/add")
def add(file: UploadFile = File(...),captions: list[str] = Form(...)):
    caps = []
    for c in captions:
        if c.strip():
            caps.append(c.strip())

    if not caps:
        return JSONResponse({"error": "нужно хотя бы одно описание"},
                            status_code = 400)
    caps = caps[:5]
    image = read_image(file)
    row = engine.add_image(image, caps)
    return {"ok": True,
            "row": row,
            "total": len(engine.meta)}


@app.post("/api/upload_archive")
def upload_archive(file: UploadFile = File(...)):
    """Принимает ZIP с фото, распаковывает картинки во временную папку и
    запускает фоновую заливку: каждому фото BLIP генерирует подпись и
    добавляет его в базу. Прогресс отслеживается по job_id."""
    data = file.file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return JSONResponse({"error": "это не ZIP-архив"}, status_code = 400)

    tmp = Path(tempfile.mkdtemp(prefix = "archive_"))
    paths = []
    with zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if Path(name).suffix.lower() not in EXTS:
                continue
            # берём только имя файла — защита от zip-slip
            target = tmp / ("%d_%s" % (len(paths), Path(name).name))
            with open(target, "wb") as out:
                out.write(zf.read(name))
            paths.append(target)

    if not paths:
        return JSONResponse({"error": "в архиве нет картинок"}, status_code = 400)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"done": 0, "total": len(paths),
                    "added": 0, "errors": 0, "finished": False}

    t = threading.Thread(target = _ingest_job, args = (job_id, paths), daemon = True)
    t.start()
    return {"job_id": job_id, "total": len(paths)}


@app.get("/api/upload_progress/{job_id}")
def upload_progress(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse({"error": "нет такой задачи"}, status_code = 404)
    return {**job, "total_base": len(engine.meta)}


def _ingest_job(job_id, paths):
    job = JOBS[job_id]
    for p in paths:
        try:
            image = Image.open(p).convert("RGB")
            caption = engine.generate_caption(image)
            engine.add_image(image, [caption])
            job["added"] += 1
        except Exception:
            job["errors"] += 1
        job["done"] += 1
    job["finished"] = True


@app.get("/image/{row}")
def image(row: int):
    return FileResponse(engine.meta[row]["path"])

def read_image(file: UploadFile):
    data = file.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")
