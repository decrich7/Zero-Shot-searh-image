import io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from app.engine import Engine

app = FastAPI(title = "Поиск картинок")
HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory = HERE / "static"), name = "static")

engine = None



@app.on_event("startup")
def startup():
    global engine
    engine = Engine()



@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
@app.get("/api/stats")
def stats():
    lenght = {"total" : len(engine.meta)}
    return lenght



@app.post("/api/search_text")
def search_text(q: str = Form(...), lang: str = Form("auto"),model: str = Form("ensemble"),weight: float = Form(0.75),clean: bool = Form(False)):
    return engine.search_text(q.strip(), lang, model, weight, clean)

@app.post("/api/search_image")
def search_image(file: UploadFile = File(...),model: str = Form("ensemble"),weight: float = Form(0.75),clean: bool = Form(False)):
    image = read_image(file)
    return engine.search_image(image, model, weight, clean)

@app.post("/api/search_multi")
def search_multi(files: list[UploadFile] = File(...),model: str = Form("clip"),weight: float = Form(0.5)):
    images = []
    for f in files:
        images.append(read_image(f))
    if len(images) < 2:
        return JSONResponse({"error": "нужно хотя бы 2 картинки"}, 
                            status_code = 400)
    
    return engine.search_multi(images, model, weight)


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
@app.get("/image/{row}")
def image(row: int):
    return FileResponse(engine.meta[row]["path"])

def read_image(file: UploadFile):
    data = file.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")



