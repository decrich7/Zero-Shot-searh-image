import json
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers import BlipProcessor, BlipForImageTextRetrieval
from transformers import MarianMTModel, MarianTokenizer

torch.set_num_threads(4)
device = "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DATA"
IMAGES = DATA / "val2017"
UPLOADS = DATA / "user_uploads"
MODELS = ROOT / "MODELS"
EMB = ROOT / "embeddings_output"

USER_BLIP = EMB / "user_blip_img.npy"
USER_CLIP = EMB / "user_clip_img.npy"
USER_META = EMB / "user_meta.json"

CLEAN_Q = 0.10

def norm(x):
    lenght = np.linalg.norm(x)
    if lenght == 0:
        return x
    return x / lenght


class Engine:


    def __init__(self):
        clip_path = MODELS / "clip-vit-large-patch14"
        self.clip = CLIPModel.from_pretrained(clip_path).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(clip_path)

        blip_path = MODELS / "blip-itm-large-flickr"
        self.blip = BlipForImageTextRetrieval.from_pretrained(blip_path).to(device).eval()
        self.blip_proc = BlipProcessor.from_pretrained(blip_path)

        mt_path = MODELS / "opus-mt-ru-en"
        self.mt = MarianMTModel.from_pretrained(mt_path).to(device).eval()
        self.mt_tok = MarianTokenizer.from_pretrained(mt_path)

        self.load_base()
        self.load_user()
        self.find_bad_captions()
        self.build_faiss()


    def load_base(self):
        caps = pd.read_csv(DATA / "captions.csv")
        order = pd.read_csv(EMB / "image_ids_order.csv")["image_id"].tolist()

        file_by_id = {}
        ru_by_id = {}
        en_by_id = {}

        for _ , row in caps.iterrows():
            image_id = row["image_id"]
            file_by_id[image_id] = row["file_name"]

            ru = str(row["caption_ru"])
            en = str(row["caption_en"])

            if ru and ru != "nan":
                if image_id not in ru_by_id:
                    ru_by_id[image_id] = []
                ru_by_id[image_id].append(ru)

            if en and en != "nan":
                if image_id not in en_by_id:
                    en_by_id[image_id] = []
                en_by_id[image_id].append(en)

        self.meta = []
        for image_id in order:
            caps_ru = ru_by_id.get(image_id) or en_by_id.get(image_id, [])

            self.meta.append({
                "path": str(IMAGES / file_by_id[image_id]),
                "captions": caps_ru,
                "source": "base",
            })

        self.base_clip = np.load(EMB / "clip_image_embeddings.npy").astype("float32")
        self.base_blip = np.load(EMB / "blip_image_embeddings.npy").astype("float32")

        self.txt_clip = np.load(EMB / "clip_text_embeddings.npy").astype("float32")
        self.txt_blip = np.load(EMB / "blip_text_embeddings.npy").astype("float32")
        self.txt_ids = pd.read_csv(EMB / "caption_ids_order.csv")["image_id"].tolist()

        self.img_ids = order

        self.clip_img = self.base_clip
        self.blip_img = self.base_blip

    def load_user(self):
        UPLOADS.mkdir(exist_ok = True)

        if USER_META.exists():
            user_meta = json.load(open(USER_META, encoding = "utf-8"))
            u_clip = np.load(USER_CLIP).astype("float32")
            u_blip = np.load(USER_BLIP).astype("float32")

            self.meta += user_meta

            self.clip_img = np.vstack([self.base_clip, u_clip])
            self.blip_img = np.vstack([self.base_blip, u_blip])
            self.user_clip = u_clip
            self.user_blip = u_blip

        else:
            dim = self.base_clip.shape[1]
            self.user_clip = np.zeros((0, dim),
                                       dtype = "float32")
            
            self.user_blip = np.zeros((0, dim),
                                       dtype = "float32")


    def find_bad_captions(self):
        row_by_id = {}
        for i, image_id in enumerate(self.img_ids):
            row_by_id[image_id] = i
        

        sim_clip = []
        for i, image_id in enumerate(self.txt_ids):
            row = row_by_id[image_id]
            
            s = np.sum(self.txt_clip[i] * self.base_clip[row])
            sim_clip.append(s)
        
        sim_clip = np.array(sim_clip)


        sim_blip = []
        for i, img_id in enumerate(self.txt_ids):
            row = row_by_id[img_id]
            
            s = np.sum(self.txt_blip[i] * self.base_blip[row])
            sim_blip.append(s)
        
        sim_blip = np.array(sim_blip)


        def pct(x):
            order = x.argsort()
            r = np.empty(len(x), dtype="float64")
            r[order] = np.arange(len(x))
            return (r + 1) / len(x)

        suspicion = np.maximum(pct(sim_clip), pct(sim_blip))

        
        best = {}
        for i, image_id in enumerate(self.txt_ids):

            if img_id not in best:
                best[img_id] = suspicion[i]

            else:
                if suspicion[i] > best[img_id]:
                    best[img_id] = suspicion[i]

        self.bad_rows = set()
        for row, image_id in enumerate(self.img_ids):
            if img_id in best:
                if best[img_id] <= CLEAN_Q:
                    self.bad_rows.add(row)
       

    def build_faiss(self):
        dim_clip = self.clip_img.shape[1]
        self.clip_index = faiss.IndexFlatIP(dim_clip)
        self.clip_index.add(self.clip_img)

        dim_blip = self.blip_img.shape[1]
        self.blip_index = faiss.IndexFlatIP(dim_blip)
        self.blip_index.add(self.blip_img)

    def faiss_search(self, index, vec):
        n = index.ntotal
        distances, indices = index.search(vec.reshape(1, -1).astype("float32"), n)

        result = np.empty(n, dtype="float32")
        result[indices[0]] = distances[0]
        return result

    def clip_image_vec(self, image):

        inputs = self.clip_proc(images = image, 
                                return_tensors = "pt").to(device)
        
        with torch.no_grad():
            pooled = self.clip.vision_model(pixel_values = inputs["pixel_values"]).pooler_output
            emb = self.clip.visual_projection(pooled)
        
        vector = norm(emb[0].cpu().numpy())
        return vector

    def clip_text_vec(self, text):

        inputs = self.clip_proc(text = [text],return_tensors = "pt",padding = True,  truncation = True).to(device)
        
        with torch.no_grad():
            pooled = self.clip.text_model(**inputs).pooler_output
            emb = self.clip.text_projection(pooled)
        
        vector = norm(emb[0].cpu().numpy())
        return vector


    def blip_image_vec(self, image):

        inputs = self.blip_proc(images = image,return_tensors = "pt").to(device)
        
        with torch.no_grad():
            out = self.blip.vision_model(pixel_value = inputs["pixel_values"])[0][:, 0, :]
            emb = self.blip.vision_proj(out)

        vector = norm(emb[0].cpu().numpy())
        return vector





    def blip_text_vec(self, text):

        inputs = self.blip_proc(text = text,return_tensors = "pt",   padding = True, truncation = True).to(device)
        
        with torch.no_grad():
            out = self.blip.text_encoder(input_ids = inputs["input_ids"], 
                                         attention_mask=inputs["attention_mask"])[0][:, 0, :]
            emb = self.blip.text_proj(out)

        vector = norm(emb[0].cpu().numpy())
        return vector



    def translate(self, text):
        inp = self.mt_tok([text], return_tensors = "pt", padding = True, truncation = True,   max_length = 128)
        
        with torch.no_grad():
            gen = self.mt.generate(**inp, max_length = 128)

        result = self.mt_tok.batch_decode(gen, skip_special_tokens = True)
        return result[0]


    def scores(self, model, clip_vec, blip_vec, weight):
        
        if model == "clip":
            result = self.faiss_search(self.clip_index, clip_vec)
            return result

        if model == "blip":
            result = self.faiss_search(self.blip_index, blip_vec)
            return result



        s_clip = self.faiss_search(self.clip_index, clip_vec)

        s_blip = self.faiss_search(self.blip_index, blip_vec)

        return (1 - weight) * s_clip + weight * s_blip

    def top(self, scores, k, skip = (), clean = False):
        rows = np.argsort(-scores)
        out = []

        for row in rows:
            row = int(row)
            if row in skip:
                continue
            if clean and row in self.bad_rows:
                continue

            out.append({
                "row": row,
                "score": round(float(scores[row]), 3),

                "captions": self.meta[row]["captions"][:5],
                "source": self.meta[row]["source"],
            })

            if len(out) >= k:
                break

        return out






    def search_text(self, text, lang, model, weight, clean, k = 12):
        translated = None
        if lang == "ru" or self.has_cyrillic(text):
            translated = self.translate(text)
            text = translated

        cv = None
        bv = None

        if model != "blip":
            cv = self.clip_text_vec(text)
            
        if model != "clip":
            bv = self.blip_text_vec(text)

        sc = self.scores(model, cv, bv, weight)

        return {"results": self.top(sc, k, clean = clean), 
                "translated": translated}

    def search_image(self, image, model, weight, clean, k = 12):
        cv = None
        bv = None
        if model != "blip":
            cv = self.clip_image_vec(image)
        if model != "clip":
            bv = self.blip_image_vec(image)

        sc = self.scores(model, cv, bv, weight)

        return {"results": self.top(sc, k, clean = clean)}

    def search_multi(self, images, model, weight, k=12):
        cvs = []
        bvs = []

        for im in images:
            cvs.append(self.clip_image_vec(im))
            bvs.append(self.blip_image_vec(im))

        if len(images) == 2:
            cv = norm((1 - weight) * cvs[0] + weight * cvs[1])
            bv = norm((1 - weight) * bvs[0] + weight * bvs[1])

        else:
            cv = norm(np.mean(cvs, axis = 0))
            bv = norm(np.mean(bvs, axis = 0))

        sc = self.scores(model, cv, bv, weight if model == "ensemble" else 0.75)
        return {"results": self.top(sc, k)}

    def add_image(self, image, captions):
        UPLOADS.mkdir(exist_ok=True)
        n = len(self.user_clip)
        name = "user_%d.jpg" % n
        image.save(UPLOADS / name)

        cv = self.clip_image_vec(image).astype("float32")
        bv = self.blip_image_vec(image).astype("float32")

        self.user_clip = np.vstack([self.user_clip, cv])
        self.user_blip = np.vstack([self.user_blip, bv])
        self.clip_img = np.vstack([self.base_clip, self.user_clip])
        self.blip_img = np.vstack([self.base_blip, self.user_blip])
        self.meta.append({"path": str(UPLOADS / name), "captions": captions, "source": "user"})
        self.build_faiss()

        np.save(USER_CLIP, self.user_clip)
        np.save(USER_BLIP, self.user_blip)
        user_meta = [m for m in self.meta if m["source"] == "user"]
        json.dump(user_meta, open(USER_META, "w", encoding="utf-8"), ensure_ascii=False)
        return len(self.meta) - 1

    def has_cyrillic(self, text):
        return any("а" <= c.lower() <= "я" or c == "ё" for c in text)
