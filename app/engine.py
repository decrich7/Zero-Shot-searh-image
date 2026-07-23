import json
import re
import unicodedata
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers import BlipProcessor, BlipForImageTextRetrieval
from transformers import BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory, LangDetectException
import requests
from app.generate import Y_API_KEY, Y_FOLDER_ID

DetectorFactory.seed = 0

torch.set_num_threads(4)
device = "cpu"

_LEAD = re.compile(
    r"^(there (is|are)|this is (a picture|an image|a photo) of"
    r"|(a|an) (picture|image|photo) of)\s+", re.I)
_ARTIFACT = re.compile(r"\b(arafe?d|araffes?)\b", re.I)


def clean_caption(text):
    t = _ARTIFACT.sub("", text.strip())
    prev = None
    while prev != t:
        prev = t
        t = _LEAD.sub("", t).strip()
    t = re.sub(r"\s{2,}", " ", t).strip(" .,")
    if not t:
        return text.strip()
    t = t[0].upper() + t[1:]
    if not t.endswith((".", "!", "?")):
        t += "."
    return t


_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F000-\U0001F0FF"
    "\U00002700-\U000027BF"
    "\U0000200D"
    "\U000024C2"
    "]+", flags = re.UNICODE)

_INVISIBLE = re.compile(r"[​‌‎‏‪-‮﻿]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION = re.compile(r"(?<!\w)@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_WS = re.compile(r"\s+")


def preprocess_query(text):
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r"\1", t)
    t = _EMOJI.sub(" ", t)
    t = _INVISIBLE.sub("", t)
    t = _CTRL.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if not re.search(r"\w", t):
        return ""
    return t


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DATA"
IMAGES = DATA / "val2017"
TRAIN_IMAGES = DATA / "coco_balanced_15k" / "images"
UPLOADS = DATA / "user_uploads"
MODELS = ROOT / "MODELS"
EMB = ROOT / "embeddings_output_20k"

USER_BLIP = EMB / "user_blip_img.npy"
USER_CLIP = EMB / "user_clip_img.npy"
USER_BLIP2 = EMB / "user_blip2_img.npy"
USER_META = EMB / "user_meta.json"

CLEAN_Q = 0.10

W3 = (0.10, 0.35, 0.55)

GEN_MARGIN_THRESHOLD = 0.22

YANDEX_TRANSLATE_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"

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

        self.cap_model = None
        self.cap_proc = None

        self.blip2 = None
        self.blip2_proc = None

        self.lock = threading.Lock()

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

        val_files = {p.name for p in IMAGES.iterdir()} if IMAGES.exists() else set()

        self.meta = []
        for image_id in order:
            caps_ru = ru_by_id.get(image_id) or en_by_id.get(image_id, [])

            fname = file_by_id[image_id]
            folder = IMAGES if fname in val_files else TRAIN_IMAGES

            self.meta.append({
                "path": str(folder / fname),
                "captions": caps_ru,
                "source": "base",
            })

        self.base_clip = np.load(EMB / "clip_image_embeddings.npy").astype("float32")
        self.base_blip = np.load(EMB / "blip_image_embeddings.npy").astype("float32")

        self.txt_clip = np.load(EMB / "clip_text_embeddings.npy").astype("float32")
        self.txt_blip = np.load(EMB / "blip_text_embeddings.npy").astype("float32")
        self.txt_ids = pd.read_csv(EMB / "caption_ids_order.csv")["image_id"].tolist()

        self.img_ids = order

        b2 = np.load(EMB / "image_embeds_BLIP-2.npy").astype("float32")
        sorted_ids = sorted(order)
        pos = {v: k for k, v in enumerate(sorted_ids)}
        perm = np.array([pos[i] for i in order])
        b2 = b2[perm]
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        self.base_blip2 = b2

        self.clip_img = self.base_clip
        self.blip_img = self.base_blip
        self.blip2_img = self.base_blip2

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

            if USER_BLIP2.exists():
                u_blip2 = np.load(USER_BLIP2).astype("float32")
            else:
                u_blip2 = np.zeros((0, 32, 256), dtype = "float32")
            if len(u_blip2) != len(user_meta):
                u_blip2 = self._backfill_user_blip2(user_meta)
            self.user_blip2 = u_blip2
            self.blip2_img = np.vstack([self.base_blip2, u_blip2])

        else:
            dim = self.base_clip.shape[1]
            self.user_clip = np.zeros((0, dim),
                                       dtype = "float32")

            self.user_blip = np.zeros((0, dim),
                                       dtype = "float32")
            self.user_blip2 = np.zeros((0, 32, 256), dtype = "float32")

    def _backfill_user_blip2(self, user_meta):
        vecs = []
        for m in user_meta:
            image = Image.open(m["path"]).convert("RGB")
            vecs.append(self.blip2_image_vec(image))
        arr = np.stack(vecs).astype("float32") if vecs else np.zeros((0, 32, 256), dtype = "float32")
        np.save(USER_BLIP2, arr)
        return arr


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

            if image_id not in best:
                best[image_id] = suspicion[i]

            else:
                if suspicion[i] > best[image_id]:
                    best[image_id] = suspicion[i]

        self.bad_rows = set()
        for row, image_id in enumerate(self.img_ids):
            if image_id in best:
                if best[image_id] <= CLEAN_Q:
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
            out = self.blip.vision_model(pixel_values = inputs["pixel_values"])[0][:, 0, :]
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


    def _ensure_blip2(self):
        if self.blip2 is None:
            from transformers import Blip2ForImageTextRetrieval, Blip2Processor
            p = MODELS / "blip-2"
            self.blip2 = Blip2ForImageTextRetrieval.from_pretrained(
                p, dtype = torch.float32).to(device).eval()
            self.blip2_proc = Blip2Processor.from_pretrained(p)

    def blip2_text_vec(self, text):
        self._ensure_blip2()
        t = self.blip2_proc(text = text, return_tensors = "pt",
                            padding = True, truncation = True).to(device)
        with torch.no_grad():
            qe = self.blip2.embeddings(input_ids = t["input_ids"])
            to = self.blip2.qformer(query_embeds = qe, query_length = 0,
                                    attention_mask = t["attention_mask"], return_dict = True)
            emb = torch.nn.functional.normalize(
                self.blip2.text_projection(to.last_hidden_state[:, 0, :]), dim = -1)
        return emb[0].cpu().numpy().astype("float32")

    def blip2_image_vec(self, image):
        self._ensure_blip2()
        pv = self.blip2_proc(images = image, return_tensors = "pt").to(device)["pixel_values"]
        with torch.no_grad():
            vo = self.blip2.vision_model(pixel_values = pv)
            ie = vo[0]
            am = torch.ones(ie.size()[:-1], dtype = torch.long, device = device)
            qt = self.blip2.query_tokens.expand(ie.shape[0], -1, -1)
            qo = self.blip2.qformer(query_embeds = qt, encoder_hidden_states = ie,
                                    encoder_attention_mask = am, return_dict = True)
            emb = torch.nn.functional.normalize(
                self.blip2.vision_projection(qo.last_hidden_state), dim = -1)
        return emb[0].cpu().numpy().astype("float32")

    def b2_text_scores(self, b2t):
        sims = self.blip2_img @ b2t
        return sims.max(axis = 1).astype("float32")

    def scores_triple(self, cv, bv, b2t):
        s_clip = self.faiss_search(self.clip_index, cv)
        s_blip = self.faiss_search(self.blip_index, bv)
        s_b2 = self.b2_text_scores(b2t)
        return W3[0] * s_clip + W3[1] * s_blip + W3[2] * s_b2


    def _ensure_caption_model(self):
        if self.cap_model is None:
            cap_path = MODELS / "blip-caption"
            self.cap_proc = BlipProcessor.from_pretrained(cap_path)
            self.cap_model = BlipForConditionalGeneration.from_pretrained(
                cap_path).to(device).eval()

    def generate_caption(self, image, num_beams = 1, max_new_tokens = 30):
        self._ensure_caption_model()

        inputs = self.cap_proc(images = image, return_tensors = "pt").to(device)

        with torch.no_grad():
            out = self.cap_model.generate(**inputs,
                                          num_beams = num_beams,
                                          max_new_tokens = max_new_tokens)

        raw = self.cap_proc.decode(out[0], skip_special_tokens = True)
        return clean_caption(raw)


    def translate(self, text):
        inp = self.mt_tok([text], return_tensors = "pt", padding = True, truncation = True,   max_length = 128)

        with torch.no_grad():
            gen = self.mt.generate(**inp, max_length = 128)

        result = self.mt_tok.batch_decode(gen, skip_special_tokens = True)
        return result[0]


    def yandex_translate(self, text, target = "en"):
        headers = {
            "Authorization": f"Api-Key {Y_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "folderId": Y_FOLDER_ID,
            "targetLanguageCode": target,
            "texts": [text],
        }
        r = requests.post(YANDEX_TRANSLATE_URL, headers = headers,
                          json = body, timeout = 5)
        r.raise_for_status()
        return r.json()["translations"][0]["text"]


    def detect_lang(self, text, hint = "auto"):
        if hint and hint != "auto":
            return hint
        text = (text or "").strip()
        if not text:
            return "en"
        try:
            return detect(text)
        except LangDetectException:
            return "ru" if self.has_cyrillic(text) else "en"


    def to_english(self, text, lang = "auto"):
        text = preprocess_query((text or "").strip())
        if not text:
            return {"en": "", "detected": None, "via": "none"}

        detected = self.detect_lang(text, lang)

        if detected == "en":
            return {"en": text, "detected": "en", "via": "none"}

        if detected == "ru":
            return {"en": self.translate(text), "detected": "ru", "via": "model"}

        try:
            en = self.yandex_translate(text, "en")
            if en:
                return {"en": en, "detected": detected, "via": "api"}
        except Exception:
            pass

        return {"en": text, "detected": detected, "via": "none"}


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

    def top(self, scores, k, skip = (), clean = False, scope = "all"):
        rows = np.argsort(-scores)
        out = []

        for row in rows:
            row = int(row)
            if row in skip:
                continue
            if clean and row in self.bad_rows:
                continue
            if scope == "user" and self.meta[row]["source"] != "user":
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


    def search_text(self, text, lang, model, weight, clean, precise = False, k = 12):
        tr = self.to_english(text, lang)
        text = tr["en"]
        translated = tr["en"] if tr["via"] != "none" else None
        detected, via = tr["detected"], tr["via"]

        if model == "blip2":
            b2t = self.blip2_text_vec(text)
            with self.lock:
                sc = self.b2_text_scores(b2t)

        elif model == "ensemble3":
            cv = self.clip_text_vec(text)
            bv = self.blip_text_vec(text)
            b2t = self.blip2_text_vec(text)
            with self.lock:
                sc = self.scores_triple(cv, bv, b2t)

        else:
            cv = self.clip_text_vec(text) if model != "blip" else None
            bv = self.blip_text_vec(text) if model != "clip" else None
            with self.lock:
                sc = self.scores(model, cv, bv, weight)

        margin = float(sc.max() - np.median(sc))
        should_generate = bool(margin < GEN_MARGIN_THRESHOLD)

        if precise:
            cand = self.top(sc, 32, clean = clean)
            rows = [c["row"] for c in cand]
            reranked = self.itm_rerank(text, rows)
            results = []
            for it in reranked[:k]:
                row = it["row"]
                results.append({
                    "row": row,
                    "score": round(it["itm"], 3),
                    "captions": self.meta[row]["captions"][:5],
                    "source": self.meta[row]["source"],
                })
            return {"results": results, "translated": translated,
                    "detected": detected, "via": via, "precise": True,
                    "confidence": round(margin, 3),
                    "should_generate": should_generate}

        return {"results": self.top(sc, k, clean = clean),
                "translated": translated, "detected": detected, "via": via,
                "confidence": round(margin, 3),
                "should_generate": should_generate}

    def itm_rerank(self, text, rows, topn = 8):
        rows = rows[:topn]
        scored = []
        for row in rows:
            image = Image.open(self.meta[row]["path"]).convert("RGB")
            inp = self.blip_proc(images = image, text = text,
                                 return_tensors = "pt", truncation = True).to(device)
            with torch.no_grad():
                out = self.blip(input_ids = inp["input_ids"],
                                attention_mask = inp["attention_mask"],
                                pixel_values = inp["pixel_values"],
                                use_itm_head = True)
                prob = torch.softmax(out.itm_score, dim = 1)[0, 1].item()
            scored.append({"row": row, "itm": prob})
        scored.sort(key = lambda x: -x["itm"])
        return scored

    _STOP = {"a", "an", "the", "in", "on", "of", "and", "with", "at", "to",
             "is", "are", "this", "that", "there", "for", "by", "as", "it"}

    @staticmethod
    def _smooth3(g):
        pad = np.pad(g, 1, mode = "edge")
        out = np.zeros_like(g)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out += pad[1 + dy:1 + dy + g.shape[0], 1 + dx:1 + dx + g.shape[1]]
        return out / 9.0

    def attention_map(self, text, row):
        image = Image.open(self.meta[row]["path"]).convert("RGB")
        inp = self.blip_proc(images = image, text = text,
                             return_tensors = "pt", truncation = True).to(device)

        toks = self.blip_proc.tokenizer.convert_ids_to_tokens(inp["input_ids"][0].tolist())
        content = [i for i, t in enumerate(toks)
                   if t not in ("[CLS]", "[SEP]") and t.lower() not in self._STOP]
        if not content:
            nvalid = int(inp["attention_mask"][0].sum())
            content = list(range(1, nvalid - 1))

        with torch.no_grad():
            ie = self.blip.vision_model(pixel_values = inp["pixel_values"]).last_hidden_state
        ia = torch.ones(ie.size()[:-1], dtype = torch.long, device = device)
        to = self.blip.text_encoder(input_ids = inp["input_ids"],
                                    attention_mask = inp["attention_mask"],
                                    encoder_hidden_states = ie,
                                    encoder_attention_mask = ia,
                                    output_attentions = True)
        logits2 = self.blip.itm_head(to.last_hidden_state[:, 0, :])
        prob = torch.softmax(logits2, dim = 1)[0, 1].item()

        atts = to.cross_attentions
        for a in atts:
            a.retain_grad()
        self.blip.zero_grad()
        logits2[0, 1].backward()

        cams = []
        for l in (6, 7, 8, 9):
            A = atts[l][0]
            G = atts[l].grad[0].clamp(min = 0)
            cam = (A * G).mean(0)
            cams.append(cam[content].mean(0)[1:].reshape(24, 24).detach().cpu().numpy())
        g = self._smooth3(np.mean(cams, axis = 0))

        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        lo, hi = np.percentile(g, 55), np.percentile(g, 98)
        g = np.clip((g - lo) / (hi - lo + 1e-8), 0, 1)
        return {"grid": g.tolist(), "size": 24, "prob": round(prob, 3)}

    def search_image(self, image, model, weight, clean, k = 12):
        cv = None
        bv = None
        if model != "blip":
            cv = self.clip_image_vec(image)
        if model != "clip":
            bv = self.blip_image_vec(image)

        sc = self.scores(model, cv, bv, weight)

        return {"results": self.top(sc, k, clean = clean)}

    def search_smart(self, text, lang = "auto", neg_text = None, ref_image = None,
                     ref_weight = 0.5, neg_weight = 0.5, scope = "all",
                     clean = True, k = 12):
        text = (text or "").strip()
        tr = self.to_english(text, lang)
        text = tr["en"]
        translated = tr["en"] if tr["via"] != "none" else None
        detected, via = tr["detected"], tr["via"]

        neg_text = (neg_text or "").strip()
        if neg_text:
            neg_text = self.to_english(neg_text, lang)["en"]

        cv = None
        bv = None
        if text:
            cv = self.clip_text_vec(text)
            bv = self.blip_text_vec(text)

        if ref_image is not None:
            icv = self.clip_image_vec(ref_image)
            ibv = self.blip_image_vec(ref_image)
            if cv is None:
                cv, bv = icv, ibv
            else:
                cv = norm((1 - ref_weight) * cv + ref_weight * icv)
                bv = norm((1 - ref_weight) * bv + ref_weight * ibv)

        if cv is None:
            return {"results": [], "translated": translated,
                    "detected": detected, "via": via}

        if neg_text:
            ncv = self.clip_text_vec(neg_text)
            nbv = self.blip_text_vec(neg_text)
            cv = norm(cv - neg_weight * ncv)
            bv = norm(bv - neg_weight * nbv)

        use_triple = text and ref_image is None
        if use_triple:
            b2t = self.blip2_text_vec(text)
            if neg_text:
                b2t = norm(b2t - neg_weight * self.blip2_text_vec(neg_text))
            with self.lock:
                sc = self.scores_triple(cv, bv, b2t)
        else:
            with self.lock:
                sc = self.scores("ensemble", cv, bv, 0.75)

        return {"results": self.top(sc, k, clean = clean, scope = scope),
                "translated": translated, "detected": detected, "via": via}

    def search_similar(self, row, scope = "all", clean = False, k = 12):
        cv = self.clip_img[row]
        bv = self.blip_img[row]
        with self.lock:
            sc = self.scores("ensemble", cv, bv, 0.75)
        return {"results": self.top(sc, k, skip = {row}, clean = clean, scope = scope)}

    def search_multi(self, images, model, weight, mode = "blend", clean = True, k = 12):
        cvs = []
        bvs = []
        for im in images:
            cvs.append(self.clip_image_vec(im))
            bvs.append(self.blip_image_vec(im))

        if mode == "and":
            with self.lock:
                per = [self.scores("ensemble", cv, bv, 0.75)
                       for cv, bv in zip(cvs, bvs)]
            sc = np.min(np.stack(per), axis = 0)
            return {"results": self.top(sc, k, clean = clean)}

        if len(images) == 2:
            cv = norm((1 - weight) * cvs[0] + weight * cvs[1])
            bv = norm((1 - weight) * bvs[0] + weight * bvs[1])
        else:
            cv = norm(np.mean(cvs, axis = 0))
            bv = norm(np.mean(bvs, axis = 0))

        with self.lock:
            sc = self.scores(model, cv, bv, weight if model == "ensemble" else 0.75)
        return {"results": self.top(sc, k, clean = clean)}

    def add_image(self, image, captions):
        UPLOADS.mkdir(exist_ok=True)
        n = len(self.user_clip)
        name = "user_%d.jpg" % n
        image.save(UPLOADS / name)

        cv = self.clip_image_vec(image).astype("float32")
        bv = self.blip_image_vec(image).astype("float32")
        b2 = self.blip2_image_vec(image).astype("float32")

        with self.lock:
            self.user_clip = np.vstack([self.user_clip, cv])
            self.user_blip = np.vstack([self.user_blip, bv])
            self.user_blip2 = np.vstack([self.user_blip2, b2[None]])
            self.clip_img = np.vstack([self.base_clip, self.user_clip])
            self.blip_img = np.vstack([self.base_blip, self.user_blip])
            self.blip2_img = np.vstack([self.base_blip2, self.user_blip2])
            self.meta.append({"path": str(UPLOADS / name), "captions": captions, "source": "user"})
            self.build_faiss()

            np.save(USER_CLIP, self.user_clip)
            np.save(USER_BLIP, self.user_blip)
            np.save(USER_BLIP2, self.user_blip2)
            user_meta = [m for m in self.meta if m["source"] == "user"]
            json.dump(user_meta, open(USER_META, "w", encoding="utf-8"), ensure_ascii=False)
            return len(self.meta) - 1

    def has_cyrillic(self, text):
        return any("а" <= c.lower() <= "я" or c == "ё" for c in text)
