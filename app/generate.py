import io
import os
import time
import base64

import requests
from PIL import Image


def pil_to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else "image/png"
    return base64.b64encode(buf.getvalue()).decode(), mime


Y_API_KEY = os.environ.get("YANDEX_API_KEY", "")
Y_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
Y_GENERATE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGenerationAsync"
Y_OPERATION_URL = "https://llm.api.cloud.yandex.net/operations/"


def generate_yandex(prompt, aspect_ratio=(1, 1), seed=None,
                    poll_interval=2.0, timeout=90.0):
    headers = {
        "Authorization": f"Api-Key {Y_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "modelUri": f"art://{Y_FOLDER_ID}/yandex-art/latest",
        "generationOptions": {
            "aspectRatio": {"widthRatio": str(aspect_ratio[0]),
                            "heightRatio": str(aspect_ratio[1])},
        },
        "messages": [{"weight": 1, "text": prompt}],
    }
    if seed is not None:
        body["generationOptions"]["seed"] = str(seed)

    resp = requests.post(Y_GENERATE_URL, headers=headers, json=body, timeout=15)
    resp.raise_for_status()
    operation_id = resp.json()["id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        op = requests.get(Y_OPERATION_URL + operation_id, headers=headers, timeout=15)
        op.raise_for_status()
        data = op.json()
        if data.get("done"):
            if "error" in data:
                raise RuntimeError(f"YandexART: {data['error']}")
            return data["response"]["image"], "image/jpeg"
        time.sleep(poll_interval)

    raise TimeoutError("превышено время ожидания генерации")


_sd_pipe = None


def _ensure_sd():
    raise NotImplementedError(
        "Локальная генерация не подключена: заполни _ensure_sd() и "
        "generate_local() в app/generate.py своим Stable Diffusion инференсом."
    )


def generate_local(prompt, aspect_ratio=(1, 1), seed=None):
    raise NotImplementedError(
        "Локальная генерация не подключена: заполни generate_local() в "
        "app/generate.py."
    )


def _ratio_to_wh(aspect_ratio, base=512):
    aw, ah = aspect_ratio
    if aw >= ah:
        w, h = base, round(base * ah / aw)
    else:
        w, h = round(base * aw / ah), base
    w -= w % 8
    h -= h % 8
    return max(w, 8), max(h, 8)


PROVIDERS = {"yandex", "local"}


def generate(prompt, provider="yandex", aspect_ratio=(1, 1), seed=None):
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("пустой запрос")
    if provider not in PROVIDERS:
        raise ValueError(f"неизвестный провайдер: {provider}")

    if provider == "yandex":
        return generate_yandex(prompt, aspect_ratio=aspect_ratio, seed=seed)

    img = generate_local(prompt, aspect_ratio=aspect_ratio, seed=seed)
    return pil_to_b64(img)
