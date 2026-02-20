"""
AI extraction using GLM-4V-Flash via Zhipu AI (free vision model).
Get a free API key at: https://bigmodel.cn → 开放平台 → API Keys
OpenAI-compatible endpoint.
"""

import json
import base64
import io
import time
import requests
from PIL import Image
import streamlit as st

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# Models to try in order — glm-4v-flash is free
CANDIDATES = [
    "glm-4v-flash",
    "glm-4v-plus",
    "glm-4v",
]

_working_model = {"model": None}


def get_client():
    api_key = st.secrets.get("ZHIPU_API_KEY", "")
    if not api_key:
        st.error(
            "ZHIPU_API_KEY not found in Streamlit secrets. "
            "Get a free key at bigmodel.cn → API Keys."
        )
        st.stop()
    return api_key


def image_to_base64(image: Image.Image) -> tuple:
    """Returns (base64_string, mime_type). Resizes and compresses to JPEG for Zhipu AI.
    GLM-4V-Flash works best with images under 1MB. Keep max side at 1024px, quality 75.
    """
    img = image.convert("RGB")
    max_side = 1024
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode()
    kb = len(buf.getvalue()) / 1024
    # If still too large, compress further
    if kb > 800:
        buf2 = io.BytesIO()
        img.save(buf2, format="JPEG", quality=50)
        b64 = base64.b64encode(buf2.getvalue()).decode()
    return b64, "image/jpeg"


def _parse_json(content: str) -> list:
    content = content.strip()

    # Strip markdown code fences
    if "```" in content:
        for part in content.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                r = json.loads(part)
                if isinstance(r, list):
                    return r
            except Exception:
                continue

    # Direct parse
    try:
        r = json.loads(content)
        if isinstance(r, list):
            return r
    except Exception:
        pass

    # Try between first [ and last ]
    s, e = content.find("["), content.rfind("]")
    if s != -1 and e != -1:
        try:
            r = json.loads(content[s:e + 1])
            if isinstance(r, list):
                return r
        except Exception:
            pass

    # Truncated JSON — find last complete object and close the array
    s = content.find("[")
    if s != -1:
        last_close = content.rfind("}")
        if last_close != -1:
            try:
                r = json.loads(content[s:last_close + 1] + "]")
                if isinstance(r, list):
                    return r
            except Exception:
                pass

    return []


def _call(api_key: str, model: str, image: Image.Image, prompt: str) -> tuple:
    """Returns (response_text, error_string). Retries on 429 with backoff."""
    img_b64, mime = image_to_base64(image)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 2048,
        "temperature": 0.1,
        "stream": False,
    }
    for attempt in range(4):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=90)
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                return text, ""
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)   # 15s → 30s → 45s
                time.sleep(wait)
                continue
            return "", f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as ex:
            return "", str(ex)
    return "", "Rate limited after retries"


def _call_best(api_key: str, image: Image.Image, prompt: str) -> tuple:
    """Try model candidates, return (text, error, model_used)."""
    if _working_model["model"]:
        text, err = _call(api_key, _working_model["model"], image, prompt)
        if not err:
            return text, "", _working_model["model"]

    for model in CANDIDATES:
        text, err = _call(api_key, model, image, prompt)
        if not err:
            _working_model["model"] = model
            return text, "", model

    return "", f"All models failed. Last error: {err}", ""


PROMPT = """You are reading a portion of a lighting product catalog or price list.

YOUR TASK: Extract EVERY row that has a product code (article number) visible in this image.
Scan from top to bottom — do NOT stop early.

WHAT TO EXTRACT:
1. MAIN product rows — each color/variant row under a product family header
2. ACCESSORY rows — rows under "Accessories" or "Accessori" sections (they have own codes + prices)
3. ALL product families visible — if multiple products are shown, extract ALL of them
4. Every single row with a code = one JSON entry, no exceptions

RULES:
- One JSON object per code/row
- Prices are plain numbers (e.g. 3120.00) — convert comma decimals: 3120,00 → 3120.00
- Find currency from the column HEADER (e.g. "RMBexcl. VAT" → "RMB")
- OMIT any field that has no value — do NOT write null, just skip the key entirely
- Keep field values short and factual
- If no product codes visible (cover, index, pure text page), return []

Fields to include (only when value exists):
- codes: ["CODE"] — required
- name: product family name or accessory description — required
- color: color name (e.g. "Arancio", "Bianco")
- light_source: e.g. "7.5W 1110lm Integrated LED"
- cct: e.g. "2700K"
- dimensions: e.g. "Ø15.5 H28 cm"
- wattage: e.g. "7.5W"
- price: number (e.g. 3120.00) — required if visible
- currency: e.g. "RMB" — required if visible
- description: short description, mounting type, or accessory use
- extra_fields: object with any of {ip_rating, dimming, voltage, driver, structure, diffuser, net_weight}

Return ONLY a valid JSON array. No explanation. No markdown. Include EVERY code row."""


def _split_image(image: Image.Image):
    """Split a page into 4 sections with overlap so no row gets cut off."""
    w, h = image.size
    overlap = int(h * 0.08)
    q = h // 4
    s1 = image.crop((0, 0,               w, q + overlap))
    s2 = image.crop((0, q - overlap,     w, q * 2 + overlap))
    s3 = image.crop((0, q * 2 - overlap, w, q * 3 + overlap))
    s4 = image.crop((0, q * 3 - overlap, w, h))
    return s1, s2, s3, s4


def _extract_section(api_key: str, section_image: Image.Image) -> list:
    text, error, _ = _call_best(api_key, section_image, PROMPT)
    if error:
        return []
    return _parse_json(text)


def _dedup(products: list) -> list:
    seen = set()
    result = []
    for p in products:
        key = str(p.get("codes", ""))
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def extract_products_from_page(api_key: str, page_image: Image.Image, page_num: int) -> list:
    """
    Extract all products from a page.
    Splits into 4 sections; waits 3s between calls to stay within 20 RPM free tier.
    """
    s1, s2, s3, s4 = _split_image(page_image)
    results = []
    for section in (s1, s2, s3, s4):
        results.extend(_extract_section(api_key, section))
        time.sleep(3)   # 3s gap → ~20 calls/min, within SiliconFlow free tier
    return _dedup(results)


def extract_products_debug(api_key: str, page_image: Image.Image) -> dict:
    """Debug version — shows raw responses from all 4 sections."""
    s1, s2, s3, s4 = _split_image(page_image)

    t1, e1, m1 = _call_best(api_key, s1, PROMPT)
    time.sleep(3)
    t2, e2, _  = _call_best(api_key, s2, PROMPT)
    time.sleep(3)
    t3, e3, _  = _call_best(api_key, s3, PROMPT)
    time.sleep(3)
    t4, e4, _  = _call_best(api_key, s4, PROMPT)

    p1 = _parse_json(t1) if t1 else []
    p2 = _parse_json(t2) if t2 else []
    p3 = _parse_json(t3) if t3 else []
    p4 = _parse_json(t4) if t4 else []
    all_products = _dedup(p1 + p2 + p3 + p4)

    raw = (
        f"=== SECTION 1 ({len(p1)} products) ===\n{t1}\n\n"
        f"=== SECTION 2 ({len(p2)} products) ===\n{t2}\n\n"
        f"=== SECTION 3 ({len(p3)} products) ===\n{t3}\n\n"
        f"=== SECTION 4 ({len(p4)} products) ===\n{t4}"
    )

    return {
        "model": m1 or "unknown",
        "raw_response": raw,
        "parsed": all_products,
        "error": e1 or e2 or e3 or e4,
    }


def describe_image(api_key: str, image: Image.Image) -> str:
    prompt = "Describe this lighting product briefly: type, shape, color, style, any visible codes."
    text, _, _ = _call_best(api_key, image, prompt)
    return text
