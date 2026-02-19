"""
AI extraction using Google Gemini REST API directly.
Auto-detects the correct model name and API version.
Free tier: 1,500 requests/day.
Pages with many products are split into halves to avoid token limits.
"""

import json
import base64
import io
import requests
from PIL import Image
import streamlit as st

# Candidates to try in order — first working one is used
CANDIDATES = [
    ("v1", "gemini-2.0-flash"),
    ("v1", "gemini-2.5-flash"),
    ("v1", "gemini-2.0-flash-001"),
    ("v1", "gemini-2.0-flash-lite"),
    ("v1", "gemini-2.5-flash-lite"),
]

# Cache which combo works so we don't retry every page
_working = {"version": None, "model": None}


def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Get one free at aistudio.google.com")
        st.stop()
    return api_key


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_json(content: str) -> list:
    content = content.strip()

    # Extract from markdown code fences if present
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
            r = json.loads(content[s:e+1])
            if isinstance(r, list):
                return r
        except Exception:
            pass

    # Handle TRUNCATED JSON — find last complete object and close the array
    s = content.find("[")
    if s != -1:
        last_close = content.rfind("}")
        if last_close != -1:
            try:
                truncated = content[s:last_close+1] + "]"
                r = json.loads(truncated)
                if isinstance(r, list):
                    return r
            except Exception:
                pass

    return []


def _call(api_key: str, version: str, model: str, image: Image.Image, prompt: str) -> tuple:
    """Returns (response_text, error_string)."""
    url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={api_key}"
    img_b64 = image_to_base64(image)
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": img_b64}}
        ]}],
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.1}
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return text, ""
        return "", f"HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return "", str(e)


def _call_best(api_key: str, image: Image.Image, prompt: str) -> tuple:
    """Try candidates, return (text, error, version_used, model_used)."""
    # Use cached working combo first
    if _working["version"]:
        text, err = _call(api_key, _working["version"], _working["model"], image, prompt)
        if not err:
            return text, "", _working["version"], _working["model"]

    # Try all candidates
    for version, model in CANDIDATES:
        text, err = _call(api_key, version, model, image, prompt)
        if not err:
            _working["version"] = version
            _working["model"] = model
            return text, "", version, model

    return "", f"All models failed. Last error: {err}", "", ""


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
    """Split a page image into top and bottom halves with a small overlap."""
    w, h = image.size
    # 10% overlap at the split point so no row gets cut off
    overlap = int(h * 0.10)
    top = image.crop((0, 0, w, h // 2 + overlap))
    bottom = image.crop((0, h // 2 - overlap, w, h))
    return top, bottom


def _extract_half(api_key: str, half_image: Image.Image) -> list:
    """Extract products from one image (half-page or full page)."""
    text, error, _, _ = _call_best(api_key, half_image, PROMPT)
    if error:
        return []
    return _parse_json(text)


def _dedup(products: list) -> list:
    """Remove duplicate entries by code (keeps first occurrence)."""
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
    Splits into top/bottom halves to handle pages with many products
    that would otherwise exceed the AI token limit.
    """
    top, bottom = _split_image(page_image)
    top_products = _extract_half(api_key, top)
    bottom_products = _extract_half(api_key, bottom)
    all_products = _dedup(top_products + bottom_products)
    return all_products


def extract_products_debug(api_key: str, page_image: Image.Image) -> dict:
    """Debug version — shows raw responses from both halves."""
    top, bottom = _split_image(page_image)

    top_text, top_err, version, model = _call_best(api_key, top, PROMPT)
    bottom_text, bottom_err, _, _ = _call_best(api_key, bottom, PROMPT)

    top_parsed = _parse_json(top_text) if top_text else []
    bottom_parsed = _parse_json(bottom_text) if bottom_text else []
    all_products = _dedup(top_parsed + bottom_parsed)

    raw = f"=== TOP HALF ({len(top_parsed)} products) ===\n{top_text}\n\n=== BOTTOM HALF ({len(bottom_parsed)} products) ===\n{bottom_text}"

    return {
        "model": f"{version}/{model}",
        "raw_response": raw,
        "parsed": all_products,
        "error": top_err or bottom_err,
    }


def describe_image(api_key: str, image: Image.Image) -> str:
    prompt = "Describe this lighting product briefly: type, shape, color, style, any visible codes."
    text, _, _, _ = _call_best(api_key, image, prompt)
    return text
