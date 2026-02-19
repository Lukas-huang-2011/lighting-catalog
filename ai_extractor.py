"""
AI extraction using Google Gemini REST API directly.
Auto-detects the correct model name and API version.
Free tier: 1,500 requests/day.
"""

import json
import base64
import io
import requests
from PIL import Image
import streamlit as st

# Candidates to try in order — first working one is used
CANDIDATES = [
    ("v1beta", "gemini-1.5-flash"),
    ("v1beta", "gemini-1.5-flash-latest"),
    ("v1",     "gemini-1.5-flash"),
    ("v1beta", "gemini-1.5-pro"),
    ("v1beta", "gemini-pro-vision"),
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
    if "```" in content:
        for part in content.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                r = json.loads(part)
                if isinstance(r, list):
                    return r
            except Exception:
                continue
    try:
        r = json.loads(content)
        if isinstance(r, list):
            return r
    except Exception:
        pass
    s, e = content.find("["), content.rfind("]")
    if s != -1 and e != -1:
        try:
            r = json.loads(content[s:e+1])
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
        "generationConfig": {"maxOutputTokens": 4000, "temperature": 0.1}
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


PROMPT = """You are reading a page from a lighting product catalog or price list.

Extract ONE entry for each product CODE (article number) visible on this page.
Each row in the price table = one code with its own color and price.

RULES:
- One JSON object per code/row
- Prices are plain numbers, no currency symbol (e.g. 3120,00 or 3120.00)
- Find the currency label in column HEADER (e.g. "RMBexcl. VAT" means currency="RMB")
- Convert comma decimals to dots: 3120,00 → 3120.00
- Include accessories with codes and prices
- If no product codes on this page (index, cover), return []

Fields per entry (null if not found):
- codes: ["ONE_CODE"]
- name: product family name (e.g. "CABRIOLETTE Body lamp")
- description: emission type, light source description
- color: color name + RAL (e.g. "White Polished RAL 9003")
- light_source: full text (e.g. "7.5W 1110lm Integrated LED")
- cct: color temperature (e.g. "2700K")
- dimensions: size string (e.g. "Ø15.5 H28 cm")
- wattage: watts only (e.g. "7.5W")
- price: number dot decimal (e.g. 3120.00)
- currency: from column header (e.g. "RMB", "EUR", "USD")
- extra_fields: {ip_rating, dimming, voltage, driver, structure, diffuser, net_weight, gross_weight, package_dimension}

Return ONLY a valid JSON array. No explanation, no markdown."""


def extract_products_from_page(api_key: str, page_image: Image.Image, page_num: int) -> list:
    text, error, _, _ = _call_best(api_key, page_image, PROMPT)
    if error:
        print(f"Page {page_num}: {error}")
        return []
    return _parse_json(text)


def extract_products_debug(api_key: str, page_image: Image.Image) -> dict:
    out = {"model": "", "raw_response": "", "parsed": [], "error": ""}
    text, error, version, model = _call_best(api_key, page_image, PROMPT)
    out["model"] = f"{version}/{model}"
    out["raw_response"] = text
    out["error"] = error
    if text:
        out["parsed"] = _parse_json(text)
    return out


def describe_image(api_key: str, image: Image.Image) -> str:
    prompt = "Describe this lighting product briefly: type, shape, color, style, any visible codes."
    text, _, _, _ = _call_best(api_key, image, prompt)
    return text
