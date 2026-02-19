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
                # Try to close the array after the last complete object
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


PROMPT = """You are reading a page from a lighting product catalog or price list.

YOUR TASK: Extract EVERY row that has a product code (article number) on this page.
Scan the ENTIRE page from top to bottom — do NOT stop after the first product group.

WHAT TO EXTRACT:
1. MAIN product rows — each color/variant row under a product family header
2. ACCESSORY rows — rows listed under "Accessories" or "Accessori" sections (they have their own codes and prices)
3. MULTIPLE product families — if the page has e.g. "AVRO Studio Natural" AND "AVRO Junior", extract ALL rows from BOTH families
4. Every single row with a code = one JSON entry

RULES:
- One JSON object per code/row — never skip a row that has a code
- Prices are plain numbers, no currency symbol (e.g. 3120,00 or 3120.00)
- Find the currency label in the column HEADER (e.g. "RMBexcl. VAT" → currency="RMB")
- Convert comma decimals to dots: 3120,00 → 3120.00
- For accessory rows: set name = accessory description (e.g. "Canopy", "Suspension kit"), color = null unless listed
- For accessory rows: inherit currency from the same page header
- If a product family has 8 color variants AND 6 accessory rows = 14 total entries for that family
- If the page has 2 product families each with variants + accessories = extract ALL of them
- If no product codes on this page (index page, cover, intro text), return []

Fields per entry (null if not found):
- codes: ["ONE_CODE"]
- name: product family name (e.g. "AVRO Studio Natural") OR accessory name (e.g. "Canopy white")
- description: emission type, mounting type, or accessory description
- color: color name (e.g. "Arancio", "Bianco") — null for most accessories
- light_source: full text (e.g. "7.5W 1110lm Integrated LED") — null for accessories
- cct: color temperature (e.g. "2700K", "3000K") — null for accessories
- dimensions: size string (e.g. "Ø15.5 H28 cm")
- wattage: watts only (e.g. "7.5W") — null for accessories
- price: number dot decimal (e.g. 3120.00)
- currency: from column header (e.g. "RMB", "EUR", "USD")
- extra_fields: {ip_rating, dimming, voltage, driver, structure, diffuser, net_weight, gross_weight, package_dimension}

Return ONLY a valid JSON array. No explanation, no markdown. Include EVERY code row."""


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
