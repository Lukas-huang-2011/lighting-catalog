"""
AI extraction using Google Gemini REST API directly.
No SDK version issues. Free tier: 1,500 requests/day.
"""

import json
import base64
import io
import requests
from PIL import Image
import streamlit as st

MODEL = "gemini-1.5-flash-latest"
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL}:generateContent"


def get_client():
    """Returns the API key — kept as 'client' for compatibility."""
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


def _call_gemini(api_key: str, image: Image.Image, prompt: str) -> tuple:
    """Call Gemini REST API. Returns (response_text, error_string)."""
    img_b64 = image_to_base64(image)
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}}
            ]
        }],
        "generationConfig": {"maxOutputTokens": 4000, "temperature": 0.1}
    }
    try:
        resp = requests.post(
            f"{API_URL}?key={api_key}",
            json=payload,
            timeout=60
        )
        if resp.status_code != 200:
            return "", f"HTTP {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text, ""
    except Exception as e:
        return "", str(e)


PROMPT = """You are reading a page from a lighting product catalog or price list.

Extract ONE entry for each product CODE (article number) visible on this page.
Each row in the price table = one code with its own color and price.

RULES:
- One JSON object per code/row
- Prices are plain numbers, no currency symbol (e.g. 3120,00 or 3120.00)
- Find the currency label in the column HEADER (e.g. "RMBexcl. VAT" means currency="RMB")
- Convert comma decimals to dots: 3120,00 → 3120.00
- Include accessories if they have codes and prices
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

Return ONLY a valid JSON array. No explanation, no markdown, just the array."""


def extract_products_from_page(api_key: str, page_image: Image.Image, page_num: int) -> list:
    text, error = _call_gemini(api_key, page_image, PROMPT)
    if error:
        print(f"Page {page_num} error: {error}")
        return []
    return _parse_json(text)


def extract_products_debug(api_key: str, page_image: Image.Image) -> dict:
    out = {"model": MODEL, "raw_response": "", "parsed": [], "error": ""}
    text, error = _call_gemini(api_key, page_image, PROMPT)
    out["raw_response"] = text
    out["error"] = error
    if text:
        out["parsed"] = _parse_json(text)
    return out


def describe_image(api_key: str, image: Image.Image) -> str:
    prompt = "Describe this lighting product briefly: type, shape, color, style, any visible codes."
    text, _ = _call_gemini(api_key, image, prompt)
    return text
