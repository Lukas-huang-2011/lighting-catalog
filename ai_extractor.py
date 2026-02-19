"""
AI extraction using Google Gemini via its OpenAI-compatible API.
Free tier: 1,500 requests/day. Uses the standard openai package.
"""

import json
import base64
import io
from PIL import Image
from openai import OpenAI
import streamlit as st

MODEL = "gemini-1.5-flash"


def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Get one free at aistudio.google.com")
        st.stop()
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )


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


PROMPT = """You are reading a page from a lighting product catalog or price list.

Extract ONE entry for each product CODE (article number) visible on this page.
Each row in the price table = one code with its own color and price.

RULES:
- One JSON object per code/row
- Prices are plain numbers, no currency symbol (e.g. 3120,00 or 3120.00)
- Find the currency label in the column HEADER (e.g. "RMBexcl. VAT" means currency="RMB")
- Convert comma decimals to dots: 3120,00 → 3120.00
- Include accessories if they have codes and prices
- If no product codes on this page (index, cover page), return []

Fields per entry (use null if not found):
- codes: ["ONE_CODE"] — just this one code
- name: product family name (e.g. "CABRIOLETTE Body lamp")
- description: emission type, light source description
- color: color name + RAL for this row (e.g. "White Polished RAL 9003")
- light_source: full text (e.g. "7.5W 1110lm Integrated LED")
- cct: color temperature (e.g. "2700K")
- dimensions: size string (e.g. "Ø15.5 H28 cm")
- wattage: watts only (e.g. "7.5W")
- price: number with dot decimal (e.g. 3120.00)
- currency: from column header (e.g. "RMB", "EUR", "USD")
- extra_fields: {ip_rating, dimming, voltage, driver, structure, diffuser, net_weight, gross_weight, package_dimension}

Return ONLY a valid JSON array, no explanation, no markdown."""


def extract_products_from_page(client: OpenAI, page_image: Image.Image, page_num: int) -> list:
    try:
        img_b64 = image_to_base64(page_image)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT}
            ]}],
            max_tokens=4000
        )
        return _parse_json(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Page {page_num} error: {e}")
        return []


def extract_products_debug(client: OpenAI, page_image: Image.Image) -> dict:
    out = {"model": MODEL, "raw_response": "", "parsed": [], "error": ""}
    try:
        img_b64 = image_to_base64(page_image)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT}
            ]}],
            max_tokens=4000
        )
        out["raw_response"] = response.choices[0].message.content.strip()
        out["parsed"] = _parse_json(out["raw_response"])
    except Exception as e:
        out["error"] = str(e)
    return out


def describe_image(client: OpenAI, image: Image.Image) -> str:
    try:
        img_b64 = image_to_base64(image)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Describe this lighting product briefly: type, shape, color, style, any visible codes."}
            ]}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""
