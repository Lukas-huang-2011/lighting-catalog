"""
AI extraction using Google Gemini API directly.
Free tier: 1,500 requests/day, no credit card needed.
"""

import json
import base64
import io
from PIL import Image
import google.generativeai as genai
import streamlit as st

MODEL_NAME = "gemini-1.5-flash"


def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Add it from aistudio.google.com")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def image_to_pil(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _parse_json_response(content: str) -> list:
    content = content.strip()
    if "```" in content:
        for part in content.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                result = json.loads(part)
                if isinstance(result, list):
                    return result
            except Exception:
                continue
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1:
        try:
            result = json.loads(content[start:end+1])
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return []


EXTRACTION_PROMPT = """You are reading a page from a lighting product catalog or price list.

Extract ONE entry for each product CODE (article number) visible on this page.
Each row in the price table = one code with its own color and price.

RULES:
- One JSON object per code/row
- Prices are plain numbers with no currency symbol (e.g. 3120,00 or 3120.00)
- Find the currency label in the column HEADER (e.g. "RMBexcl. VAT" means currency is "RMB")
- Convert comma decimals to dots: 3120,00 → 3120.00
- Include accessories if they have codes and prices
- If the page has no product codes (e.g. index, cover), return []

Fields per entry (null if missing):
- codes: ["SINGLE_CODE"] — only one code per entry
- name: product family name (e.g. "CABRIOLETTE Body lamp")
- description: emission type, light source description
- color: color + RAL for this specific row (e.g. "White Polished RAL 9003")
- light_source: full text (e.g. "7,5W 1110lm Integrated LED")
- cct: color temperature (e.g. "2700K")
- dimensions: from drawing or spec (e.g. "Ø15,5 H 28 cm")
- wattage: just the watts (e.g. "7.5W")
- price: number only, dot decimal (e.g. 3120.00)
- currency: from column header (e.g. "RMB", "EUR", "USD")
- extra_fields: object with ip_rating, dimming, voltage, driver, structure, diffuser,
  net_weight, gross_weight, package_nr, package_dimension, etc.

Return ONLY a valid JSON array. No explanation, no markdown, just the JSON array."""


def extract_products_from_page(client, page_image: Image.Image, page_num: int) -> list:
    try:
        response = client.generate_content([EXTRACTION_PROMPT, page_image])
        content = response.text.strip()
        return _parse_json_response(content)
    except Exception as e:
        print(f"Page {page_num} extraction error: {e}")
        return []


def extract_products_debug(client, page_image: Image.Image) -> dict:
    out = {"model": MODEL_NAME, "raw_response": "", "parsed": [], "error": ""}
    try:
        response = client.generate_content([EXTRACTION_PROMPT, page_image])
        out["raw_response"] = response.text.strip()
        out["parsed"] = _parse_json_response(out["raw_response"])
    except Exception as e:
        out["error"] = str(e)
    return out


def describe_image(client, image: Image.Image) -> str:
    try:
        prompt = "Describe this lighting product briefly: type, shape, color, style, any visible codes."
        response = client.generate_content([prompt, image])
        return response.text.strip()
    except Exception:
        return ""
