"""
AI extraction using Google Gemini Flash via OpenRouter.
Free model, excellent at reading structured PDF tables.
"""

import json
import base64
import io
from PIL import Image
from openai import OpenAI

# Primary model: Gemini Flash (free, excellent vision)
# Fallback: Qwen VL (also free tier)
PRIMARY_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"
FALLBACK_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"


def get_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"HTTP-Referer": "https://lighting-catalog.streamlit.app"}
    )


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_json_response(content: str) -> list:
    """Try to extract a JSON array from the model response."""
    content = content.strip()

    # Strip markdown code fences
    if "```" in content:
        for part in content.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                result = json.loads(part)
                if isinstance(result, list):
                    return result
            except Exception:
                continue

    # Try direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    # Try finding array inside the text
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


def _call_model(client: OpenAI, model: str, img_b64: str, prompt: str) -> tuple[str, str]:
    """Call a model, return (raw_response_text, error_message)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=4000
        )
        return response.choices[0].message.content.strip(), ""
    except Exception as e:
        return "", str(e)


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

Return ONLY a valid JSON array. Example for this type of page:
[
  {"codes":["40224/BI"],"name":"CABRIOLETTE Body lamp","color":"White Polished RAL 9003","light_source":"7,5W 1110lm","cct":"2700K","price":3120.00,"currency":"RMB","wattage":"7.5W","dimensions":"Ø15,5 H28cm","description":"Direct Light, Adjustable Light, Integrated LED","extra_fields":{"voltage":"24V","driver":"Excluded External","structure":"Aluminium","diffuser":"Methacrylate 040","net_weight":"1,6 Kg","package_dimension":"20x32x20 cm"}},
  {"codes":["40224/NE"],"name":"CABRIOLETTE Body lamp","color":"Black Polished RAL 9005","price":3120.00,"currency":"RMB",...}
]"""


def extract_products_from_page(client: OpenAI, page_image: Image.Image, page_num: int) -> list:
    """Extract all product rows from a PDF page. Returns list of product dicts."""
    img_b64 = image_to_base64(page_image)

    # Try primary model first
    raw, error = _call_model(client, PRIMARY_MODEL, img_b64, EXTRACTION_PROMPT)

    if error or not raw:
        # Try fallback model
        raw, error = _call_model(client, FALLBACK_MODEL, img_b64, EXTRACTION_PROMPT)

    if error:
        print(f"Page {page_num} — both models failed: {error}")
        return []

    result = _parse_json_response(raw)
    return result


def extract_products_debug(client: OpenAI, page_image: Image.Image) -> dict:
    """
    Debug version: returns raw response + parsed result + any errors.
    Used by the Debug & Test page.
    """
    img_b64 = image_to_base64(page_image)

    out = {"primary_model": PRIMARY_MODEL, "fallback_model": FALLBACK_MODEL,
           "raw_response": "", "parsed": [], "error": ""}

    raw, error = _call_model(client, PRIMARY_MODEL, img_b64, EXTRACTION_PROMPT)
    out["raw_response"] = raw
    out["error"] = error

    if error or not raw:
        raw2, error2 = _call_model(client, FALLBACK_MODEL, img_b64, EXTRACTION_PROMPT)
        out["raw_response_fallback"] = raw2
        out["error_fallback"] = error2
        raw = raw2

    if raw:
        out["parsed"] = _parse_json_response(raw)

    return out


def describe_image(client: OpenAI, image: Image.Image) -> str:
    """Generate a text description of a product image for semantic search."""
    img_b64 = image_to_base64(image)
    prompt = """Describe this lighting product for matching. Include type, shape, materials, colors, style, any visible codes. Be specific and concise."""
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return ""
