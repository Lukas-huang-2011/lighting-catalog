"""
AI extraction using Qwen vision via OpenRouter.
Reads PDF pages and extracts product information.
"""

import json
import base64
import io
from PIL import Image
from openai import OpenAI


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


def extract_products_from_page(client: OpenAI, page_image: Image.Image, page_num: int) -> list:
    """
    Send a rendered PDF page to Qwen and extract all product rows from it.
    Returns a list of dicts — one dict per product CODE (article number).
    """
    img_b64 = image_to_base64(page_image)

    prompt = """You are reading a page from a lighting product catalog or price list.

Extract ONE entry for each product CODE (article number) you find.
Each row in the table is typically one code with its own color and price.

CRITICAL RULES:
- Return one JSON object per code/row, not one per product family
- Prices are often numbers WITHOUT a currency symbol (e.g. "14469,00" or "1234.50")
- Find the currency by reading column HEADERS (e.g. "RMB excl. VAT", "EUR excl. VAT", "$ Price")
- Convert comma-decimal prices to dot-decimal: 14469,00 → 14469.00
- If there are accessories listed separately, include those too (they have codes and prices)
- If a page has no product codes or prices (e.g. table of contents, cover), return []

For each code row extract these fields (use null if not found):
- codes: array with ONLY this one code, e.g. ["21019/DIM/AR"]
- name: full product name including family and variant (e.g. "AVRO Studio Natural")
- description: product description, type, emission type
- color: color name and RAL/code for this specific row (e.g. "Orange Polished RAL 2001")
- light_source: full light source text (e.g. "36W 5000lm Integrated LED", "max 75W E27")
- cct: color temperature (e.g. "2700K", "3000K", "4000K")
- dimensions: dimensions string (e.g. "Ø 400 H 2200 mm")
- wattage: wattage with unit (e.g. "36W")
- price: price as a decimal NUMBER only, no symbols (e.g. 14469.00)
- currency: currency label from the column header (e.g. "RMB", "EUR", "USD", "GBP")
- extra_fields: object with any other specs — IP rating, dimming info, voltage, driver type,
  materials, weight, package dimensions, mounting type, etc.

Return ONLY a valid JSON array. Example:
[
  {"codes": ["21019/DIM/AR"], "name": "AVRO Studio Natural", "color": "Orange Polished RAL 2001",
   "light_source": "36W 5000lm Integrated LED", "cct": "2700K", "price": 14469.00,
   "currency": "RMB", "wattage": "36W", "dimensions": "Ø 400 H 2200 mm",
   "extra_fields": {"ip": "IP20", "dimming": "DALI 2", "voltage": "220-240V"}},
  {"codes": ["21019/DIM/AZ"], "name": "AVRO Studio Natural", "color": "Azure Polished RAL 5014", ...}
]

Return [] if no products are on this page. Return only JSON, no explanation."""

    try:
        response = client.chat.completions.create(
            model="qwen/qwen2.5-vl-72b-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=4000
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if "```" in content:
            for part in content.split("```"):
                part = part.strip().lstrip("json").strip()
                try:
                    result = json.loads(part)
                    if isinstance(result, list):
                        return result
                except Exception:
                    continue

        result = json.loads(content)
        return result if isinstance(result, list) else []

    except Exception as e:
        print(f"AI extraction error on page {page_num}: {e}")
        return []


def describe_image(client: OpenAI, image: Image.Image) -> str:
    """Generate a text description of a product image for semantic search."""
    img_b64 = image_to_base64(image)

    prompt = """Describe this lighting product for product matching. Include:
- Type (pendant, wall light, floor lamp, ceiling light, spotlight, etc.)
- Shape and silhouette
- Materials and finish
- Colors
- Style (modern, industrial, classic, minimalist, etc.)
- Any visible codes or numbers

Be specific. Return only the description."""

    try:
        response = client.chat.completions.create(
            model="qwen/qwen2.5-vl-72b-instruct",
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
        print(f"Image description error: {e}")
        return ""
