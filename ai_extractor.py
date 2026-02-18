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
    Send a rendered PDF page to Qwen and extract all product data from it.
    Returns a list of product dicts.
    """
    img_b64 = image_to_base64(page_image)

    prompt = """You are reading a page from a lighting product catalog.
Extract EVERY product visible on this page.

For each product return a JSON object with these fields (use null if not found):
- codes: array of ALL article/product codes (one image often covers multiple codes)
- name: product name
- description: full product description
- color: color name or code
- light_source: LED, halogen, fluorescent, etc.
- dimensions: size/dimensions
- wattage: power in watts
- price: price as a number ONLY (no currency symbol)
- currency: the currency symbol found (€, $, £, etc.)
- extra_fields: object with any other specifications (IP rating, lumen, CRI, kelvin, material, etc.)

Return ONLY a valid JSON array like:
[{"codes": ["AB123", "AB124"], "name": "Wall Light", "color": "White", "price": 149.00, "currency": "€", ...}]

If no products are on this page, return [].
Do not include any explanation, only the JSON array."""

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
            max_tokens=2500
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
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
    """
    Generate a text description of a product image.
    Used to enable semantic image search.
    """
    img_b64 = image_to_base64(image)

    prompt = """Describe this lighting product in detail for product matching purposes. Include:
- Type (pendant, wall light, floor lamp, ceiling light, spotlight, etc.)
- Shape and silhouette
- Materials and finish
- Colors
- Style (modern, industrial, classic, minimalist, etc.)
- Any visible codes or numbers

Be specific. Return only the description, no extra text."""

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
