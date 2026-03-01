"""
AI extraction using Moonshot AI (Kimi) vision model.
Get an API key at: https://platform.moonshot.cn
OpenAI-compatible endpoint.
"""

import json
import re
import base64
import io
import time
import requests
from PIL import Image
import streamlit as st


API_URL = "https://api.moonshot.cn/v1/chat/completions"

# Moonshot vision model
CANDIDATES = [
    "moonshot-v1-8k-vision-preview",
    "moonshot-v1-32k-vision-preview",
]

_working_model = {"model": None}


def get_client():
    api_key = st.secrets.get("MOONSHOT_API_KEY", "")
    if not api_key:
        st.error(
            "MOONSHOT_API_KEY not found in Streamlit secrets. "
            "Get a key at platform.moonshot.cn"
        )
        st.stop()
    return api_key


def image_to_base64(image: Image.Image) -> tuple:
    """Returns (base64_string, mime_type).  Resizes and compresses to JPEG.
    Keep max side at 1024px, quality 75.
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
    # Truncated JSON
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


_CCT_RE = re.compile(r"^(\d{3,4}[Kk])\s+(.+)$")


def _clean_products(products: list) -> list:
    """Strip CCT values that the AI accidentally puts inside codes."""
    cleaned = []
    for p in products:
        p = dict(p)
        new_codes = []
        found_cct = p.get("cct", "")
        for code in p.get("codes", []):
            m = _CCT_RE.match(str(code).strip())
            if m:
                if not found_cct:
                    found_cct = m.group(1)
                new_codes.append(m.group(2).strip())
            else:
                new_codes.append(code)
        if new_codes:
            p["codes"] = new_codes
        if found_cct:
            p["cct"] = found_cct
        cleaned.append(p)
    return cleaned


def _call(api_key: str, model: str, image: Image.Image, prompt: str) -> tuple:
    """Returns (response_text, error_string).  Retries on 429 with backoff."""
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
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
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
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return text, ""
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
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

CRITICAL NAME RULE:
- Product catalogs show a FAMILY NAME in a bold header above rows of variants (e.g. "AVRO Studio Natural", "SPIDER LED", "MARBLE 40")
- The full name may be 2–4 words — copy it EXACTLY and COMPLETELY (e.g. "AVRO Studio Natural" NOT just "AVRO")
- ALL rows below that header share that EXACT same full family name until a new header appears
- If the header is cut off (not visible in this image), look at the codes: find the nearest name visible anywhere and use it; or use the code prefix (e.g. "21019")
- NEVER shorten the name — use every word that appears in the header
- NEVER leave name as null or "?" — always put something
- For accessories, use the full accessory description as the name

IMPORTANT — AVOID DUPLICATES:
- This image is ONE SECTION of a full page. Some rows near the very top or bottom edges may be partially visible (cut off).
- Do NOT extract a row if its code text is clearly cut off or only partially visible at the edge.
- Only extract rows where you can fully read the product code.

RULES:
- One JSON object per code/row
- Prices are plain numbers (e.g. 3120.00) — convert comma decimals: 3120,00 → 3120.00
- Find currency from the column HEADER (e.g. "RMBexcl. VAT" → "RMB")
- OMIT any field that has no value — do NOT write null, just skip the key entirely
- Keep field values short and factual
- If no product codes visible (cover, index, pure text page), return []
- CCT RULE: Values like "2700K", "3000K", "4000K" are color temperatures shown in a CCT column — put them in the \`cct\` field ONLY, NEVER include them in the \`codes\` array.  Codes are alphanumeric article numbers like "21019/DIM/AR" — they never start with a temperature value
- If a CCT value (e.g. 2700K) appears next to multiple codes in the same group, apply that cct value to all those codes

Fields to include (only when value exists):
- codes: ["CODE"]  — required
- name: FULL product family name (e.g. "AVRO Studio Natural", "SPIDER LED 40") — ALWAYS required, use every word, never "?"
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
    """Split a page into 2 halves (top/bottom) with a small overlap.

    Catalog pages typically have 1–2 product families stacked vertically.
    Using 2 sections instead of 4 greatly reduces the chance of the same
    product rows appearing in multiple sections and being extracted twice.
    """
    w, h = image.size
    overlap = int(h * 0.06)  # 6 % overlap to avoid cutting rows at the boundary
    mid = h // 2
    s1 = image.crop((0, 0, w, mid + overlap))
    s2 = image.crop((0, mid - overlap, w, h))
    return s1, s2


def _normalize_code(code: str) -> str:
    """Normalize a product code for comparison: uppercase, strip whitespace,
    remove common separators."""
    return re.sub(r"[\s/\-_]+", "", str(code).strip().upper())


def _extract_section(api_key: str, section_image: Image.Image) -> list:
    text, error, _ = _call_best(api_key, section_image, PROMPT)
    if error:
        return []
    return _clean_products(_parse_json(text))


def _dedup(products: list) -> list:
    """De-duplicate products using normalized code matching.

    Two products are considered duplicates if any of their codes match after
    normalization (uppercased, separators stripped).  When duplicates are found,
    the entry with MORE fields filled in is kept.
    """
    result = []
    seen_codes = set()

    for p in products:
        codes = p.get("codes", [])
        norm_codes = [_normalize_code(c) for c in codes]

        # Check if ANY of this product's codes were already seen
        if any(nc in seen_codes for nc in norm_codes):
            # Duplicate — check if this version has more info
            for i, existing in enumerate(result):
                ex_codes = [_normalize_code(c) for c in existing.get("codes", [])]
                if any(nc in ex_codes for nc in norm_codes):
                    # Keep whichever has more non-empty fields
                    new_fields = sum(1 for v in p.values() if v)
                    old_fields = sum(1 for v in existing.values() if v)
                    if new_fields > old_fields:
                        result[i] = p
                    break
            continue

        seen_codes.update(norm_codes)
        result.append(p)

    return result


def extract_products_from_page(api_key: str, page_image: Image.Image, page_num: int) -> list:
    """Extract all products from a page.

    Splits into 2 halves (top / bottom); waits 3 s between calls to avoid
    rate limiting.  Uses fuzzy dedup to remove products extracted from the
    overlap zone.
    """
    s1, s2 = _split_image(page_image)
    results = []
    for section in (s1, s2):
        results.extend(_extract_section(api_key, section))
        time.sleep(3)
    return _dedup(results)


def extract_products_debug(api_key: str, page_image: Image.Image) -> dict:
    """Debug version — shows raw responses from both sections."""
    s1, s2 = _split_image(page_image)

    t1, e1, m1 = _call_best(api_key, s1, PROMPT)
    time.sleep(3)
    t2, e2, _ = _call_best(api_key, s2, PROMPT)

    p1 = _clean_products(_parse_json(t1)) if t1 else []
    p2 = _clean_products(_parse_json(t2)) if t2 else []

    all_products = _dedup(p1 + p2)

    raw = (
        f"=== SECTION 1 ({len(p1)} products) ===\n{t1}\n\n"
        f"=== SECTION 2 ({len(p2)} products) ===\n{t2}"
    )

    return {
        "model": m1 or "unknown",
        "raw_response": raw,
        "parsed": all_products,
        "error": e1 or e2,
    }


def describe_image(api_key: str, image: Image.Image) -> str:
    prompt = "Describe this lighting product briefly: type, shape, color, style, any visible codes."
    text, _, _ = _call_best(api_key, image, prompt)
    return text


DIM_BOX_PROMPT = """This is a page from a lighting product catalog.

Find every DIMENSION DRAWING on this page.  A dimension drawing is a technical
outline/silhouette of a lamp or light fixture (pendant, wall lamp, floor lamp,
etc.) that has measurement annotations around it — numbers like Ø60, Ø35, H25,
W40, etc.  These drawings are always inside a clearly bordered rectangular box,
located on the LEFT side of each product section.

For each dimension drawing box, return its bounding box as percentages of the
full image width and height.

Return ONLY a JSON array, nothing else:
[{"x0": 5, "y0": 8, "x1": 35, "y1": 50}, {"x0": 5, "y0": 53, "x1": 35, "y1": 97}]

Rules:
- x0,y0 = top-left of the drawing box (percentages 0-100)
- x1,y1 = bottom-right of the drawing box (percentages 0-100)
- Include ONLY the bordered box containing the lamp silhouette and measurement labels
- Do NOT include product name text, spec tables, "Light source", "Type", "Volt" text, or any accessory lists
- Crop tightly to the drawing box border
- If no dimension drawings exist on this page, return []
- Sort top to bottom
- Return ALL drawings found, not just the first two"""


def find_dim_boxes(api_key: str, page_image: Image.Image) -> list:
    """
    Ask the vision AI to locate dimension drawing bounding boxes on a catalog page.
    Returns a list of dicts: [{"x0": %, "y0": %, "x1": %, "y1": %}, ...]
    Percentages are 0-100 relative to the full image size.
    Returns [] on failure.
    """
    text, err, _ = _call_best(api_key, page_image, DIM_BOX_PROMPT)
    if err or not text:
        return []
    boxes = _parse_json(text)
    # Validate: each entry must have x0,y0,x1,y1 all as numbers in 0-100
    valid = []
    for b in boxes:
        try:
            x0, y0, x1, y1 = float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])
            if 0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 100:
                valid.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
        except (KeyError, TypeError, ValueError):
            continue
    return valid
