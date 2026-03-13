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
    """Returns (base64_string, mime_type).
    Resizes and compresses to JPEG.
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
    """Returns (response_text, error_string). Retries on 429 with backoff."""
    img_b64, mime = image_to_base64(image)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": 4096,
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


PROMPT = """You are reading a FULL PAGE of a lighting product catalog or price list.
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
- *** ACCESSORY EXCEPTION ***: "Accessories" / "Accessori" is a SECTION HEADER, NOT a product name. NEVER use "Accessories" or "Accessori" as the name field. For each row under that header, the name is the ITEM DESCRIPTION from the first column (e.g. "Multisocket with cable", "Tuya WiFi, RF, BT device for dimming", "Button for Tuya"). Multiple color variants of the same accessory item share that item's description as their name.
DIMENSION ANNOTATION RULE: Labels like "Ø35", "Ø60", "H25", "W120" next to technical drawings are SIZE MEASUREMENTS — they are NEVER product names or codes. The product name always comes from a bold section header (e.g. "AVRO Junior Studio Natural"). Never use a measurement label as a name.
CRITICAL — PRODUCT POSITION INDEX:
- This page may show MULTIPLE product families stacked vertically (e.g. one product at the top half, another at the bottom half).
- Each product family has a PHOTO/IMAGE associated with it — the photo is positioned near that product's section.
- You MUST assign a `product_index` to every row: 0 for the FIRST (topmost) product family on the page, 1 for the SECOND product family, 2 for the third, etc.
- ALL variant rows and accessories under the SAME product family header share the SAME `product_index`.
- This index is critical for matching product photos to the correct product. Count product families from top to bottom starting at 0.
CRITICAL — AVOID DUPLICATES:
- If the same accessory codes appear under MULTIPLE product families on this page (e.g. the same "Multisocket with cable" or "Tuya WiFi" rows repeat for different products), extract them ONLY ONCE — the first occurrence.
- Each unique product code should appear exactly ONCE in your output.
- Do NOT repeat an accessory row just because it shows up under a second product family.
RULES:
- CRITICAL — ONE CODE PER OBJECT: Each JSON object = exactly ONE product row = exactly ONE code in the `codes` array. NEVER group multiple codes from the same family into one object. If a family has 10 color variants (10 rows), output 10 separate JSON objects — one per row. The `codes` array always has exactly 1 element: ["SINGLE-CODE"].
- Prices are plain numbers (e.g. 3120.00) — convert comma decimals: 3120,00 → 3120.00
- Find currency from the column HEADER (e.g. "RMBexcl. VAT" → "RMB")
- OMIT any field that has no value — do NOT write null, just skip the key entirely
- Keep field values short and factual
- If no product codes visible (cover, index, pure text page), return []
- CCT RULE: Values like "2700K", "3000K", "4000K" are color temperatures shown in a CCT column — put them in the `cct` field ONLY, NEVER include them in the `codes` array. Codes are alphanumeric article numbers like "21019/DIM/AR" — they never start with a temperature value
- If a CCT value (e.g. 2700K) appears next to multiple codes in the same group, apply that cct value to all those codes
Fields to include (only when value exists):
- codes: ["CODE"] — required, always exactly ONE code per object
- name: FULL product family name (e.g. "AVRO Studio Natural", "SPIDER LED 40") — ALWAYS required, use every word, never "?"
- product_index: integer — REQUIRED — 0 for topmost product family, 1 for the next one down, etc.
- color: color name (e.g. "Arancio", "Bianco")
- light_source: e.g. "7.5W 1110lm Integrated LED"
- cct: e.g. "2700K"
- dimensions: e.g. "Ø15.5 H28 cm"
- wattage: e.g. "7.5W"
- price: number (e.g. 3120.00) — required if visible
- currency: e.g. "RMB" — required if visible
- type: lamp type from the "Type:" spec line — e.g. "pendant", "wall", "ceiling", "floor", "table" — include for ALL rows in that product family if visible; omit if not stated
- description: short description, mounting type, or accessory use
- extra_fields: object with any of {ip_rating, dimming, voltage, driver, structure, diffuser, net_weight}
- is_accessory: true — include ONLY for rows under an "Accessories" / "Accessori" section header; omit this field entirely for main product rows
Return ONLY a valid JSON array. No explanation. No markdown. Include EVERY unique code row."""

PROMPT_SECTION = """You are reading a portion of a lighting product catalog or price list.
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
- *** ACCESSORY EXCEPTION ***: "Accessories" / "Accessori" is a SECTION HEADER, NOT a product name. NEVER use "Accessories" or "Accessori" as the name field. For each row under that header, the name is the ITEM DESCRIPTION from the first column (e.g. "Multisocket with cable", "Tuya WiFi, RF, BT device for dimming", "Button for Tuya"). Multiple color variants of the same accessory item share that item's description as their name.
DIMENSION ANNOTATION RULE: Labels like "Ø35", "Ø60", "H25", "W120" next to technical drawings are SIZE MEASUREMENTS — they are NEVER product names or codes. The product name always comes from a bold section header. Never use a measurement label as a name.
CRITICAL — PRODUCT POSITION INDEX:
- This section may show MULTIPLE product families stacked vertically.
- You MUST assign a `product_index` to every row: 0 for the FIRST (topmost) product family visible in this section, 1 for the SECOND, etc.
- ALL variant rows and accessories under the SAME product family header share the SAME `product_index`.
IMPORTANT — AVOID DUPLICATES:
- This image is ONE SECTION of a full page. Some rows near the very top or bottom edges may be partially visible (cut off).
- Do NOT extract a row if its code text is clearly cut off or only partially visible at the edge.
- Only extract rows where you can fully read the product code.
RULES:
- CRITICAL — ONE CODE PER OBJECT: Each JSON object = exactly ONE product row = exactly ONE code in the `codes` array. NEVER group multiple codes from the same family into one object. If a family has 10 color variants (10 rows), output 10 separate JSON objects — one per row. The `codes` array always has exactly 1 element: ["SINGLE-CODE"].
- Prices are plain numbers (e.g. 3120.00) — convert comma decimals: 3120,00 → 3120.00
- Find currency from the column HEADER (e.g. "RMBexcl. VAT" → "RMB")
- OMIT any field that has no value — do NOT write null, just skip the key entirely
- Keep field values short and factual
- If no product codes visible (cover, index, pure text page), return []
- CCT RULE: Values like "2700K", "3000K", "4000K" are color temperatures shown in a CCT column — put them in the `cct` field ONLY, NEVER include them in the `codes` array. Codes are alphanumeric article numbers like "21019/DIM/AR" — they never start with a temperature value
- If a CCT value (e.g. 2700K) appears next to multiple codes in the same group, apply that cct value to all those codes
Fields to include (only when value exists):
- codes: ["CODE"] — required, always exactly ONE code per object
- name: FULL product family name (e.g. "AVRO Studio Natural", "SPIDER LED 40") — ALWAYS required, use every word, never "?"
- product_index: integer — REQUIRED — 0 for topmost product family in this section, 1 for the next, etc.
- color: color name (e.g. "Arancio", "Bianco")
- light_source: e.g. "7.5W 1110lm Integrated LED"
- cct: e.g. "2700K"
- dimensions: e.g. "Ø15.5 H28 cm"
- wattage: e.g. "7.5W"
- price: number (e.g. 3120.00) — required if visible
- currency: e.g. "RMB" — required if visible
- type: lamp type from the "Type:" spec line — e.g. "pendant", "wall", "ceiling", "floor", "table" — include for ALL rows in that product family if visible; omit if not stated
- description: short description, mounting type, or accessory use
- extra_fields: object with any of {ip_rating, dimming, voltage, driver, structure, diffuser, net_weight}
- is_accessory: true — include ONLY for rows under an "Accessories" / "Accessori" section header; omit this field entirely for main product rows
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
    """Normalize a product code for comparison: uppercase, strip whitespace, remove common separators."""
    return re.sub(r"[\s/\-_]+", "", str(code).strip().upper())


def _extract_section(api_key: str, section_image: Image.Image) -> list:
    text, error, _ = _call_best(api_key, section_image, PROMPT_SECTION)
    if error:
        return []
    return _clean_products(_parse_json(text))


def _is_truncated(raw_text: str, products: list) -> bool:
    """Detect if the AI response was truncated (ran out of tokens).
    Signs of truncation:
    - raw text ends abruptly without closing ]
    - raw text ends with ... or incomplete JSON
    - we parsed via the truncated-JSON fallback path
    """
    if not raw_text:
        return True
    stripped = raw_text.strip()
    # If it ends properly with ] then it's not truncated
    if stripped.endswith("]"):
        return False
    # If we got products but the JSON didn't close, it was truncated
    if products and not stripped.endswith("]"):
        return True
    return False


def _dedup(products: list) -> list:
    """De-duplicate products using normalized code matching.
    Two products are considered duplicates if any of their codes match
    after normalization (uppercased, separators stripped).
    When duplicates are found, the entry with MORE fields filled in is kept.
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


def _fix_section_indices(section1_products: list, section2_products: list) -> list:
    """When using split-page extraction, section 2 indices need to be offset
    by the number of unique product families found in section 1.
    """
    # Find the max product_index in section 1
    max_idx_s1 = -1
    for p in section1_products:
        idx = p.get("product_index", 0)
        if idx > max_idx_s1:
            max_idx_s1 = idx
    # If section 2 products start at index 0 but section 1 already has products,
    # we need to check if section 2 continues the same product family or starts new ones.
    # Use product name matching: if section 2's index-0 name matches section 1's last name, don't offset.
    if not section1_products or not section2_products:
        return section1_products + section2_products
    # Get the name of the last product family in section 1
    last_s1_name = ""
    for p in section1_products:
        if p.get("product_index", 0) == max_idx_s1:
            last_s1_name = (p.get("name") or "").strip().upper()
            break
    # Get the name of the first product family in section 2
    first_s2_name = ""
    for p in section2_products:
        if p.get("product_index", 0) == 0:
            first_s2_name = (p.get("name") or "").strip().upper()
            break
    # If they match, section 2's index 0 = section 1's last index (continuation)
    if last_s1_name and first_s2_name and last_s1_name == first_s2_name:
        offset = max_idx_s1  # same family continues
    else:
        offset = max_idx_s1 + 1  # new family starts
    # Apply offset to section 2
    for p in section2_products:
        old_idx = p.get("product_index", 0)
        p["product_index"] = old_idx + offset
    return section1_products + section2_products


def extract_products_from_page(api_key: str, page_image: Image.Image, page_num: int) -> list:
    """Extract all products from a page.
    Strategy:
    1. Try the FULL PAGE as a single image first (avoids overlap duplication).
    2. If the response is truncated (too many products for one call),
       fall back to splitting into 2 halves with dedup.
    """
    # --- Attempt 1: full page, single call ---
    text, error, _ = _call_best(api_key, page_image, PROMPT)
    if not error:
        products = _clean_products(_parse_json(text))
        if products and not _is_truncated(text, products):
            # Full page worked — return as-is (already deduplicated by nature)
            return _dedup(products)
    # --- Attempt 2: split into 2 halves ---
    time.sleep(3)
    s1, s2 = _split_image(page_image)
    p1 = _extract_section(api_key, s1)
    time.sleep(3)
    p2 = _extract_section(api_key, s2)
    # Fix product_index for section 2 so indices are page-global
    combined = _fix_section_indices(p1, p2)
    return _dedup(combined)


def extract_products_debug(api_key: str, page_image: Image.Image) -> dict:
    """Debug version — tries full page first, falls back to sections. Shows raw responses."""
    # --- Attempt 1: full page ---
    t_full, e_full, m1 = _call_best(api_key, page_image, PROMPT)
    p_full = _clean_products(_parse_json(t_full)) if t_full else []
    truncated = _is_truncated(t_full, p_full)
    if p_full and not truncated and not e_full:
        # Full page extraction worked
        all_products = _dedup(p_full)
        raw = (
            f"=== FULL PAGE ({len(p_full)} products, no split needed) ===\n{t_full}"
        )
        return {
            "model": m1 or "unknown",
            "raw_response": raw,
            "parsed": all_products,
            "error": e_full,
        }
    # --- Fallback: 2 sections ---
    time.sleep(3)
    s1, s2 = _split_image(page_image)
    t1, e1, m1b = _call_best(api_key, s1, PROMPT_SECTION)
    time.sleep(3)
    t2, e2, _ = _call_best(api_key, s2, PROMPT_SECTION)
    p1 = _clean_products(_parse_json(t1)) if t1 else []
    p2 = _clean_products(_parse_json(t2)) if t2 else []
    # Fix product_index for section 2
    combined = _fix_section_indices(p1, p2)
    all_products = _dedup(combined)
    raw = (
        f"=== FULL PAGE attempt ({len(p_full)} products, truncated={truncated}) ===\n{t_full}\n\n"
        f"=== SECTION 1 ({len(p1)} products) ===\n{t1}\n\n"
        f"=== SECTION 2 ({len(p2)} products) ===\n{t2}"
    )
    return {
        "model": m1 or m1b or "unknown",
        "raw_response": raw,
        "parsed": all_products,
        "error": e_full or e1 or e2,
    }


def describe_image(api_key: str, image: Image.Image) -> str:
    prompt = "Describe this lighting product briefly: type, shape, color, style, any visible codes."
    text, _, _ = _call_best(api_key, image, prompt)
    return text


DIM_BOX_PROMPT = """This is a page from a lighting product catalog.
Find every DIMENSION DRAWING on this page.
A dimension drawing is a technical outline/silhouette of a lamp or light fixture
(pendant, wall lamp, floor lamp, etc.) — it shows the shape of the product with
measurement annotations like Ø60, Ø35, H25, W40 around it.
These drawings sit inside a clearly bordered rectangular box on the LEFT side of
each product section.
CRITICAL — HOW MANY TO FIND:
- Catalog pages often show 2 or more product families stacked vertically, separated
  by a thin horizontal line. Each product family has its OWN bordered drawing box.
- Count the horizontal dividers on the page — each section above/below a divider
  that contains a lamp silhouette has its own drawing box. Find ALL of them.
- Do NOT stop after finding one. Keep scanning the full page top to bottom.
For each dimension drawing box, return its bounding box as percentages of the full
image width and height.
Return ONLY a JSON array, nothing else:
[{"x0": 5, "y0": 8, "x1": 35, "y1": 50}, {"x0": 5, "y0": 53, "x1": 35, "y1": 97}]
Rules:
- x0,y0 = top-left of the drawing box (percentages 0-100)
- x1,y1 = bottom-right of the drawing box (percentages 0-100)
- Include ONLY the bordered box containing the lamp silhouette and measurement labels
- Do NOT include product name text, spec tables, "Light source", "Type", "Volt" text, or any accessory lists
- Crop tightly to the drawing box border
- If no dimension drawings exist on this page, return []
- Sort top to bottom — first drawing in array = topmost on page"""

PHOTO_BOX_PROMPT = """This is a page from a lighting product catalog.
Find every PRODUCT PHOTO on this page.
A product photo is a real-life photograph (not a technical drawing) of a lamp or
light fixture. It shows the actual product in color — you can see materials,
textures, the lamp shade, the light effect, etc.
Product photos are typically positioned on the RIGHT side or CENTER of each
product section, or sometimes as a large background image.
Do NOT include:
- Dimension/technical drawings (silhouettes with measurement labels like Ø60, H25)
- Logos, brand marks, or decorative icons
- Color swatches or small accessory thumbnails
For each product photo, return its bounding box as percentages of the full
image width and height.
Return ONLY a JSON array, nothing else:
[{"x0": 40, "y0": 5, "x1": 95, "y1": 48}, {"x0": 40, "y0": 52, "x1": 95, "y1": 95}]
Rules:
- x0,y0 = top-left of the photo (percentages 0-100)
- x1,y1 = bottom-right of the photo (percentages 0-100)
- Sort top to bottom (the first photo in the array = topmost on page)
- If no product photos exist on this page, return []
- Return ALL photos found"""


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


def find_photo_boxes(api_key: str, page_image: Image.Image) -> list:
    """
    Ask the vision AI to locate product photo bounding boxes on a catalog page.
    Returns a list of dicts sorted top-to-bottom: [{"x0": %, "y0": %, "x1": %, "y1": %}, ...]
    Percentages are 0-100 relative to the full image size.
    Returns [] on failure.
    """
    text, err, _ = _call_best(api_key, page_image, PHOTO_BOX_PROMPT)
    if err or not text:
        return []
    boxes = _parse_json(text)
    valid = []
    for b in boxes:
        try:
            x0, y0, x1, y1 = float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])
            if 0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 100:
                valid.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
        except (KeyError, TypeError, ValueError):
            continue
    # Sort by vertical position (top to bottom)
    valid.sort(key=lambda b: b["y0"])
    return valid
