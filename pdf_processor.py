"""
PDF processing: page rendering, image extraction, price conversion.
Uses PyMuPDF (fitz).
"""

import re
import io
import fitz  # PyMuPDF
from PIL import Image


def get_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def render_single_page(pdf_bytes: bytes, page_num: int, dpi: int = 100) -> Image.Image:
    """Render one specific page as a PIL Image. Memory efficient."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_num = min(page_num, len(doc) - 1)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_num].get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def render_pages(pdf_bytes: bytes, dpi: int = 100):
    """Yield each page of a PDF as a PIL Image. Uses generator to save memory."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        yield img
    doc.close()


def _render_page_full(pdf_bytes: bytes, page_num: int, dpi: int = 150) -> Image.Image:
    """Render one page at the given DPI and return as RGB PIL image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_num].get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def _find_split_y(full: Image.Image) -> tuple:
    """
    Scan the full page for the brightest horizontal band between 30–70% height.
    Returns (best_y, brightness).  brightness > 235 means a clear separator
    between two stacked product families.
    """
    w, h = full.size
    scan_start = int(h * 0.30)
    scan_end   = int(h * 0.70)
    best_y, best_brightness = h // 2, 0.0
    for y in range(scan_start, scan_end, 5):
        strip = full.crop((0, y, w, y + 8))
        pxls  = list(strip.getdata())
        bright = sum(r + g + b for r, g, b in pxls) / (len(pxls) * 3)
        if bright > best_brightness:
            best_brightness = bright
            best_y = y + 4
    return best_y, best_brightness


def extract_page_images(pdf_bytes: bytes, page_num: int) -> dict:
    """
    Extract product illustration images AND dimension-drawing images from one page.

    Layout assumption (typical lighting catalog, vector-graphic style):
      LEFT  ~40%  — product illustration (for 图片 column)
      RIGHT ~58%  — dimension drawing with measurement labels (for 尺寸 column + image search)

    Returns:
        {
          'product': [PIL.Image, ...],   # 1–2 images, index 0=top product, 1=bottom product
          'dim':     [PIL.Image, ...],   # same structure, right-side dimension drawings
        }

    If the PDF has embedded raster images (photo-based catalogs), those are returned
    in 'product' and 'dim' is empty — embedded images can't be split by zone.
    """
    # ── Try embedded raster images first ─────────────────────────────────────
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    rasters, seen = [], set()
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            base_image = doc.extract_image(xref)
            img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            if img.width > 100 and img.height > 100:
                rasters.append(img)
        except Exception:
            pass
    doc.close()

    if rasters:
        # Photo-based catalog — return embedded images as product photos; no dim drawings
        return {"product": rasters[:2], "dim": []}

    # ── Fallback: render page and crop by zone ────────────────────────────────
    full = _render_page_full(pdf_bytes, page_num, dpi=150)
    w, h = full.size

    # Column boundaries
    prod_right  = int(w * 0.40)   # right edge of illustration zone
    dim_left    = int(w * 0.42)   # left  edge of dimension-drawing zone

    # Detect horizontal separator (two products stacked vertically)
    best_y, best_brightness = _find_split_y(full)
    two_products = best_brightness > 235

    if two_products:
        prod_regions = [
            full.crop((0,        0,      prod_right, best_y)),
            full.crop((0,        best_y, prod_right, h)),
        ]
        dim_regions = [
            full.crop((dim_left, 0,      w,          best_y)),
            full.crop((dim_left, best_y, w,          h)),
        ]
    else:
        prod_regions = [full.crop((0,        0, prod_right, h))]
        dim_regions  = [full.crop((dim_left, 0, w,          h))]

    product_imgs = [t for r in prod_regions if (t := _trim_whitespace(r))      is not None]
    dim_imgs     = [t for r in dim_regions  if (t := _trim_whitespace_dim(r))  is not None]

    return {"product": product_imgs, "dim": dim_imgs}


def extract_images_from_page(pdf_bytes: bytes, page_num: int) -> list:
    """
    Backward-compatible wrapper — returns product illustration images only.
    Use extract_page_images() for both illustration + dimension drawings.
    """
    return extract_page_images(pdf_bytes, page_num)["product"]


def _trim_whitespace(img: Image.Image, threshold: int = 245) -> Image.Image | None:
    """
    Crop white/near-white borders from a PIL Image, then focus on the
    densest illustration zone to remove text headers / peripheral annotations.
    Returns None if the entire image is blank.
    """
    import numpy as np
    arr = np.array(img)
    # Mask of non-white pixels (any channel below threshold)
    mask = (arr < threshold).any(axis=2)
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    if not rows.any():
        return None
    r_min, r_max = rows.argmax(), len(rows) - rows[::-1].argmax()
    c_min, c_max = cols.argmax(), len(cols) - cols[::-1].argmax()
    # Tight padding — just enough to not clip the illustration edge
    pad = 6
    r_min = max(0, r_min - pad)
    r_max = min(img.height, r_max + pad)
    c_min = max(0, c_min - pad)
    c_max = min(img.width,  c_max + pad)
    trimmed = img.crop((c_min, r_min, c_max, r_max))

    # ── Focus on the densest illustration zone ───────────────────────────────
    # After trimming whitespace the image may still include a text header
    # (product name, article number) above the lamp illustration.
    # We find the vertical window that contains the most dark content —
    # the illustration body — and crop to that zone.
    arr2 = np.array(trimmed)
    h2, w2 = arr2.shape[:2]
    if h2 < 40:
        return trimmed   # too small to bother
    # Row-level darkness fraction (pixels darker than 180 = actual drawn content)
    dark_row = (arr2 < 180).any(axis=2).mean(axis=1)   # shape (h2,)

    # Sliding window: find the contiguous block of rows with the most content.
    # Window size = 55% of image height, so we always keep the majority.
    win = max(20, int(h2 * 0.55))
    best_start, best_score = 0, -1.0
    for y in range(0, h2 - win + 1, 3):
        score = dark_row[y: y + win].sum()
        if score > best_score:
            best_score = score
            best_start = y
    best_end = min(h2, best_start + win)

    # Only apply the zone crop if it meaningfully changes the bounds
    # (skip if we'd be removing less than 8% from either side — not worth it)
    top_skip = best_start
    bot_skip = h2 - best_end
    if top_skip > int(h2 * 0.08) or bot_skip > int(h2 * 0.08):
        zone_pad = 8
        y0 = max(0, best_start - zone_pad)
        y1 = min(h2, best_end + zone_pad)
        trimmed = trimmed.crop((0, y0, w2, y1))

    return trimmed


def _trim_whitespace_dim(img: Image.Image, threshold: int = 248) -> Image.Image | None:
    """
    Light whitespace trim for dimension drawings — keeps more context than the
    product-illustration trim so measurement labels at the edges aren't cut off.
    Does NOT apply the dense-zone crop (the illustration finder would strip away
    the thin dimension arrows and text that are sparse by nature).
    """
    import numpy as np
    arr  = np.array(img)
    mask = (arr < threshold).any(axis=2)
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    if not rows.any():
        return None
    r_min = max(0, rows.argmax() - 12)
    r_max = min(img.height, len(rows) - rows[::-1].argmax() + 12)
    c_min = max(0, cols.argmax() - 12)
    c_max = min(img.width,  len(cols) - cols[::-1].argmax() + 12)
    cropped = img.crop((c_min, r_min, c_max, r_max))
    return cropped if cropped.width > 20 and cropped.height > 20 else None


def _parse_price(price_str: str) -> float | None:
    """
    Parse a price string that may use comma or dot as decimal separator.
    Examples: "14469,00" → 14469.0 | "1.234,50" → 1234.5 | "1234.50" → 1234.5
    """
    s = price_str.strip()
    # Case: European format "1.234,50" — dot as thousands, comma as decimal
    if re.match(r'^\d{1,3}(\.\d{3})+(,\d{1,2})?$', s):
        s = s.replace('.', '').replace(',', '.')
    # Case: "14469,00" — comma as decimal only
    elif re.match(r'^\d+(,\d{1,2})$', s):
        s = s.replace(',', '.')
    # Case: already dot decimal "1234.50"
    else:
        s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def convert_prices(pdf_bytes: bytes, from_currency: str, multiplier: float, to_currency: str) -> bytes:
    """
    Convert prices in a PDF.

    from_currency can be:
      - A symbol like "€", "$", "£"  → finds prices written as €149,00
      - A text label like "RMB", "EUR" → finds standalone numbers on pages
        that contain that label in a column header

    Replaces price text and prepends to_currency symbol/label.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Determine if from_currency is a symbol or a text label
    is_symbol = len(from_currency.strip()) <= 2 and not from_currency.strip().isalpha()

    if is_symbol:
        escaped = re.escape(from_currency.strip())
        # Match: €149 | €149.00 | €1,234.50 | € 149,00
        pattern = re.compile(rf'{escaped}\s*([\d][\d\s,\.]*)', re.UNICODE)
    else:
        # Match standalone numbers that look like prices (4+ digits or decimal numbers)
        # e.g. 14469,00 or 1234.50 — not small numbers like dimensions "400" or weights "2.5"
        pattern = re.compile(r'\b(\d{3,}(?:[.,]\d{2})?)\b')

    for page in doc:
        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        page_text = page.get_text()

        # For label-based currencies, skip pages that don't mention the label
        if not is_symbol and from_currency.upper() not in page_text.upper():
            continue

        redactions = []

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    m = pattern.search(text)
                    if not m:
                        continue

                    price_str = m.group(1) if is_symbol else m.group(0)
                    parsed = _parse_price(price_str)
                    if parsed is None or parsed < 10:  # skip tiny numbers (not prices)
                        continue

                    new_price = parsed * multiplier
                    new_price_str = f"{new_price:,.2f}"
                    new_text = text[:m.start()] + f"{to_currency} {new_price_str}" + text[m.end():]

                    bbox = fitz.Rect(span["bbox"])
                    font_size = span.get("size", 10)
                    c = span.get("color", 0)
                    rgb = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)
                    redactions.append((bbox, new_text, font_size, rgb))

        for bbox, _, _, _ in redactions:
            page.add_redact_annot(bbox, fill=(1, 1, 1))
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        for bbox, new_text, font_size, rgb in redactions:
            page.insert_text(
                (bbox.x0, bbox.y0 + font_size * 0.85),
                new_text,
                fontsize=font_size,
                color=rgb
            )

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()
