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


def extract_images_from_page(pdf_bytes: bytes, page_num: int) -> list:
    """
    Extract product images from a single PDF page.

    Strategy:
    1. Try extracting embedded raster images (works for photo-based catalogs).
    2. If none found (vector-drawing catalogs like Martinelli), render the page
       at 150 DPI and crop the left ~38% which is where product illustrations
       typically live. Splits into top/bottom halves to get one image per
       product family when two products share a page.

    Returns a list of PIL Images.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    images = []
    seen = set()

    # ── Try embedded raster images first ─────────────────────────────────────
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if img.width > 100 and img.height > 100:
                images.append(img)
        except Exception:
            pass

    if images:
        doc.close()
        return images

    # ── Fallback: render page and crop product illustration area ─────────────
    # Most lighting catalogs place product illustrations in the left ~38% of
    # the page. 1–2 product families are stacked vertically per page.
    mat = fitz.Matrix(150 / 72, 150 / 72)   # 150 DPI for detail
    pix = page.get_pixmap(matrix=mat, alpha=False)
    full = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    w, h = full.size
    img_col_w = int(w * 0.40)   # left illustration column width

    # Find the horizontal separator between two products by scanning the FULL
    # page width (not just the left column) for the whitest horizontal strip.
    # Scan between 30% and 70% of the page height in 5-pixel steps.
    scan_start = int(h * 0.30)
    scan_end   = int(h * 0.70)
    best_y, best_brightness = h // 2, 0.0

    for y in range(scan_start, scan_end, 5):
        strip = full.crop((0, y, w, y + 8))
        pxls  = list(strip.getdata())
        bright = sum(r + g + b for r, g, b in pxls) / (len(pxls) * 3)
        if bright > best_brightness:
            best_brightness = bright
            best_y = y + 4   # centre of the strip

    # If the whitest band is clearly bright (separator line), split there.
    # Otherwise treat the page as a single-product page.
    if best_brightness > 235:
        regions = [
            full.crop((0, 0,      img_col_w, best_y)),
            full.crop((0, best_y, img_col_w, h)),
        ]
    else:
        regions = [full.crop((0, 0, img_col_w, h))]

    # Trim white borders from each region and keep only non-blank ones.
    for region in regions:
        trimmed = _trim_whitespace(region)
        if trimmed is not None:
            images.append(trimmed)

    return images


def _trim_whitespace(img: Image.Image, threshold: int = 245) -> Image.Image | None:
    """
    Crop white/near-white borders from a PIL Image.
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
    # Add a small padding so the image doesn't feel too tight
    pad = 10
    r_min = max(0, r_min - pad)
    r_max = min(img.height, r_max + pad)
    c_min = max(0, c_min - pad)
    c_max = min(img.width,  c_max + pad)
    return img.crop((c_min, r_min, c_max, r_max))


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
