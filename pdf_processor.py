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
    # Most lighting catalogs place product photos in the left ~38% of the page.
    # A page usually contains 1–2 product families stacked vertically.
    mat = fitz.Matrix(150 / 72, 150 / 72)   # 150 DPI
    pix = page.get_pixmap(matrix=mat, alpha=False)
    full = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    w, h = full.size
    img_col_w = int(w * 0.38)   # left illustration column

    # Detect how many product sections: look for a wide horizontal white band
    # in the middle of the page (separator between two products)
    mid = h // 2
    band_start, band_end = mid - 20, mid + 20
    mid_strip = full.crop((0, band_start, img_col_w, band_end))
    pixels = list(mid_strip.getdata())
    avg_brightness = sum(r + g + b for r, g, b in pixels) / (len(pixels) * 3)

    if avg_brightness > 230:
        # Bright band → two products, return top-half and bottom-half crops
        top = full.crop((0, 0,       img_col_w, mid))
        bot = full.crop((0, mid,     img_col_w, h))
        # Only return halves that have actual content (not all-white)
        for crop in (top, bot):
            cpix = list(crop.getdata())
            brightness = sum(r + g + b for r, g, b in cpix) / (len(cpix) * 3)
            if brightness < 250:   # not blank white
                images.append(crop)
    else:
        # Single product — return full left column
        left = full.crop((0, 0, img_col_w, h))
        images.append(left)

    return images


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
