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


def render_pages(pdf_bytes: bytes, dpi: int = 150):
    """Yield each page of a PDF as a PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        yield img
    doc.close()


def extract_images_from_page(pdf_bytes: bytes, page_num: int) -> list:
    """
    Extract embedded images from a single PDF page.
    Returns list of PIL Images (only those > 80x80 px to skip icons/logos).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    images = []
    seen = set()

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if img.width > 80 and img.height > 80:
                images.append(img)
        except Exception:
            pass

    doc.close()
    return images


def convert_prices(pdf_bytes: bytes, from_symbol: str, multiplier: float, to_symbol: str) -> bytes:
    """
    Convert all prices in a PDF from one currency to another.
    Finds price text, covers it with white, writes new value.
    Preserves all images and page layout.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    escaped = re.escape(from_symbol)
    # Matches: €149, €149.00, €1,234.50, € 149
    pattern = re.compile(rf'{escaped}\s*([\d][\d\s,\.]*)', re.UNICODE)

    for page in doc:
        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        redactions = []

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    m = pattern.search(text)
                    if not m:
                        continue
                    price_str = m.group(1).replace(",", "").replace(" ", "")
                    try:
                        old_price = float(price_str)
                        new_price = old_price * multiplier
                        new_text = text[:m.start()] + f"{to_symbol}{new_price:,.2f}" + text[m.end():]
                        bbox = fitz.Rect(span["bbox"])
                        font_size = span.get("size", 10)
                        # Decode color int → RGB floats
                        c = span.get("color", 0)
                        rgb = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)
                        redactions.append((bbox, new_text, font_size, rgb))
                    except (ValueError, TypeError):
                        pass

        # Apply redactions (white boxes) then write new text
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
