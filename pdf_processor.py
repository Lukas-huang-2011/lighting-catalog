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


def _find_split_x(img: Image.Image) -> int | None:
    """
    Find where the lamp drawing ends and the spec-text columns begin, by scanning
    for the rightmost clear vertical gap (nearly empty column band) in the left
    65 % of the cropped rect image.

    The lamp silhouette is compact and centred; to its right there is usually a
    white gap before the "Light source / Type / Volt …" text block starts.

    Returns the x-pixel coordinate to crop to (keep everything left of that point),
    or None if no clear gap is found.
    """
    import numpy as np
    arr  = np.array(img)
    h, w = arr.shape[:2]

    # Analyse only the left 80 % — beyond that we're already deep in text territory
    scan_to = max(10, int(w * 0.80))

    # Column-wise fraction of dark pixels (any RGB channel < 200 = drawn content)
    dark_col = (arr[:, :scan_to, :] < 200).any(axis=2).mean(axis=0)   # (scan_to,)

    # Tag each column as "gap" if < 4 % of its pixels are dark
    is_gap = dark_col < 0.04

    # Collect runs of consecutive gap-columns
    gaps, gap_start = [], None
    for x in range(scan_to):
        if is_gap[x]:
            if gap_start is None:
                gap_start = x
        else:
            if gap_start is not None:
                gaps.append((gap_start, x))
                gap_start = None
    if gap_start is not None:
        gaps.append((gap_start, scan_to))

    # Keep only gaps that are past the left margin and at least 2 px wide
    min_start = int(w * 0.15)
    valid = [(s, e) for s, e in gaps if s >= min_start and (e - s) >= 2]

    if not valid:
        return None

    # Return the centre of the rightmost valid gap
    s, e = valid[-1]
    return (s + e) // 2


def _rect_from_path_items(items, pw, ph) -> "fitz.Rect | None":
    """
    Try to extract a rectangle from a path's item list.

    Handles two cases:
      A) Single "re" command  → items = [("re", Rect)]
      B) 4–5 line/move commands forming a closed rectangle
         → items like [("l",p1,p2),("l",p2,p3),("l",p3,p4),("l",p4,p1)]

    Returns a fitz.Rect or None.
    """
    if not items:
        return None

    # Case A: explicit rectangle command (most PDF creators use this)
    if items[0][0] == "re":
        return fitz.Rect(items[0][1])

    # Case B: 4–5 line/moveto segments that together form a closed rectangle
    if not (4 <= len(items) <= 5):
        return None

    pts = []
    for item in items:
        t = item[0]
        if t == "l" and len(item) >= 3:
            pts.append(item[1])   # start-point of the line
            pts.append(item[2])   # end-point of the line
        elif t == "m" and len(item) >= 2:
            pts.append(item[1])

    if len(pts) < 4:
        return None

    xs = sorted({round(p.x) for p in pts})
    ys = sorted({round(p.y) for p in pts})

    # A rectangle has exactly 2 distinct x values and 2 distinct y values
    if len(xs) != 2 or len(ys) != 2:
        return None

    return fitz.Rect(xs[0], ys[0], xs[1], ys[1])


def _find_drawing_rects(page) -> list:
    """
    Find the bordered drawing boxes for 尺寸 images.

    Layout assumption:
      • Each product section has a bordered rectangle on the LEFT that contains
        the lamp silhouette + Ø/height measurement labels.
      • The spec-text columns are to the RIGHT of that box.
      • Therefore the drawing box's right edge is always in the LEFT ~55 % of
        the page width — this is the key position filter that rules out the
        outer product-section border (which spans ~90 % of page width).

    Algorithm:
      1. Find all Ø label positions (guaranteed inside drawing boxes).
      2. Collect all rectangular vector paths that end before 55 % of page width
         and are big enough to be a drawing box (≥ 8 % of page in each dim).
      3. Anchor each Ø label to the smallest containing rect → that IS the box.
      4. Fallback: if no Ø labels found, return all left-side rects sorted by Y.

    Returns up to 2 fitz.Rect objects sorted top → bottom.
    """
    pw = page.rect.width
    ph = page.rect.height

    # ── 1. Find Ø label positions ─────────────────────────────────────────────
    label_pts = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if "Ø" in txt or "ø" in txt:
                    b = span["bbox"]
                    label_pts.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))

    # ── 2. Collect rectangle-shaped border strokes in the LEFT zone ───────────
    rect_pool = []
    for path in page.get_drawings():
        items = path.get("items", [])
        r = _rect_from_path_items(items, pw, ph)
        if r is None or r.is_empty or r.is_infinite:
            continue
        # Size filter: must be a substantial box (not a tick mark or table rule)
        if r.width < pw * 0.08 or r.height < ph * 0.08:
            continue
        # *** Position filter ***
        # Drawing boxes sit in the LEFT half of the page.
        # The outer product-section border (drawing + text columns combined)
        # extends to ~90 % of page width — exclude it by capping r.x1 here.
        if r.x1 > pw * 0.58:
            continue
        rect_pool.append(r)

    if not rect_pool:
        return []

    # ── 3. Anchor on Ø labels → find smallest containing rect ─────────────────
    if label_pts:
        found, seen = [], set()
        for px, py in label_pts:
            containing = [
                r for r in rect_pool
                if r.x0 <= px <= r.x1 and r.y0 <= py <= r.y1
            ]
            if not containing:
                continue
            best = min(containing, key=lambda r: r.width * r.height)
            key  = (round(best.x0), round(best.y0))
            if key not in seen:
                seen.add(key)
                found.append(best)
        if found:
            found.sort(key=lambda r: r.y0)
            return found[:2]

    # ── 4. Fallback: return all qualifying left-zone rects sorted by Y ─────────
    rect_pool.sort(key=lambda r: r.y0)
    return rect_pool[:2]



def _extend_crop_to_content(full, x0, y0, x1, y1, px_w, px_h):
    """
    Extend a detected rectangle's crop boundaries upward to capture the full
    lamp drawing including ceiling mounts and stems that extend above the
    bordered box in the PDF.

    The PDF's bordered rectangle often only encloses the lower portion of the
    lamp drawing (the shade/body + dimension labels). The ceiling mount plate
    and the hanging stem extend above that box. This function scans the
    rendered page image upward from the rect's top edge to find where the
    drawing content actually starts.

    Returns (new_x0, new_y0, new_x1, new_y1).
    """
    import numpy as np
    arr = np.array(full)

    # Use the horizontal range of the rect (with some margin)
    margin_x = int((x1 - x0) * 0.1)
    scan_x0 = max(0, x0 - margin_x)
    scan_x1 = min(px_w, x1 + margin_x)

    # Scan upward from y0, looking for rows with dark pixels (drawn content)
    new_y0 = y0
    # Maximum upward extension: go up to 80% of the rect's own height above it
    max_extend = int((y1 - y0) * 0.8)
    scan_top = max(0, y0 - max_extend)

    for scan_y in range(y0 - 1, scan_top - 1, -1):
        row_strip = arr[scan_y, scan_x0:scan_x1, :]
        dark_pixels = (row_strip < 200).any(axis=1)
        dark_fraction = dark_pixels.mean()
        if dark_fraction > 0.005:
            new_y0 = scan_y
        else:
            # Bridge small gaps (e.g. between stem and ceiling mount)
            found_more = False
            for ahead in range(1, min(26, scan_y - scan_top + 1)):
                ahead_row = arr[scan_y - ahead, scan_x0:scan_x1, :]
                ahead_dark = (ahead_row < 200).any(axis=1).mean()
                if ahead_dark > 0.005:
                    found_more = True
                    break
            if not found_more:
                break

    # Add padding around the final crop
    pad = 10
    new_y0 = max(0, new_y0 - pad)
    new_x0 = max(0, x0 - pad)
    new_x1 = min(px_w, x1 + pad)
    new_y1 = min(px_h, y1 + pad)

    return new_x0, new_y0, new_x1, new_y1



def extract_page_images(pdf_bytes: bytes, page_num: int, api_key: str = None) -> dict:
    """
    Extract dimension drawings from one PDF page for the 尺寸 column.

    Strategy (best → fallback):
      1. AI vision  — asks GLM-4V-Flash for exact bounding-box percentages of
         each dimension drawing.  Precise, layout-independent.  Requires api_key.
      2. PyMuPDF rect detection — finds bordered frames via vector drawing data.
      3. Zone fallback — crops the left ~35 % of each product row.

    Real-life product photos (图片) are NOT extracted — must be uploaded manually.

    Returns:
        {
          'product': [],               # always empty — 图片 is manual
          'dim':     [PIL.Image, ...], # 1–2 尺寸 drawings, index 0=top, 1=bottom
        }
    """
    # ── Render page at 150 DPI (needed by all paths) ─────────────────────────
    full   = _render_page_full(pdf_bytes, page_num, dpi=150)
    px_w, px_h = full.size

    # ── Path 1: AI bounding-box detection ────────────────────────────────────
    if api_key:
        import ai_extractor as ai
        boxes = ai.find_dim_boxes(api_key, full)
        if boxes:
            pad = 6
            dim_imgs = []
            for b in boxes:
                x0 = max(0,    int(b["x0"] / 100 * px_w) - pad)
                y0 = max(0,    int(b["y0"] / 100 * px_h) - pad)
                x1 = min(px_w, int(b["x1"] / 100 * px_w) + pad)
                y1 = min(px_h, int(b["y1"] / 100 * px_h) + pad)
                crop = full.crop((x0, y0, x1, y1))
                if crop.width > 30 and crop.height > 30:
                    dim_imgs.append(crop)
            if dim_imgs:
                return {"product": [], "dim": dim_imgs}

    # ── Path 2: PyMuPDF vector-rect detection ─────────────────────────────────
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    page   = doc[page_num]
    page_w = page.rect.width
    page_h = page.rect.height
    rects  = _find_drawing_rects(page)
    doc.close()

    sx = px_w / page_w
    sy = px_h / page_h

    if rects:
        dim_imgs = []
        for r in rects:
            # Convert PDF coords to pixel coords
            rx0 = int(r.x0 * sx)
            ry0 = int(r.y0 * sy)
            rx1 = int(r.x1 * sx)
            ry1 = int(r.y1 * sy)

            # Extend the crop upward to capture ceiling mounts and stems
            # that extend above the bordered rectangle in the PDF
            ex0, ey0, ex1, ey1 = _extend_crop_to_content(
                full, rx0, ry0, rx1, ry1, px_w, px_h
            )

            crop = full.crop((ex0, ey0, ex1, ey1))
            if crop.width > 40 and crop.height > 40:
                dim_imgs.append(crop)
        if dim_imgs:
            return {"product": [], "dim": dim_imgs}

    # ── Path 3: zone-based fallback ───────────────────────────────────────────
    draw_right = int(px_w * 0.35)
    best_y, best_brightness = _find_split_y(full)
    two_products = best_brightness > 235 and (0.25 * px_h < best_y < 0.75 * px_h)
    regions = (
        [full.crop((0, 0, draw_right, best_y)), full.crop((0, best_y, draw_right, px_h))]
        if two_products else
        [full.crop((0, 0, draw_right, px_h))]
    )
    dim_imgs = [t for r in regions if (t := _trim_whitespace_dim(r)) is not None]
    return {"product": [], "dim": dim_imgs}


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
