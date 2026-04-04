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
    Scan the full page for the brightest horizontal band between 30-70% height.
    Returns (best_y, brightness).
    brightness > 235 means a clear separator between two stacked product families.
    """
    w, h = full.size
    scan_start = int(h * 0.30)
    scan_end = int(h * 0.70)
    best_y, best_brightness = h // 2, 0.0
    for y in range(scan_start, scan_end, 5):
        strip = full.crop((0, y, w, y + 8))
        pxls = list(strip.getdata())
        bright = sum(r + g + b for r, g, b in pxls) / (len(pxls) * 3)
        if bright > best_brightness:
            best_brightness = bright
            best_y = y + 4
    return best_y, best_brightness


def _find_split_x(img: Image.Image) -> int | None:
    """
    Find where the lamp drawing ends and the spec-text columns begin,
    by scanning for the rightmost clear vertical gap (nearly empty column band)
    in the left 65% of the cropped rect image.
    The lamp silhouette is compact and centred; to its right there is usually
    a white gap before the "Light source / Type / Volt ..." text block starts.
    Returns the x-pixel coordinate to crop to (keep everything left of that point),
    or None if no clear gap is found.
    """
    import numpy as np
    arr = np.array(img)
    h, w = arr.shape[:2]
    # Analyse only the left 80% -- beyond that we're already deep in text territory
    scan_to = max(10, int(w * 0.80))
    # Column-wise fraction of dark pixels (any RGB channel < 200 = drawn content)
    dark_col = (arr[:, :scan_to, :] < 200).any(axis=2).mean(axis=0)  # (scan_to,)
    # Tag each column as "gap" if < 4% of its pixels are dark
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
      A) Single "re" command -> items = [("re", Rect)]
      B) 4-5 line/move commands forming a closed rectangle
         -> items like [("l",p1,p2),("l",p2,p3),("l",p3,p4),("l",p4,p1)]
    Returns a fitz.Rect or None.
    """
    if not items:
        return None
    # Case A: explicit rectangle command (most PDF creators use this)
    if items[0][0] == "re":
        return fitz.Rect(items[0][1])
    # Case B: 4-5 line/moveto segments that together form a closed rectangle
    if not (4 <= len(items) <= 5):
        return None
    pts = []
    for item in items:
        t = item[0]
        if t == "l" and len(item) >= 3:
            pts.append(item[1])  # start-point of the line
            pts.append(item[2])  # end-point of the line
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
    Find the bordered drawing boxes for dimension images.
    Layout assumption:
      - Each product section has a bordered rectangle on the LEFT that contains
        the lamp silhouette + dimension measurement labels.
      - The spec-text columns are to the RIGHT of that box.
      - Therefore the drawing box's right edge is always in the LEFT ~55% of
        the page width -- this is the key position filter that rules out the
        outer product-section border (which spans ~90% of page width).
    Algorithm:
      1. Find all diameter label positions (guaranteed inside drawing boxes).
      2. Collect all rectangular vector paths that end before 55% of page width
         and are big enough to be a drawing box (>= 8% of page in each dim).
      3. Anchor each diameter label to the smallest containing rect -> that IS the box.
      4. Fallback: if no diameter labels found, return all left-side rects sorted by Y.
    Returns up to 2 fitz.Rect objects sorted top -> bottom.
    """
    pw = page.rect.width
    ph = page.rect.height
    # -- 1. Find diameter label positions ----
    label_pts = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if "\u00d8" in txt or "\u00f8" in txt:
                    b = span["bbox"]
                    label_pts.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    # -- 2. Collect rectangle-shaped border strokes in the LEFT zone ----
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
        if r.x1 > pw * 0.58:
            continue
        rect_pool.append(r)
    if not rect_pool:
        return []
    # -- 3. Anchor on diameter labels -> find smallest containing rect ----
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
            key = (round(best.x0), round(best.y0))
            if key not in seen:
                seen.add(key)
                found.append(best)
        if found:
            found.sort(key=lambda r: r.y0)
            return found[:6]
    # -- 4. Fallback: return all qualifying left-zone rects sorted by Y ----
    rect_pool.sort(key=lambda r: r.y0)
    return rect_pool[:6]


def _extend_crop_to_content(full, x0, y0, x1, y1, px_w, px_h):
    """
    Extend a detected rectangle's crop boundaries upward to capture the full
    lamp drawing including ceiling mounts and stems that extend above the
    bordered box in the PDF.
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
    Extract dimension drawings AND product photos from one PDF page.
    Strategy for dimension drawings (best -> fallback):
      1. AI vision -- asks for exact bounding-box percentages of each dimension drawing.
      2. PyMuPDF rect detection -- finds bordered frames via vector drawing data.
      3. Zone fallback -- crops the left ~35% of each product row.
    Strategy for product photos:
      1. AI vision -- asks for bounding boxes of real-life product photos.
         Photos are sorted top-to-bottom so photo[0] = top product, photo[1] = bottom product.
         This matches the product_index from AI extraction.
    Returns:
      {
          'product': [PIL.Image, ...],  # real-life product photos, sorted top-to-bottom
          'dim': [PIL.Image, ...],      # dimension drawings, sorted top-to-bottom
      }
    """
    # -- Render page at 150 DPI (needed by all paths) ----
    full = _render_page_full(pdf_bytes, page_num, dpi=150)
    px_w, px_h = full.size
    dim_imgs = []
    product_imgs = []
    # -- Path 1: AI bounding-box detection ----
    if api_key:
        import ai_extractor as ai
        # -- Dimension drawings --
        boxes = ai.find_dim_boxes(api_key, full)
        if boxes:
            pad = 4
            for b in boxes:
                x0 = max(0, int(b["x0"] / 100 * px_w) - pad)
                y0 = max(0, int(b["y0"] / 100 * px_h) - pad)
                x1 = min(px_w, int(b["x1"] / 100 * px_w) + pad)
                y1 = min(px_h, int(b["y1"] / 100 * px_h) + pad)
                crop = full.crop((x0, y0, x1, y1))
                # Step 1: find the border line and cut off any header text above it
                crop = _crop_to_drawing_box(crop)
                # Step 2: trim remaining whitespace padding
                cleaned = _trim_whitespace_dim(crop)
                if cleaned is not None and cleaned.width > 30 and cleaned.height > 30:
                    dim_imgs.append(cleaned)
        # -- Product photos (real-life images) --
        photo_boxes = ai.find_photo_boxes(api_key, full)
        if photo_boxes:
            pad = 4
            for b in photo_boxes:
                x0 = max(0, int(b["x0"] / 100 * px_w) - pad)
                y0 = max(0, int(b["y0"] / 100 * px_h) - pad)
                x1 = min(px_w, int(b["x1"] / 100 * px_w) + pad)
                y1 = min(px_h, int(b["y1"] / 100 * px_h) + pad)
                crop = full.crop((x0, y0, x1, y1))
                if crop.width > 40 and crop.height > 40:
                    product_imgs.append(crop)
        if dim_imgs:
            return {"product": product_imgs, "dim": dim_imgs}
    # -- Path 2: PyMuPDF vector-rect detection (dim drawings only) ----
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    page_w = page.rect.width
    page_h = page.rect.height
    rects = _find_drawing_rects(page)
    doc.close()
    sx = px_w / page_w
    sy = px_h / page_h
    if rects:
        for r in rects:
            # Convert PDF coords to pixel coords
            rx0 = int(r.x0 * sx)
            ry0 = int(r.y0 * sy)
            rx1 = int(r.x1 * sx)
            ry1 = int(r.y1 * sy)
            # Extend the crop upward to capture ceiling mounts and stems
            ex0, ey0, ex1, ey1 = _extend_crop_to_content(
                full, rx0, ry0, rx1, ry1, px_w, px_h
            )
            crop = full.crop((ex0, ey0, ex1, ey1))
            if crop.width > 40 and crop.height > 40:
                dim_imgs.append(crop)
        if dim_imgs:
            return {"product": product_imgs, "dim": dim_imgs}
    # -- Path 3: zone-based fallback ----
    draw_right = int(px_w * 0.35)
    best_y, best_brightness = _find_split_y(full)
    two_products = best_brightness > 235 and (0.25 * px_h < best_y < 0.75 * px_h)
    regions = (
        [full.crop((0, 0, draw_right, best_y)),
         full.crop((0, best_y, draw_right, px_h))]
        if two_products
        else [full.crop((0, 0, draw_right, px_h))]
    )
    dim_imgs = [t for r in regions if (t := _trim_whitespace_dim(r)) is not None]
    return {"product": product_imgs, "dim": dim_imgs}


def extract_images_from_page(pdf_bytes: bytes, page_num: int) -> list:
    """Backward-compatible wrapper -- returns product illustration images only.
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
    # Tight padding
    pad = 6
    r_min = max(0, r_min - pad)
    r_max = min(img.height, r_max + pad)
    c_min = max(0, c_min - pad)
    c_max = min(img.width, c_max + pad)
    trimmed = img.crop((c_min, r_min, c_max, r_max))
    # -- Focus on the densest illustration zone ----
    arr2 = np.array(trimmed)
    h2, w2 = arr2.shape[:2]
    if h2 < 40:
        return trimmed
    dark_row = (arr2 < 180).any(axis=2).mean(axis=1)
    win = max(20, int(h2 * 0.55))
    best_start, best_score = 0, -1.0
    for y in range(0, h2 - win + 1, 3):
        score = dark_row[y: y + win].sum()
        if score > best_score:
            best_score = score
            best_start = y
    best_end = min(h2, best_start + win)
    top_skip = best_start
    bot_skip = h2 - best_end
    if top_skip > int(h2 * 0.08) or bot_skip > int(h2 * 0.08):
        zone_pad = 8
        y0 = max(0, best_start - zone_pad)
        y1 = min(h2, best_end + zone_pad)
        trimmed = trimmed.crop((0, y0, w2, y1))
    return trimmed


def _crop_to_drawing_box(img: Image.Image,
                         min_run_fraction: float = 0.60,
                         dark_thresh: int = 80) -> Image.Image:
    """
    When an AI bounding box includes section-header text above the bordered drawing
    frame, find the top border line of that frame and crop from there.

    Key insight: a border line is a SOLID horizontal stroke — one very long
    consecutive run of dark pixels.  Text characters have white gaps between
    letters; even a wide bold heading has a max consecutive run of only ~20-40 px,
    well below the 60 % threshold used here.

    Scans rows top-to-bottom; the first row whose longest consecutive run of pixels
    darker than `dark_thresh` spans ≥ 60 % of the image width is the box border.
    """
    import numpy as np
    arr  = np.array(img.convert("L"))
    h, w = arr.shape

    for y in range(h):
        row = arr[y] < dark_thresh           # boolean: True where pixel is dark
        if row.sum() < int(w * 0.4):         # cheap pre-filter: skip sparse rows
            continue
        # Longest consecutive run of True values
        padded = np.r_[False, row, False]
        diffs  = np.diff(padded.view(np.int8))
        starts = np.where(diffs >  0)[0]
        ends   = np.where(diffs <  0)[0]
        if not len(starts):
            continue
        max_run = int((ends - starts).max())
        if max_run / w >= min_run_fraction:
            return img.crop((0, max(0, y - 1), img.width, img.height))

    return img   # no solid border found — return unchanged


def _trim_whitespace_dim(img: Image.Image, threshold: int = 248) -> Image.Image | None:
    """
    Light whitespace trim for dimension drawings -- keeps more context
    than the product-illustration trim so measurement labels at the edges
    aren't cut off.
    """
    import numpy as np
    arr = np.array(img)
    mask = (arr < threshold).any(axis=2)
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    if not rows.any():
        return None
    r_min = max(0, rows.argmax() - 12)
    r_max = min(img.height, len(rows) - rows[::-1].argmax() + 12)
    c_min = max(0, cols.argmax() - 12)
    c_max = min(img.width, len(cols) - cols[::-1].argmax() + 12)
    cropped = img.crop((c_min, r_min, c_max, r_max))
    return cropped if cropped.width > 20 and cropped.height > 20 else None


def _is_currency_header_text(text: str, fc: str) -> bool:
    """
    Return True if `text` is a table column header indicating prices
    (as opposed to an inline price like "€149,00").

    Strips the currency marker and common header words; what remains must
    contain no digits (otherwise it is a price or a product code).

    True examples : "€"  "EUR"  "PRICE EUR"  "List price (€)"  "€/pc"
    False examples: "€149,00"  "149,00 €"  "1.234"
    """
    if not text.strip():
        return False
    if not re.search(re.escape(fc), text, re.IGNORECASE):
        return False
    remainder = re.sub(re.escape(fc), "", text, flags=re.IGNORECASE)
    # Remove common surrounding words and punctuation
    remainder = re.sub(
        r"(?i)(price|list|netto?|gross|msrp|rrp|per|pc|pcs|unit|stk|prijs|preis)",
        "", remainder,
    )
    remainder = re.sub(r"[/\\()\[\]{}\s|_\-,.:;]", "", remainder)
    return not re.search(r"\d", remainder)


def _find_price_column_headers(page, from_currency: str) -> list:
    """
    Scan a page for table column headers that indicate a price column.
    Returns list of dicts: {'x0', 'x1', 'y_below'} — any number whose
    x-centre falls in [x0, x1] and whose y is below y_below is a price.
    """
    fc = from_currency.strip()
    raw = page.get_text("dict")
    price_cols = []
    seen: set = set()
    # horizontal tolerance in PDF points (expands the column's detected x-range)
    TOL = 70

    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            # Check the combined line text first (cheaper)
            line_text = "".join(s.get("text", "") for s in line.get("spans", []))
            if not _is_currency_header_text(line_text, fc):
                continue
            # Anchor on the span that actually contains the currency marker
            for span in line.get("spans", []):
                if not re.search(re.escape(fc), span.get("text", ""), re.IGNORECASE):
                    continue
                bbox = span.get("bbox")
                if not bbox:
                    continue
                # De-duplicate on a 10-pt grid
                key = (round(bbox[0], -1), round(bbox[1], -1))
                if key in seen:
                    continue
                seen.add(key)
                x_centre = (bbox[0] + bbox[2]) / 2
                price_cols.append(
                    {"x0": x_centre - TOL, "x1": x_centre + TOL, "y_below": bbox[3]}
                )
    return price_cols


def _parse_price(price_str: str) -> float | None:
    """
    Parse a price string that may use comma or dot as decimal separator.
    Examples: "14469,00" -> 14469.0 | "1.234,50" -> 1234.5 | "1234.50" -> 1234.5
    """
    s = price_str.strip()
    # Case: European format "1.234,50" -- dot as thousands, comma as decimal
    if re.match(r'^\d{1,3}(\.\d{3})+(,\d{1,2})?$', s):
        s = s.replace('.', '').replace(',', '.')
    # Case: "14469,00" -- comma as decimal only
    elif re.match(r'^\d+(,\d{1,2})$', s):
        s = s.replace(',', '.')
    # Case: already dot decimal "1234.50"
    else:
        s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def _format_price_num(value: float) -> str:
    """
    Format a converted price in European catalog style.
    e.g. 1234.56 -> "1.234,56"
    """
    rounded  = round(value, 2)
    int_part = int(rounded)
    dec_part = round((rounded - int_part) * 100)
    int_str  = f"{int_part:,}".replace(",", ".")   # thousands dot
    return f"{int_str},{dec_part:02d}"


def _pick_fontname(flags: int) -> str:
    """Return the closest built-in PDF font matching the span's bold/italic flags."""
    bold   = bool(flags & 16)
    italic = bool(flags & 2)
    if bold and italic:
        return "Helvetica-BoldOblique"
    if bold:
        return "Helvetica-Bold"
    if italic:
        return "Helvetica-Oblique"
    return "Helvetica"


def _insert_fitted(page, bbox, text: str, fsize: float, rgb: tuple,
                   fontname: str, align: str = "right") -> None:
    """
    Insert `text` near `bbox`, matching the original colour and staying as
    close to the original font size as possible.

    Scaling rules:
      • Try to fit within the bbox at the original font size.
      • If text is wider than the bbox, scale down proportionally — but never
        below 70 % of the original size or 7 pt, whichever is larger.
        It is better to overflow slightly than to be unreadable.
      • align='right'  — right-align prices inside the bbox.
      • align='left'   — left-align labels; they may flow into whitespace
                         on their right (original currency bbox can be tiny).
    """
    fname = "helv"
    for attempt_font in (fontname, "Helvetica", "helv"):
        try:
            fitz.get_textlength("X", fontname=attempt_font, fontsize=fsize)
            fname = attempt_font
            break
        except Exception:
            continue

    def _tw(fs):
        try:
            return fitz.get_textlength(text, fontname=fname, fontsize=fs)
        except Exception:
            return len(text) * fs * 0.55

    bbox_w = max(1.0, bbox.x1 - bbox.x0)
    tw = _tw(fsize)
    actual_fsize = fsize

    if align == "right" and tw > bbox_w:
        # Scale so text fits; never go below 70 % of original or 7 pt
        min_fsize = max(fsize * 0.7, 7.0)
        scaled = fsize * (bbox_w / tw)
        actual_fsize = max(scaled, min_fsize)
        tw = _tw(actual_fsize)

    x = max(bbox.x0, bbox.x1 - tw) if align == "right" else bbox.x0
    y = bbox.y0 + actual_fsize * 0.85

    page.insert_text((x, y), text, fontname=fname,
                     fontsize=actual_fsize, color=rgb)


def _chars_bbox(chars: list, start: int, end: int):
    """
    Return a fitz.Rect covering chars[start:end], or None if no valid bboxes.
    `chars` is the 'chars' list from a PyMuPDF rawdict span.
    """
    rects = []
    for ch in chars[start:end]:
        b = ch.get("bbox")
        if b and len(b) == 4 and b[2] > b[0]:
            rects.append(fitz.Rect(b))
    if not rects:
        return None
    r = rects[0]
    for x in rects[1:]:
        r |= x
    return r


def _span_substr_bbox(span_bbox, span_text: str, start: int, end: int):
    """
    Estimate the bounding box of a substring within a span by proportional
    position.  Used as a fallback when character-level bboxes are unavailable.
    Returns a fitz.Rect.
    """
    if not span_text:
        return None
    total_len = len(span_text)
    if total_len == 0:
        return None
    x0, y0, x1, y1 = span_bbox
    span_w = x1 - x0
    char_w = span_w / total_len
    sub_x0 = x0 + start * char_w
    sub_x1 = x0 + end * char_w
    return fitz.Rect(sub_x0, y0, sub_x1, y1)


def convert_prices(pdf_bytes: bytes, from_currency: str, multiplier: float,
                   to_currency: str, progress_cb=None) -> bytes:
    """
    Convert every price in a PDF from one currency to another and replace the
    currency label/symbol.

    Detection tiers (applied per span, in order; each position claimed once):

    A  Combined unit  — currency + price or price + currency as a SINGLE match.
       Redacted and reinserted as one string so label and number never overlap.
       Handles both decimal prices (149,00) and bare integers (149).
         Prefix:  €149,00  →  ¥19,37    |  €149  →  ¥19,37
         Suffix:  149,00€  →  19,37¥    |  149€  →  19,37¥

    B  Standalone decimal price  — 1.188,00 | 335,00 | 1.234,56
       Right-aligned replacement in the original number's bbox.

    C  Standalone bare integer / thousands number in price context
       (span is in a currency-header column, or the line contains the currency)
       149 | 1.234 → converted.  Skipped if followed by unit letter (W/K/V/A/%).

    D  Standalone currency label  — any remaining occurrence of the marker.

    Redaction boxes are expanded by 1 pt each side to guarantee full coverage.
    Replacement text always uses the original span's colour and font.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    fc  = from_currency.strip()
    tc  = to_currency.strip()
    esc = re.escape(fc)
    n_pages = len(doc)

    fc_is_symbol = len(fc) <= 2 and not fc.isalpha()

    # ── Patterns ─────────────────────────────────────────────────────────────
    _DEC  = r"\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d+[.,]\d{2}"
    _BARE = r"(?:\d{1,3}(?:\.\d{3})+|\d{2,6})"

    # A: combined units (group 1 always captures the numeric part)
    pat_pfx_dec  = re.compile(esc + r"\s*(" + _DEC  + r")",                             re.IGNORECASE)
    pat_sfx_dec  = re.compile(r"(" + _DEC  + r")\s*" + esc,                             re.IGNORECASE)
    pat_pfx_bare = re.compile(esc + r"\s*(" + _BARE + r")(?!\d|[.,]\d|[A-Za-z°%])",    re.IGNORECASE)
    pat_sfx_bare = re.compile(r"(?<![.\d])(" + _BARE + r")(?!\d|[.,]\d|[A-Za-z°%])\s*" + esc, re.IGNORECASE)
    # B: standalone decimal
    pat_price = re.compile(_DEC)
    # C: standalone bare
    pat_bare  = re.compile(r"(?<![.\d])" + _BARE + r"(?!\d|[.,]\d|[A-Za-z°%])")
    # D: standalone label
    pat_label = re.compile(esc, re.IGNORECASE)

    def _get_bbox(chars, span_bbox, span_text, start, end):
        b = _chars_bbox(chars, start, end)
        if b is None and span_bbox:
            b = _span_substr_bbox(span_bbox, span_text, start, end)
        return b

    for page_idx, page in enumerate(doc):
        if progress_cb:
            progress_cb(page_idx / n_pages, f"Page {page_idx + 1} / {n_pages}")

        page_text = page.get_text()
        if not fc_is_symbol and fc.upper() not in page_text.upper():
            continue

        price_cols = _find_price_column_headers(page, fc)
        raw        = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        redactions = []   # (bbox, new_text, fsize, rgb, fontname, align)

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                line_spans = line.get("spans", [])
                if not line_spans:
                    continue
                line_text = "".join(s.get("text", "") for s in line_spans)
                line_has_currency = bool(pat_label.search(line_text))

                for span in line_spans:
                    span_text = span.get("text", "")
                    if not span_text:
                        continue

                    chars    = span.get("chars", [])
                    fsize    = span.get("size", 10)
                    flags    = span.get("flags", 0)
                    c        = span.get("color", 0)
                    rgb      = ((c >> 16 & 255) / 255,
                                (c >>  8 & 255) / 255,
                                (c       & 255) / 255)
                    fontname = _pick_fontname(flags)
                    span_bbox = span.get("bbox", None)

                    span_in_col = False
                    if span_bbox and price_cols:
                        sx = (span_bbox[0] + span_bbox[2]) / 2
                        sy = (span_bbox[1] + span_bbox[3]) / 2
                        for col in price_cols:
                            if col["x0"] <= sx <= col["x1"] and sy > col["y_below"]:
                                span_in_col = True
                                break

                    claimed: set = set()

                    # ── A: combined currency+price units ─────────────────────
                    for pat, order in (
                        (pat_pfx_dec,  "prefix"),
                        (pat_sfx_dec,  "suffix"),
                        (pat_pfx_bare, "prefix"),
                        (pat_sfx_bare, "suffix"),
                    ):
                        for m in pat.finditer(span_text):
                            if any(i in claimed for i in range(m.start(), m.end())):
                                continue
                            price_str = m.group(1)
                            parsed = _parse_price(price_str)
                            if parsed is None or parsed < 1:
                                continue
                            bbox = _get_bbox(chars, span_bbox, span_text,
                                             m.start(), m.end())
                            if bbox is None:
                                continue
                            new_price = _format_price_num(parsed * multiplier)
                            new_text  = (tc + new_price) if order == "prefix" \
                                        else (new_price + tc)
                            claimed.update(range(m.start(), m.end()))
                            redactions.append(
                                (bbox, new_text, fsize, rgb, fontname, "left"))

                    # ── B: standalone decimal prices ──────────────────────────
                    for m in pat_price.finditer(span_text):
                        if any(i in claimed for i in range(m.start(), m.end())):
                            continue
                        parsed = _parse_price(m.group())
                        if parsed is None or parsed < 1:
                            continue
                        bbox = _get_bbox(chars, span_bbox, span_text,
                                         m.start(), m.end())
                        if bbox is None:
                            continue
                        claimed.update(range(m.start(), m.end()))
                        redactions.append(
                            (bbox, _format_price_num(parsed * multiplier),
                             fsize, rgb, fontname, "right"))

                    # ── C: bare integers in price context ─────────────────────
                    if span_in_col or line_has_currency:
                        for m in pat_bare.finditer(span_text):
                            if any(i in claimed for i in range(m.start(), m.end())):
                                continue
                            parsed = _parse_price(m.group())
                            if parsed is None or parsed < 1:
                                continue
                            bbox = _get_bbox(chars, span_bbox, span_text,
                                             m.start(), m.end())
                            if bbox is None:
                                continue
                            claimed.update(range(m.start(), m.end()))
                            redactions.append(
                                (bbox, _format_price_num(parsed * multiplier),
                                 fsize, rgb, fontname, "right"))

                    # ── D: standalone currency labels ─────────────────────────
                    for m in pat_label.finditer(span_text):
                        if any(i in claimed for i in range(m.start(), m.end())):
                            continue
                        bbox = _get_bbox(chars, span_bbox, span_text,
                                         m.start(), m.end())
                        if bbox is None:
                            continue
                        redactions.append(
                            (bbox, tc, fsize, rgb, fontname, "left"))

        # ── Fallback: page.search_for() ───────────────────────────────────────
        if not redactions:
            for m in pat_price.finditer(page_text):
                parsed = _parse_price(m.group())
                if parsed is None or parsed < 1:
                    continue
                for bbox in page.search_for(m.group(), quads=False):
                    redactions.append(
                        (bbox, _format_price_num(parsed * multiplier),
                         10, (0, 0, 0), "Helvetica", "right"))
            for m in pat_label.finditer(page_text):
                for bbox in page.search_for(m.group(), quads=False):
                    redactions.append(
                        (bbox, tc, 10, (0, 0, 0), "Helvetica", "left"))

        if not redactions:
            continue

        # 1. Whiteout — expand each box by 1 pt to guarantee full coverage
        for bbox, *_ in redactions:
            pad = fitz.Rect(bbox.x0 - 1, bbox.y0 - 1, bbox.x1 + 1, bbox.y1 + 1)
            page.add_redact_annot(pad, fill=(1, 1, 1))
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # 2. Insert converted text
        for bbox, new_text, fsize, rgb, fontname, align in redactions:
            _insert_fitted(page, bbox, new_text, fsize, rgb, fontname, align=align)

    if progress_cb:
        progress_cb(1.0, "Finalizing…")

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()
