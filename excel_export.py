"""
Generate Excel files for customer quotes.
Includes all product fields + discounted prices.
"""

import io
import requests
from PIL import Image
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage


def _fetch_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=5)
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def build_excel(products: list, discount: float, include_images: bool = True) -> bytes:
    """
    Build an Excel file for a customer quote.

    Args:
        products: list of product dicts from the database
        discount: multiplier, e.g. 0.7 means customer pays 70% (30% off)
        include_images: whether to embed product images

    Returns:
        Excel file as bytes
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Quote"

    # ── Styles ────────────────────────────────────────────────────────────────
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    alt_fill = PatternFill("solid", fgColor="EBF0FA")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    thin = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin")
    )

    # ── Columns ───────────────────────────────────────────────────────────────
    base_cols = [
        ("Image", 18),
        ("Code(s)", 20),
        ("Name", 28),
        ("Description", 40),
        ("Color", 16),
        ("Light Source", 16),
        ("Dimensions", 18),
        ("Wattage", 12),
        ("Original Price", 16),
        ("Currency", 10),
        (f"Discount ({int((1-discount)*100)}%)", 16),
        ("Customer Price", 16),
        ("Catalog", 22),
    ]

    # Collect extra field keys across all products
    extra_keys = set()
    for p in products:
        ef = p.get("extra_fields") or {}
        extra_keys.update(ef.keys())
    extra_keys = sorted(extra_keys)

    all_cols = base_cols + [(k.title(), 16) for k in extra_keys]

    # ── Header row ────────────────────────────────────────────────────────────
    for col_idx, (col_name, col_width) in enumerate(all_cols, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = thin
        ws.column_dimensions[get_column_letter(col_idx)].width = col_width

    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    # ── Data rows ─────────────────────────────────────────────────────────────
    img_col = 1  # column A for images

    for row_idx, product in enumerate(products, start=2):
        row_height = 80 if include_images else 20

        # Image
        img_cell = ws.cell(row=row_idx, column=1, value="")
        if include_images:
            images = product.get("product_images") or []
            img_url = images[0]["image_url"] if images else None
            if img_url:
                pil_img = _fetch_image(img_url)
                if pil_img:
                    pil_img.thumbnail((100, 100))
                    img_buf = io.BytesIO()
                    pil_img.save(img_buf, format="PNG")
                    img_buf.seek(0)
                    xl_img = XLImage(img_buf)
                    xl_img.anchor = f"A{row_idx}"
                    ws.add_image(xl_img)

        # Codes
        codes = product.get("codes") or []
        ws.cell(row=row_idx, column=2, value=", ".join(codes))

        # Text fields
        fields = [
            product.get("name") or "",
            product.get("description") or "",
            product.get("color") or "",
            product.get("light_source") or "",
            product.get("dimensions") or "",
            product.get("wattage") or "",
        ]
        for i, val in enumerate(fields, start=3):
            ws.cell(row=row_idx, column=i, value=val)

        # Prices
        orig = product.get("price")
        currency = product.get("currency") or ""
        customer_price = round(orig * discount, 2) if orig else None

        ws.cell(row=row_idx, column=9, value=orig)
        ws.cell(row=row_idx, column=10, value=currency)
        ws.cell(row=row_idx, column=11, value=f"× {discount}")
        ws.cell(row=row_idx, column=12, value=customer_price)

        # Catalog name
        pdf_info = product.get("pdfs") or {}
        ws.cell(row=row_idx, column=13, value=pdf_info.get("name") or "")

        # Extra fields
        ef = product.get("extra_fields") or {}
        for i, key in enumerate(extra_keys, start=14):
            ws.cell(row=row_idx, column=i, value=ef.get(key) or "")

        # Row styling
        fill = alt_fill if row_idx % 2 == 0 else PatternFill()
        for col_idx in range(1, len(all_cols) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            if fill.fill_type:
                cell.fill = fill
            cell.border = thin
            cell.alignment = center if col_idx in (1, 9, 10, 11, 12) else left

        ws.row_dimensions[row_idx].height = row_height

    # ── Auto-filter ───────────────────────────────────────────────────────────
    ws.auto_filter.ref = ws.dimensions

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
