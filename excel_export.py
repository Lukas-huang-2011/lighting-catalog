"""Generate Excel files for customer quotes."""

import io
import requests
from PIL import Image
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage


def _fetch_image(url: str):
    try:
        r = requests.get(url, timeout=5)
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def build_excel(products: list, discount: float, include_images: bool = True) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Quote"

    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    alt_fill = PatternFill("solid", fgColor="EBF0FA")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    thin = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin")
    )

    base_cols = [
        ("Image", 18), ("Code", 20), ("Name", 28), ("Description", 40),
        ("Color", 18), ("Light Source", 20), ("CCT", 10),
        ("Dimensions", 18), ("Wattage", 12),
        ("Original Price", 16), ("Currency", 10),
        (f"Discount ({int((1-discount)*100)}%)", 16),
        ("Customer Price", 16), ("Catalog", 22),
    ]

    extra_keys = sorted({k for p in products for k in (p.get("extra_fields") or {}).keys()})
    all_cols = base_cols + [(k.title(), 16) for k in extra_keys]

    for col_idx, (col_name, col_width) in enumerate(all_cols, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = thin
        ws.column_dimensions[get_column_letter(col_idx)].width = col_width
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    for row_idx, product in enumerate(products, start=2):
        # Image
        if include_images:
            images = product.get("product_images") or []
            img_url = images[0]["image_url"] if images else None
            if img_url:
                pil_img = _fetch_image(img_url)
                if pil_img:
                    pil_img.thumbnail((100, 100))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    buf.seek(0)
                    xl_img = XLImage(buf)
                    xl_img.anchor = f"A{row_idx}"
                    ws.add_image(xl_img)

        codes = product.get("codes") or []
        ws.cell(row=row_idx, column=2, value=", ".join(codes))
        ws.cell(row=row_idx, column=3, value=product.get("name") or "")
        ws.cell(row=row_idx, column=4, value=product.get("description") or "")
        ws.cell(row=row_idx, column=5, value=product.get("color") or "")
        ws.cell(row=row_idx, column=6, value=product.get("light_source") or "")
        ef = product.get("extra_fields") or {}
        ws.cell(row=row_idx, column=7, value=ef.get("cct") or product.get("cct") or "")
        ws.cell(row=row_idx, column=8, value=product.get("dimensions") or "")
        ws.cell(row=row_idx, column=9, value=product.get("wattage") or "")

        orig = product.get("price")
        currency = product.get("currency") or ""
        customer_price = round(orig * discount, 2) if orig else None
        ws.cell(row=row_idx, column=10, value=orig)
        ws.cell(row=row_idx, column=11, value=currency)
        ws.cell(row=row_idx, column=12, value=f"× {discount}")
        ws.cell(row=row_idx, column=13, value=customer_price)
        ws.cell(row=row_idx, column=14, value=(product.get("pdfs") or {}).get("name") or "")

        for i, key in enumerate(extra_keys, start=15):
            ws.cell(row=row_idx, column=i, value=ef.get(key) or "")

        fill = alt_fill if row_idx % 2 == 0 else PatternFill()
        for col_idx in range(1, len(all_cols) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            if fill.fill_type:
                cell.fill = fill
            cell.border = thin
            cell.alignment = center if col_idx in (1, 10, 11, 12, 13) else left
        ws.row_dimensions[row_idx].height = 80 if include_images else 20

    ws.auto_filter.ref = ws.dimensions
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
