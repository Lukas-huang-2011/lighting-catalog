#!/usr/bin/env python3
"""
Price Converter — Standalone Desktop App
Drag-and-drop or browse for a PDF, convert all prices, save the result.
Works on macOS and Windows. Requires: pip install PyMuPDF
"""
import os, re, io, sys, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF is required. Install it with:  pip install PyMuPDF")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Price conversion engine (extracted from pdf_processor.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_price(price_str: str):
    s = price_str.strip()
    if re.match(r'^\d{1,3}(\.\d{3})+(,\d{1,2})?$', s):
        s = s.replace('.', '').replace(',', '.')
    elif re.match(r'^\d+(,\d{1,2})$', s):
        s = s.replace(',', '.')
    else:
        s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def _format_price_european(value: float) -> str:
    rounded = round(value, 2)
    int_part = int(rounded)
    dec_part = round((rounded - int_part) * 100)
    int_str = f"{int_part:,}".replace(",", ".")
    return f"{int_str},{dec_part:02d}"


def _pick_fontname(flags: int) -> str:
    bold = bool(flags & 16)
    italic = bool(flags & 2)
    if bold and italic:
        return "Helvetica-BoldOblique"
    if bold:
        return "Helvetica-Bold"
    if italic:
        return "Helvetica-Oblique"
    return "Helvetica"


def _chars_bbox(chars, start, end):
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


def _insert_fitted(page, bbox, text, fsize, rgb, fontname, align="right"):
    fname = "helv"
    for attempt in (fontname, "Helvetica", "helv"):
        try:
            fitz.get_textlength("X", fontname=attempt, fontsize=fsize)
            fname = attempt
            break
        except Exception:
            continue
    try:
        tw = fitz.get_textlength(text, fontname=fname, fontsize=fsize)
    except Exception:
        tw = len(text) * fsize * 0.55

    actual_fsize = fsize
    if align == "right":
        bbox_w = max(1.0, bbox.x1 - bbox.x0)
        if tw > bbox_w * 1.3:
            actual_fsize = max(fsize * 0.6, fsize * (bbox_w * 1.3 / tw))
            try:
                tw = fitz.get_textlength(text, fontname=fname, fontsize=actual_fsize)
            except Exception:
                tw = len(text) * actual_fsize * 0.55
        x = max(bbox.x0, bbox.x1 - tw)
    else:
        x = bbox.x0

    y = bbox.y0 + actual_fsize * 0.85
    page.insert_text((x, y), text, fontname=fname, fontsize=actual_fsize, color=rgb)


def convert_prices(pdf_bytes, from_currency, multiplier, to_currency, progress_cb=None):
    """Convert prices in a PDF. progress_cb(page_num, total) is called per page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    fc = from_currency.strip()
    tc = to_currency.strip()
    esc = re.escape(fc)
    fc_is_symbol = len(fc) <= 2 and not fc.isalpha()

    pat_price = re.compile(r'\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d+[.,]\d{2}')
    pat_label = re.compile(esc, re.IGNORECASE)

    total_pages = len(doc)
    total_converted = 0

    for page_idx, page in enumerate(doc):
        if progress_cb:
            progress_cb(page_idx + 1, total_pages)

        page_text = page.get_text()
        if not fc_is_symbol and fc.upper() not in page_text.upper():
            continue

        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        redactions = []

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if not span_text:
                        continue
                    chars = span.get("chars", [])
                    fsize = span.get("size", 10)
                    flags = span.get("flags", 0)
                    c = span.get("color", 0)
                    rgb = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)
                    fontname = _pick_fontname(flags)

                    for m in pat_price.finditer(span_text):
                        parsed = _parse_price(m.group())
                        if parsed is None or parsed < 5:
                            continue
                        bbox = _chars_bbox(chars, m.start(), m.end())
                        if bbox is None:
                            continue
                        new_str = _format_price_european(parsed * multiplier)
                        redactions.append((bbox, new_str, fsize, rgb, fontname, "right"))
                        total_converted += 1

                    for m in pat_label.finditer(span_text):
                        bbox = _chars_bbox(chars, m.start(), m.end())
                        if bbox is None:
                            continue
                        redactions.append((bbox, tc, fsize, rgb, fontname, "left"))

        if not redactions:
            continue

        for bbox, *_ in redactions:
            page.add_redact_annot(bbox, fill=(1, 1, 1))
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        for bbox, new_text, fsize, rgb, fontname, align in redactions:
            _insert_fitted(page, bbox, new_text, fsize, rgb, fontname, align=align)

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue(), total_converted


# ═══════════════════════════════════════════════════════════════════════════════
# Desktop GUI
# ═══════════════════════════════════════════════════════════════════════════════

class PriceConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Price Converter — 柒点")
        self.root.geometry("560x520")
        self.root.resizable(False, False)

        # Style
        style = ttk.Style()
        style.theme_use("clam")

        # ── Header ───────────────────────────────────────────────────────
        header = tk.Frame(root, bg="#1a1a2e", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="PDF Price Converter", font=("Helvetica", 18, "bold"),
                 bg="#1a1a2e", fg="white").pack(pady=15)

        main = ttk.Frame(root, padding=20)
        main.pack(fill="both", expand=True)

        # ── File selection ───────────────────────────────────────────────
        file_frame = ttk.LabelFrame(main, text="PDF File", padding=10)
        file_frame.pack(fill="x", pady=(0, 10))

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, state="readonly").pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side="right")

        # ── Currency settings ────────────────────────────────────────────
        curr_frame = ttk.LabelFrame(main, text="Conversion Settings", padding=10)
        curr_frame.pack(fill="x", pady=(0, 10))

        row1 = ttk.Frame(curr_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="From currency:").pack(side="left")
        self.from_curr = tk.StringVar(value="€")
        ttk.Entry(row1, textvariable=self.from_curr, width=10).pack(side="right")

        row2 = ttk.Frame(curr_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="To currency:").pack(side="left")
        self.to_curr = tk.StringVar(value="¥")
        ttk.Entry(row2, textvariable=self.to_curr, width=10).pack(side="right")

        row3 = ttk.Frame(curr_frame)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Multiplier:").pack(side="left")
        self.multiplier = tk.StringVar(value="7.5")
        ttk.Entry(row3, textvariable=self.multiplier, width=10).pack(side="right")

        # ── Preview ──────────────────────────────────────────────────────
        preview_frame = ttk.LabelFrame(main, text="Preview", padding=10)
        preview_frame.pack(fill="x", pady=(0, 10))
        self.preview_label = ttk.Label(preview_frame, text="€ 14.469,00  →  ¥ 108.517,50", font=("Courier", 13))
        self.preview_label.pack()

        # Update preview when settings change
        for var in (self.from_curr, self.to_curr, self.multiplier):
            var.trace_add("write", self._update_preview)

        # ── Progress ─────────────────────────────────────────────────────
        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 5))

        self.status_label = ttk.Label(main, text="Select a PDF to get started", foreground="gray")
        self.status_label.pack()

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill="x", pady=(10, 0))

        self.convert_btn = ttk.Button(btn_frame, text="Convert Prices", command=self.start_conversion)
        self.convert_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.save_btn = ttk.Button(btn_frame, text="Save As...", command=self.save_file, state="disabled")
        self.save_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        self.result_bytes = None

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if path:
            self.file_path.set(path)
            pages = fitz.open(path).page_count
            self.status_label.config(text=f"Loaded: {os.path.basename(path)} — {pages} pages", foreground="black")

    def _update_preview(self, *_):
        try:
            m = float(self.multiplier.get())
            fc = self.from_curr.get() or "€"
            tc = self.to_curr.get() or "¥"
            converted = _format_price_european(14469.0 * m)
            self.preview_label.config(text=f"{fc} 14.469,00  →  {tc} {converted}")
        except (ValueError, Exception):
            pass

    def start_conversion(self):
        path = self.file_path.get()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("No file", "Please select a PDF first.")
            return
        try:
            mult = float(self.multiplier.get())
        except ValueError:
            messagebox.showwarning("Invalid multiplier", "Please enter a valid number for the multiplier.")
            return

        self.convert_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_label.config(text="Converting...", foreground="blue")

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        fc = self.from_curr.get()
        tc = self.to_curr.get()

        def run():
            def on_progress(current, total):
                pct = current / total * 100
                self.root.after(0, lambda: self._set_progress(pct, f"Page {current} / {total}"))

            try:
                result, count = convert_prices(pdf_bytes, fc, mult, tc, progress_cb=on_progress)
                self.result_bytes = result
                self.root.after(0, lambda: self._done(count))
            except Exception as e:
                self.root.after(0, lambda: self._error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _set_progress(self, pct, msg):
        self.progress["value"] = pct
        self.status_label.config(text=msg)

    def _done(self, count):
        self.progress["value"] = 100
        self.status_label.config(text=f"Done! {count} prices converted.", foreground="green")
        self.convert_btn.config(state="normal")
        self.save_btn.config(state="normal")

        # Auto-suggest save
        self.save_file()

    def _error(self, msg):
        self.progress["value"] = 0
        self.status_label.config(text=f"Error: {msg}", foreground="red")
        self.convert_btn.config(state="normal")
        messagebox.showerror("Conversion failed", msg)

    def save_file(self):
        if not self.result_bytes:
            return
        orig = os.path.basename(self.file_path.get())
        name, ext = os.path.splitext(orig)
        tc = self.to_curr.get().replace("¥", "RMB").replace("€", "EUR").replace("$", "USD").replace("£", "GBP")
        suggested = f"{name}_converted_{tc}{ext}"

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=suggested
        )
        if path:
            with open(path, "wb") as f:
                f.write(self.result_bytes)
            self.status_label.config(text=f"Saved: {os.path.basename(path)}", foreground="green")
            messagebox.showinfo("Saved", f"Converted PDF saved to:\n{path}")


def main():
    root = tk.Tk()
    PriceConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
