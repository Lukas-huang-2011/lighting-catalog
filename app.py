"""
Lighting Catalog App v3
• Live search with autocomplete and brand display
• Upload PDFs and extract products with AI
• Convert prices between currencies
• Search by product code or image
• Generate customer quotes as Excel files
"""

import re
import io
import time
import threading
import uuid as _uuid
import streamlit as st
from PIL import Image

import database as db
import pdf_processor as pdf
import ai_extractor as ai
import image_search as imgs
import excel_export as xl


# ═══════════════════════════════════════════════════════════════════════════════
# Background job queue
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_jobs():
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}


def _new_job(job_type: str, filename: str) -> str:
    _ensure_jobs()
    job_id = _uuid.uuid4().hex[:8]
    st.session_state.jobs[job_id] = {
        "id":              job_id,
        "type":            job_type,   # "extract" | "convert"
        "filename":        filename,
        "status":          "starting", # starting | running | done | error
        "progress":        0.0,
        "message":         "Starting…",
        "result_bytes":    None,
        "result_filename": None,
        "error":           None,
        "ts":              time.time(),
        "cancel":          False,
    }
    return job_id


def _bg_extract(jobs, job_id, pdf_bytes, filename, page_count, dpi, extract_imgs):
    """Background thread — AI extraction, saves directly to Supabase."""
    try:
        jobs[job_id]["status"] = "running"
        ai_client = ai.get_client()
        cl        = db.get_client()

        jobs[job_id]["message"] = "Uploading PDF…"
        file_url = db.upload_pdf(cl, pdf_bytes, filename)
        pdf_id   = db.create_pdf_record(cl, filename, file_url, page_count)

        total  = 0
        errors = []
        for page_num, page_img in enumerate(pdf.render_pages(pdf_bytes, dpi=dpi)):
            if jobs[job_id].get("cancel"):
                jobs[job_id]["status"]  = "error"
                jobs[job_id]["message"] = "⛔ Cancelled"
                return
            jobs[job_id]["progress"] = (page_num + 1) / page_count
            jobs[job_id]["message"]  = f"Page {page_num + 1} / {page_count}"

            products         = ai.extract_products_from_page(ai_client, page_img, page_num)
            page_photo_index = {}
            page_dim_index   = {}
            if extract_imgs and products:
                try:
                    raw = pdf.extract_page_images(pdf_bytes, page_num, api_key=ai_client)
                    for pidx, pil_img in enumerate(raw.get("product", [])[:4]):
                        try:
                            url  = db.upload_image(cl, pil_img)
                            hsh  = imgs.compute_hash(pil_img)
                            page_photo_index[pidx] = (url, hsh)
                        except Exception as e:
                            errors.append(str(e))
                    for pidx, pil_img in enumerate(raw.get("dim", [])[:4]):
                        try:
                            url  = db.upload_image(cl, pil_img)
                            hsh  = imgs.compute_hash(pil_img)
                            page_dim_index[pidx] = (url, hsh)
                        except Exception as e:
                            errors.append(str(e))
                except Exception as e:
                    errors.append(str(e))

            for prod in products:
                try:
                    prod_id = db.save_product(cl, pdf_id, prod, page_num)
                    total  += 1
                    if not prod.get("is_accessory"):
                        pidx = prod.get("product_index", 0)
                        if pidx in page_photo_index:
                            u, h = page_photo_index[pidx]
                            db.save_product_image(cl, prod_id, u, h, "product")
                        elif pidx in page_dim_index:
                            u, h = page_dim_index[pidx]
                            db.save_product_image(cl, prod_id, u, h, "product")
                        if pidx in page_dim_index:
                            u, h = page_dim_index[pidx]
                            db.save_product_image(cl, prod_id, u, h, "dim")
                except Exception as e:
                    errors.append(str(e))

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["message"]  = f"✅ {total} products from {page_count} pages"
    except Exception as exc:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = "Failed"
        jobs[job_id]["error"]   = str(exc)


def _bg_convert(jobs, job_id, pdf_bytes, filename, from_sym, multiplier, to_sym):
    """Background thread — price conversion."""
    try:
        jobs[job_id]["status"]  = "running"
        jobs[job_id]["message"] = "Converting prices…"

        def _cb(pct, msg):
            jobs[job_id]["progress"] = pct
            jobs[job_id]["message"]  = msg

        result = pdf.convert_prices(pdf_bytes, from_sym, multiplier, to_sym,
                                    progress_cb=_cb)
        jobs[job_id]["status"]          = "done"
        jobs[job_id]["progress"]        = 1.0
        jobs[job_id]["message"]         = "✅ Done — download ready"
        jobs[job_id]["result_bytes"]    = result
        jobs[job_id]["result_filename"] = f"converted_{filename}"
    except Exception as exc:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = "Failed"
        jobs[job_id]["error"]   = str(exc)


def _render_jobs_sidebar(jobs: dict) -> bool:
    """Render job status panel. Returns True if any job is still active."""
    if not jobs:
        return False
    st.divider()
    st.caption("**⚙️ Jobs**")
    active  = False
    ordered = sorted(jobs.values(), key=lambda j: j["ts"], reverse=True)[:6]
    for job in ordered:
        status = job["status"]
        fname  = job["filename"]
        short  = fname if len(fname) <= 24 else fname[:21] + "…"
        label  = "📥 Extract" if job["type"] == "extract" else "💱 Convert"
        if status in ("starting", "running"):
            active = True
            st.caption(f"🔄 **{label}** · {short}")
            st.progress(job["progress"], text=job["message"])
            if st.button("⛔ Cancel", key=f"cancel_{job['id']}", use_container_width=True):
                job["cancel"] = True
                job["status"]  = "error"
                job["message"] = "⛔ Cancelling…"
                st.rerun()
        elif status == "done":
            st.caption(f"✅ **{label}** · {short}")
            st.caption(job["message"])
            if job["type"] == "convert" and job.get("result_bytes"):
                st.download_button(
                    "⬇️ Download converted PDF",
                    data=job["result_bytes"],
                    file_name=job["result_filename"],
                    mime="application/pdf",
                    key=f"dl_{job['id']}",
                    use_container_width=True,
                )
        elif status == "error":
            st.caption(f"❌ **{label}** · {short}")
            st.caption(f"Error: {(job.get('error') or '')[:80]}")
    return active

st.set_page_config(page_title="柒点 · 灯具目录", page_icon="💡", layout="wide")

st.markdown("""
<style>
/* ══════════════════════════════════════════════════════════════════
   KEYFRAMES
══════════════════════════════════════════════════════════════════ */
@keyframes qs-fadeup   { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
@keyframes qs-fadedown { from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:translateY(0)} }
@keyframes qs-fadein   { from{opacity:0} to{opacity:1} }
@keyframes qs-scalein  { from{opacity:0;transform:scale(0.95)} to{opacity:1;transform:scale(1)} }
@keyframes qs-slidein  { from{opacity:0;transform:translateX(-14px)} to{opacity:1;transform:translateX(0)} }
@keyframes rd-breathe  { 0%,100%{transform:scale(0.93);opacity:0.5} 50%{transform:scale(1.03);opacity:1} }

/* ── Base ───────────────────────────────────────────────────────── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background-color: #0a0a0a !important; }
[data-testid="stHeader"] { background-color: #0a0a0a !important; border-bottom: 1px solid #1c1c1c; }
[data-testid="stSidebar"],
section[data-testid="stSidebarContent"] { background-color: #0e0e0e !important; border-right: 1px solid #1c1c1c !important; }

/* ── Page entrance (triggers on every page switch) ──────────────── */
.block-container {
  padding-top: 1.5rem;
  animation: qs-fadeup 0.38s cubic-bezier(.22,.68,0,1.1) both;
}

/* Staggered entrance for top-level page elements */
[data-testid="stVerticalBlock"] > div.element-container {
  animation: qs-fadeup 0.3s cubic-bezier(.22,.68,0,1.1) both;
}
[data-testid="stVerticalBlock"] > div.element-container:nth-child(1)  { animation-delay:.04s }
[data-testid="stVerticalBlock"] > div.element-container:nth-child(2)  { animation-delay:.08s }
[data-testid="stVerticalBlock"] > div.element-container:nth-child(3)  { animation-delay:.12s }
[data-testid="stVerticalBlock"] > div.element-container:nth-child(4)  { animation-delay:.16s }
[data-testid="stVerticalBlock"] > div.element-container:nth-child(5)  { animation-delay:.20s }
[data-testid="stVerticalBlock"] > div.element-container:nth-child(6)  { animation-delay:.23s }
[data-testid="stVerticalBlock"] > div.element-container:nth-child(n+7){ animation-delay:.26s }

/* Sidebar slides in from the left */
section[data-testid="stSidebarContent"] {
  animation: qs-slidein 0.4s cubic-bezier(.22,.68,0,1.1) both;
}

/* ── Typography ────────────────────────────────────────────────── */
body, p, span, div, label, .stMarkdown { color: #dcdcdc; }
h1, h2, h3, h4 { color: #ffffff !important; }
[data-testid="stSidebar"] * { color: #cccccc !important; }
.stCaption p, small { color: #666 !important; }

/* ── Inputs — focus glow ────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
  background-color: #141414 !important;
  color: #f0f0f0 !important;
  border: 1px solid #2c2c2c !important;
  border-radius: 6px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: #505050 !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.05) !important;
  outline: none !important;
}

/* ── Buttons — lift on hover, press on click ────────────────────── */
.stButton > button {
  background-color: #1a1a1a !important;
  color: #dedede !important;
  border: 1px solid #333 !important;
  border-radius: 8px !important;
  transition: background-color 0.18s ease, border-color 0.18s ease,
              color 0.18s ease, transform 0.14s ease,
              box-shadow 0.18s ease !important;
}
.stButton > button:hover {
  background-color: #242424 !important;
  border-color: #666 !important;
  color: #fff !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 5px 14px rgba(255,255,255,0.05) !important;
}
.stButton > button:active {
  transform: scale(0.96) translateY(0) !important;
  box-shadow: none !important;
}
.stButton > button[kind="primary"] {
  background-color: #efefef !important;
  color: #0a0a0a !important;
  border: none !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
  background-color: #ffffff !important;
  box-shadow: 0 5px 18px rgba(255,255,255,0.12) !important;
}
.stButton > button[kind="primary"]:active { transform: scale(0.96) !important; }

/* ── Cards — hover lift + border brighten ───────────────────────── */
.product-card {
  border: 1px solid #1e1e1e !important;
  border-radius: 12px !important;
  padding: 16px !important;
  margin-bottom: 12px !important;
  background: #101010 !important;
  transition: border-color 0.22s ease, transform 0.22s ease,
              box-shadow 0.22s ease !important;
  animation: qs-fadeup 0.32s cubic-bezier(.22,.68,0,1.1) both;
}
.product-card:hover {
  border-color: #383838 !important;
  transform: translateY(-3px) !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
}
.badge {
  display:inline-block; background:#e8e8e8; color:#0a0a0a;
  border-radius:6px; padding:2px 8px; font-size:0.8em; margin:2px; font-weight:600;
}
.brand-tag {
  display:inline-block; background:#2a2a2a; color:#e0e0e0;
  border-radius:6px; padding:2px 8px; font-size:0.8em; margin:2px; font-weight:600;
}

/* ── Expanders — content fades in on open ───────────────────────── */
[data-testid="stExpander"] {
  background-color: #101010 !important;
  border: 1px solid #1e1e1e !important;
  border-radius: 8px !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stExpander"]:has(details[open]) {
  border-color: #2e2e2e !important;
}
[data-testid="stExpander"] summary {
  color: #d0d0d0 !important;
  transition: color 0.15s ease, background-color 0.15s ease !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
}
[data-testid="stExpander"] summary:hover {
  color: #ffffff !important;
  background-color: #1a1a1a !important;
}
[data-testid="stExpanderDetails"] {
  animation: qs-fadeup 0.24s ease both;
}

/* ── Metrics — scale in + hover lift ───────────────────────────── */
[data-testid="metric-container"] {
  background: #141414; border: 1px solid #1e1e1e;
  border-radius: 8px; padding: 10px;
  animation: qs-scalein 0.3s ease both;
  transition: transform 0.2s ease, box-shadow 0.2s ease,
              border-color 0.2s ease;
}
[data-testid="metric-container"]:hover {
  transform: scale(1.03);
  box-shadow: 0 6px 18px rgba(255,255,255,0.04);
  border-color: #2e2e2e;
}
[data-testid="stMetricLabel"] p { color: #777 !important; }
[data-testid="stMetricValue"] { color: #fff !important; }

/* ── Alerts / notifications slide in ───────────────────────────── */
[data-testid="stAlert"] {
  animation: qs-fadeup 0.28s ease both;
  border-radius: 8px !important;
}
[data-testid="stNotification"] {
  animation: qs-fadedown 0.3s cubic-bezier(.22,.68,0,1.1) both;
}

/* ── Dividers / Progress ───────────────────────────────────────── */
hr, [data-testid="stDivider"] { border-color: #1e1e1e !important; }
.stProgress > div > div > div {
  background-color: #d0d0d0 !important;
  transition: width 0.4s cubic-bezier(.22,.68,0,1.1) !important;
}

/* ── File uploader — hover glow ─────────────────────────────────── */
[data-testid="stFileUploader"] {
  background-color: #101010 !important;
  border-color: #2a2a2a !important;
  border-radius: 8px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #444 !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.03) !important;
}

/* ── Select / Radio ────────────────────────────────────────────── */
.stSelectbox > div > div {
  background-color: #141414 !important;
  color: #f0f0f0 !important;
  border-color: #2c2c2c !important;
  transition: border-color 0.15s ease !important;
}
.stSelectbox > div > div:hover { border-color: #444 !important; }
[data-testid="stRadio"] label { color: #ccc !important; }

/* ── Images — smooth hover scale ───────────────────────────────── */
[data-testid="stImage"] img {
  transition: transform 0.22s ease, box-shadow 0.22s ease;
  border-radius: 6px;
}
[data-testid="stImage"] img:hover {
  transform: scale(1.03);
  box-shadow: 0 6px 20px rgba(0,0,0,0.55);
}

/* ── Spinner ────────────────────────────────────────────────────── */
[data-testid="stSpinner"] { animation: qs-fadein 0.3s ease both; }

/* ── Logo ───────────────────────────────────────────────────────── */
.rd-logo {
  display:flex; flex-direction:column; align-items:center;
  padding:28px 0 16px; gap:10px; cursor:default; user-select:none;
}
.rd-mark {
  width:160px; height:45px;
  animation: rd-breathe 3.5s ease-in-out infinite;
  display:block;
}
.rd-name {
  font-size:28px; font-weight:900; color:#ffffff; letter-spacing:8px;
  font-family:'PingFang SC','Noto Sans SC','Microsoft YaHei',sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_pil_from_url(url: str) -> "Image.Image | None":
    """Download an image from a URL and return a PIL Image, or None on failure."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as resp:
            return Image.open(io.BytesIO(resp.read())).copy()
    except Exception:
        return None


def extract_brand(pdf_name: str) -> str:
    """Extract brand name from PDF filename. e.g. 'Martinelli_luce_2025.pdf' → 'Martinelli Luce'"""
    name = re.sub(r'\.(pdf|PDF)$', '', pdf_name)
    name = re.sub(r'[\-_]', ' ', name)
    name = re.sub(r'\b(20\d{2}|19\d{2}|price.?list|catalog|catalogue|pricelist)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.title()


def _render_cards(products: list, show_similarity: bool = False):
    for p in products:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 3])
        with col_img:
            images = p.get("product_images") or []
            # Prefer the "product" type image; fall back to the first available
            display_url = None
            for img in images:
                if img.get("image_description") == "product" and img.get("image_url"):
                    display_url = img["image_url"]
                    break
            if not display_url and images:
                display_url = images[0].get("image_url")
            if display_url:
                try:
                    st.image(display_url, use_container_width=True)
                except Exception:
                    st.caption("Image unavailable")
            else:
                st.caption("No image")
        with col_info:
            # Brand + codes + page row
            pdf_info = p.get("pdfs") or {}
            brand = extract_brand(pdf_info.get("name") or "")
            codes = p.get("codes") or []
            page_num = p.get("page_number")
            tags = ""
            if brand:
                tags += f'<span class="brand-tag">📦 {brand}</span> '
            tags += " ".join(f'<span class="badge">{c}</span>' for c in codes)
            if page_num is not None:
                tags += f' <span style="color:#888;font-size:0.8em;">· PDF page {page_num + 1}</span>'
            st.markdown(tags, unsafe_allow_html=True)

            if show_similarity and p.get("similarity"):
                st.caption(f"Match: {p['similarity']}%")
            if p.get("name"):
                st.subheader(p["name"])

            cols3 = st.columns(3)
            fields = [
                ("Color", p.get("color")),
                ("Light Source", p.get("light_source")),
                ("CCT", (p.get("extra_fields") or {}).get("cct") or p.get("cct")),
                ("Dimensions", p.get("dimensions")),
                ("Wattage", p.get("wattage")),
                ("Price", f"{p.get('currency','')} {p.get('price')}" if p.get("price") else None),
            ]
            shown = [(k, v) for k, v in fields if v]
            for i, (k, v) in enumerate(shown):
                with cols3[i % 3]:
                    st.metric(k, v)
            if p.get("description"):
                with st.expander("Description"):
                    st.write(p["description"])
            ef = p.get("extra_fields") or {}
            display_ef = {k: v for k, v in ef.items() if k != "cct" and v}
            if display_ef:
                with st.expander("More specifications"):
                    for k, v in display_ef.items():
                        st.write(f"**{k.title()}:** {v}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()


_ensure_jobs()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="rd-logo">
      <svg class="rd-mark" viewBox="0 0 200 50" xmlns="http://www.w3.org/2000/svg">
        <!-- q -->
        <path d="M18,25 A13,13 0 1,1 18,24.9 Z M18,14 A11,11 0 1,0 18,13.9 Z" fill="#fff"/>
        <rect x="27" y="30" width="4" height="14" fill="#fff"/>
        <!-- q -->
        <path d="M56,25 A13,13 0 1,1 56,24.9 Z M56,14 A11,11 0 1,0 56,13.9 Z" fill="#fff"/>
        <rect x="65" y="30" width="4" height="14" fill="#fff"/>
        <!-- o -->
        <path d="M94,25 A13,13 0 1,1 94,24.9 Z M94,14 A11,11 0 1,0 94,13.9 Z" fill="#fff"/>
        <!-- d -->
        <path d="M127,25 A13,13 0 1,1 127,24.9 Z M127,14 A11,11 0 1,0 127,13.9 Z" fill="#fff"/>
        <rect x="136" y="2" width="4" height="36" fill="#fff"/>
        <!-- d -->
        <path d="M160,25 A13,13 0 1,1 160,24.9 Z M160,14 A11,11 0 1,0 160,13.9 Z" fill="#fff"/>
        <rect x="169" y="2" width="4" height="36" fill="#fff"/>
      </svg>
      <div class="rd-name">柒点</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    _nav_options = [
        "📤 Upload & Extract",
        "🔄 Convert Prices",
        "🔍 Search by Code",
        "🖼️ Search by Image",
        "💰 Pricing & Export",
        "📚 Manage Catalogs",
        "🛠️ Debug & Test"
    ]
    _nav_idx = 0
    if st.session_state.get("_redirect"):
        try:
            _nav_idx = _nav_options.index(st.session_state._redirect)
        except ValueError:
            _nav_idx = 0
        del st.session_state._redirect
    page = st.radio("Navigation", _nav_options, index=_nav_idx, label_visibility="collapsed")
    st.divider()
    client = db.get_client()
    catalogs = db.list_pdfs(client)
    st.caption(f"**{len(catalogs)}** catalog(s) loaded")
    for c in catalogs[:8]:
        brand = extract_brand(c['name'])
        st.caption(f"• {brand} ({c.get('page_count','?')} pages)")

    # ── Live job status — visible on every page ────────────────────────────
    has_active = _render_jobs_sidebar(st.session_state.jobs)

# While a job is running, sleep briefly then rerun so the progress bar
# refreshes automatically — no browser JS or user clicks needed.
# Using Python-side sleep+rerun is more reliable than JS timers, which
# browsers throttle or freeze when the tab is in the background.
if has_active:
    time.sleep(1)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload & Extract
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload & Extract":
    st.header("📤 Upload Catalog & Extract Products")

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        pdf_bytes = uploaded.read()
        page_count = pdf.get_page_count(pdf_bytes)
        brand_guess = extract_brand(uploaded.name)
        st.info(f"**{uploaded.name}** — {page_count} pages · Detected brand: **{brand_guess}**")

        col1, col2 = st.columns(2)
        with col1:
            extract_images_flag = st.checkbox("Extract product images", value=True)
        with col2:
            dpi = st.select_slider("Render quality", [100, 150, 200], value=100,
                                   help="100 recommended — uses less memory and is fast enough for AI reading")

        if st.button("🚀 Start Extraction in Background", type="primary"):
            job_id   = _new_job("extract", uploaded.name)
            jobs_ref = st.session_state.jobs
            t = threading.Thread(
                target=_bg_extract,
                args=(jobs_ref, job_id, pdf_bytes, uploaded.name,
                      page_count, dpi, extract_images_flag),
                daemon=True,
            )
            t.start()
            # Auto-navigate to Search page so user is free to work
            st.session_state._redirect = "🔍 Search by Code"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Convert Prices
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Convert Prices":
    st.header("🔄 Convert Prices in a PDF")
    uploaded_conv = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_conv:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original currency in the PDF**")
            from_currency = st.text_input(
                "Currency symbol or label to find",
                value="€",
                help="e.g. € · $ · £ · RMB · EUR · USD — all price formats are detected automatically"
            )
        with col2:
            st.markdown("**Convert to**")
            to_currency = st.text_input("New currency label/symbol", value="€")
            multiplier = st.number_input("Multiplier", min_value=0.0001, value=0.13, step=0.01,
                                         help="New price = original × multiplier")

        st.caption("All price formats are detected automatically: symbol on each price (€149,00 or 149,00€), and column-header mode (header says RMB / EUR, bare numbers below).")
        st.info(f"**Example:** {from_currency} 14469,00 → {to_currency} {pdf._format_price_num(14469.0 * multiplier)}")

        if st.button("🔄 Start Conversion in Background", type="primary"):
            pdf_bytes_conv = uploaded_conv.read()
            job_id   = _new_job("convert", uploaded_conv.name)
            jobs_ref = st.session_state.jobs
            t = threading.Thread(
                target=_bg_convert,
                args=(jobs_ref, job_id, pdf_bytes_conv, uploaded_conv.name,
                      from_currency, multiplier, to_currency),
                daemon=True,
            )
            t.start()
            # Auto-navigate to Search page so user is free to work
            st.session_state._redirect = "🔍 Search by Code"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Search by Code  (live / autocomplete)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Search by Code":
    st.header("🔍 Search by Product Code")
    st.caption("Results appear as you type — no need to press Enter.")

    query = st.text_input("Start typing a product code or name…",
                          placeholder="e.g. 21019  or  AVRO  or  Martinelli")

    if query and len(query) >= 2:
        results = db.search_by_code(client, query)
        if not results:
            st.warning(f"No products found matching **{query}**")
        else:
            st.success(f"**{len(results)}** result(s) for **{query}**")
            _render_cards(results)
    elif query:
        st.caption("Keep typing…")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Search by Image
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Search by Image":
    st.header("🖼️ Search by Image")
    uploaded_img = st.file_uploader("Upload a photo of a light fitting", type=["jpg","jpeg","png","webp"])

    if uploaded_img:
        query_img = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(query_img, caption="Your image", use_container_width=True)
        with col2:
            threshold = st.slider("Sensitivity", 5, 40, 20)
            if st.button("🔍 Find Matches", type="primary"):
                with st.spinner("Comparing against all catalog images…"):
                    all_hashes = db.get_all_image_hashes(client)
                    matches = imgs.find_similar(query_img, all_hashes, threshold=threshold)
                if not matches:
                    st.warning("No similar images found. Try increasing sensitivity.")
                else:
                    st.success(f"Found **{len(matches)}** match(es)")
                    prod_results = []
                    for m in matches:
                        p = m.get("products")
                        if p:
                            p["product_images"] = [{"image_url": m.get("image_url")}]
                            p["similarity"] = m.get("similarity_score", 0)
                            prod_results.append(p)
                    _render_cards(prod_results, show_similarity=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Pricing & Export
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Pricing & Export":
    st.header("💰 Customer Pricing & Excel Export")
    st.caption("Fills your order template automatically — just enter the product codes.")

    # ── Order info ────────────────────────────────────────────────────────────
    with st.expander("📋 Order Details", expanded=True):
        oi_col1, oi_col2 = st.columns(2)
        with oi_col1:
            order_number   = st.text_input("Order Number", placeholder="e.g. 2602FF014")
            customer_name  = st.text_input("Customer Name")
        with oi_col2:
            contact_person = st.text_input("Contact Person")
            phone          = st.text_input("Phone")

    # ── Products & discount ───────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        codes_input = st.text_area("Product codes (one per line)", height=180,
                                   placeholder="21019/DIM/AR\n21019/DIM/AZ\n40189/BI")
    with col2:
        discount = st.number_input("Discount factor", min_value=0.01, max_value=1.0,
                                   value=0.45, step=0.01,
                                   help="e.g. 0.45 means customer pays 45% of list price")
        st.metric("Customer pays", f"{round(discount*100,0):.0f}% of list price")
        default_qty = st.number_input("Default quantity", min_value=1, value=1, step=1)

    if st.button("🔍 Look Up Products", type="primary"):
        codes = [c.strip() for c in codes_input.splitlines() if c.strip()]
        if not codes:
            st.warning("Please enter at least one product code.")
            st.stop()

        with st.spinner(f"Looking up {len(codes)} code(s)…"):
            products = db.get_products_by_codes(client, codes)

        if not products:
            st.error("None of the codes were found in the database. Have you uploaded and extracted a catalog yet?")
            st.stop()

        not_found = [c for c in codes if not any(
            c.upper() in [x.upper() for x in (p.get("codes") or [])] for p in products
        )]
        if not_found:
            st.warning(f"Not found in database: {', '.join(not_found)}")

        st.success(f"Found **{len(products)}** product(s). Set quantities below, then download.")

        # Build editable preview table
        import pandas as pd
        preview_rows = []
        for p in products:
            orig = p.get("price")
            cust = round(orig * discount, 2) if orig else None
            preview_rows.append({
                "Code":     ", ".join(p.get("codes") or []),
                "Brand":    extract_brand((p.get("pdfs") or {}).get("name") or ""),
                "Name":     p.get("name") or "",
                "Color":    p.get("color") or "",
                "List Price": orig,
                "Currency": p.get("currency") or "",
                "Customer Price": cust,
                "Qty":      int(default_qty),
            })
        df = pd.DataFrame(preview_rows)
        edited = st.data_editor(df, use_container_width=True,
                                column_config={"Qty": st.column_config.NumberColumn(min_value=1, step=1)},
                                hide_index=True)

        # Attach qty and discount to each product before export
        export_products = []
        for i, p in enumerate(products):
            qty = int(edited.iloc[i]["Qty"]) if i < len(edited) else int(default_qty)
            p["_qty"]      = qty
            p["_discount"] = discount
            export_products.append(p)

        order_info = {
            "order_number":   order_number   or None,
            "customer_name":  customer_name  or None,
            "contact_person": contact_person or None,
            "phone":          phone          or None,
        }

        with st.spinner("Filling order template…"):
            # Fetch product and dim images from the stored product_images records
            xl_prod_imgs: dict = {}
            xl_dim_imgs:  dict = {}
            for i, p in enumerate(export_products):
                if p.get("is_accessory"):
                    continue
                for img_rec in (p.get("product_images") or []):
                    desc = img_rec.get("image_description") or ""
                    url  = img_rec.get("image_url") or ""
                    if not url:
                        continue
                    if desc == "product" and i not in xl_prod_imgs:
                        pil = _fetch_pil_from_url(url)
                        if pil:
                            xl_prod_imgs[i] = pil
                    elif desc == "dim" and i not in xl_dim_imgs:
                        pil = _fetch_pil_from_url(url)
                        if pil:
                            xl_dim_imgs[i] = pil
            excel_bytes = xl.build_excel_from_template(
                export_products,
                order_info=order_info,
                product_images=xl_prod_imgs or None,
                dim_images=xl_dim_imgs or None,
            )

        st.download_button(
            "⬇️ Download Filled Order Template",
            data=excel_bytes,
            file_name="order_quote.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Manage Catalogs
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Manage Catalogs":
    st.header("📚 Manage Catalogs")
    catalogs = db.list_pdfs(client)
    if not catalogs:
        st.info("No catalogs uploaded yet.")
    else:
        for cat in catalogs:
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                brand = extract_brand(cat['name'])
                st.write(f"**{brand}** — {cat['name']}")
                st.caption(f"{cat.get('page_count','?')} pages · {cat['uploaded_at'][:10]}")
            with col2:
                if cat.get("file_url"):
                    st.link_button("View PDF", cat["file_url"])
            with col3:
                if st.button("🗑️ Delete", key=f"del_{cat['id']}"):
                    db.delete_pdf(client, cat["id"])
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Debug & Test
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛠️ Debug & Test":
    st.header("🛠️ Debug & Test")
    st.markdown("Use this to diagnose issues with extraction.")

    st.subheader("1. Database check")
    try:
        pdfs = db.list_pdfs(client)
        products_res = client.table("products").select("id", count="exact").execute()
        images_res = client.table("product_images").select("id", count="exact").execute()
        col1, col2, col3 = st.columns(3)
        col1.metric("Catalogs", len(pdfs))
        col2.metric("Products", products_res.count or 0)
        col3.metric("Images", images_res.count or 0)
        st.success("✅ Database tables exist and are accessible.")
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        st.warning("You may not have run the supabase_setup.sql yet. Go to Supabase → SQL Editor and run it.")

    st.divider()
    st.subheader("2. Test AI connection")
    if st.button("🔍 Test Zhipu AI connection"):
        import requests as req
        api_key = st.secrets.get("ZHIPU_API_KEY", "")
        if not api_key:
            st.error("ZHIPU_API_KEY not set in Streamlit secrets.")
        else:
            r = req.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "glm-4v-flash", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
                timeout=15
            )
            if r.status_code == 200:
                st.success("✅ Zhipu AI connection works! Model: glm-4v-flash (free)")
            else:
                st.error(f"❌ {r.status_code}: {r.text[:300]}")

    st.divider()
    st.subheader("3. Test AI extraction on one page")

    # ── Shared PDF uploader — used by sections 3, 4 and 5 ─────────────────────
    test_pdf  = st.file_uploader("Upload the PDF to test", type=["pdf"], key="debug_pdf")
    test_page = st.number_input("Page number (0 = first page, try 11 for product pages)", min_value=0, value=11)

    if test_pdf:
        pdf_bytes = test_pdf.read()
        page_count = pdf.get_page_count(pdf_bytes)
        page_num   = min(int(test_page), page_count - 1)

        # Render & show the page image once (shared)
        page_img = pdf.render_single_page(pdf_bytes, page_num, dpi=100)
        st.image(page_img, caption=f"Page {page_num + 1} of {page_count}", use_container_width=True)

        # ── 3. AI product extraction ───────────────────────────────────────────
        if st.button("🤖 Run AI product extraction"):
            ai_client = ai.get_client()
            with st.spinner("Sending to Zhipu AI — takes ~15 s…"):
                debug_result = ai.extract_products_debug(ai_client, page_img)
            if debug_result.get("error"):
                st.error(f"❌ Error: {debug_result['error']}")
            with st.expander("📄 Raw AI response"):
                st.text(debug_result.get("raw_response") or "No response")
            result = debug_result.get("parsed", [])
            if result:
                st.success(f"✅ Found **{len(result)} product(s)**")
                for i, prod in enumerate(result):
                    with st.expander(f"Product {i+1}: {prod.get('name','?')} — {prod.get('codes',[])}"):
                        st.json(prod)
                st.session_state["debug_products"] = result
                st.session_state["debug_pdf_name"] = test_pdf.name
            else:
                st.error("❌ 0 products found. Check raw response above.")

        st.divider()
        # ── 4. Image extraction ────────────────────────────────────────────────
        st.subheader("4. Test 尺寸 image extraction on this page")
        st.markdown(
            "Extracts **dimension drawings** (lamp silhouettes with measurement labels) "
            "from the left portion of the page → auto-embeds into the 尺寸 column. "
            "Index 0 = top product, index 1 = bottom product. "
            "Real-life 图片 photos are added manually in section 5."
        )

        if st.button("🖼️ Extract 尺寸 drawings from this page"):
            _api_key = st.secrets.get("ZHIPU_API_KEY", "") or None
            with st.spinner("Asking AI to locate dimension drawings…"):
                result = pdf.extract_page_images(pdf_bytes, page_num, api_key=_api_key)
            dim_imgs = result["dim"]
            st.session_state["debug_dim_images"] = dim_imgs
            st.session_state["debug_images"]     = []   # 图片 is always manual

            if not dim_imgs:
                st.warning("⚠️ No dimension drawings found on this page. Check the page layout.")
            else:
                st.success(f"📐 {len(dim_imgs)} dimension drawing(s) extracted → will auto-embed in 尺寸 column")
                cols = st.columns(len(dim_imgs))
                for idx, img in enumerate(dim_imgs):
                    cols[idx].image(img, caption=f"尺寸 {idx+1}  {img.width}×{img.height}px", use_container_width=True)

        elif st.session_state.get("debug_dim_images"):
            dim_imgs = st.session_state["debug_dim_images"]
            st.info(f"Cached: {len(dim_imgs)} dimension drawing(s). Re-click to refresh.")
            cols = st.columns(len(dim_imgs))
            for idx, img in enumerate(dim_imgs):
                cols[idx].image(img, caption=f"尺寸 {idx+1}", use_container_width=True)

        st.divider()
        # ── 5. Excel export ────────────────────────────────────────────────────
        st.subheader("5. Test Excel export")
        products_for_xl  = st.session_state.get("debug_products", [])
        pdf_name_for_xl  = st.session_state.get("debug_pdf_name", "")
        images_for_xl    = st.session_state.get("debug_images", [])      # product illustrations
        dim_images_for_xl= st.session_state.get("debug_dim_images", [])  # dimension drawings

        if not products_for_xl:
            st.info("▶ Run **section 3** first to extract products, then come back here.")
        else:
            from collections import OrderedDict

            # ── Type-keyword → Chinese (also used in excel_export, kept here for UI preview)
            _TYPE_KW = [
                ("pendant","吊灯"),("suspension","吊灯"),("chandelier","吊灯"),("hanging","吊灯"),
                ("wall","壁灯"),("sconce","壁灯"),("aplique","壁灯"),
                ("table","台灯"),("desk","台灯"),
                ("floor","落地灯"),
                ("ceiling","吸顶灯"),("flush","吸顶灯"),("plafon","吸顶灯"),
                ("spot","射灯"),("spotlight","射灯"),
                ("downlight","筒灯"),("recessed","筒灯"),
                ("track","轨道灯"),
                ("strip","灯带"),("linear","线条灯"),("profile","线条灯"),
                ("outdoor","户外灯"),("exterior","户外灯"),
                ("garden","庭院灯"),("street","路灯"),
                ("panel","面板灯"),("bollard","地埋灯"),
            ]
            def _auto_zh(text: str) -> str:
                lower = (text or "").lower()
                for kw, zh in _TYPE_KW:
                    if kw in lower:
                        return zh
                return ""

            def _brand_of(prod, fallback):
                info = prod.get("pdfs") or {}
                raw  = info.get("name") or prod.get("brand") or fallback
                return raw.replace(".pdf","").replace(".PDF","").replace("_"," ").title()

            # ── Order info ─────────────────────────────────────────────────────
            st.markdown("**Order information:**")
            col_a, col_b = st.columns(2)
            xi_order_num = col_a.text_input("订单号 Order number",    key="xi_order_num")
            xi_customer  = col_a.text_input("客户名称 Customer name", key="xi_customer")
            xi_contact   = col_b.text_input("联系人 Contact person",  key="xi_contact")
            xi_phone     = col_b.text_input("联系电话 Phone",         key="xi_phone")

            xi_delivery_default = st.text_input(
                "到货时间 Default delivery time (applies to all products)",
                value="现货", key="xi_delivery_default",
            )

            # ── Group by brand ─────────────────────────────────────────────────
            brands_order, by_brand = [], OrderedDict()
            for i, prod in enumerate(products_for_xl):
                b = _brand_of(prod, pdf_name_for_xl)
                if b not in by_brand:
                    by_brand[b] = []
                    brands_order.append(b)
                by_brand[b].append(i)

            per_product = [None] * len(products_for_xl)

            st.markdown("**Products by brand** — adjust brand discount, then fine-tune each product:")
            st.markdown(
                "<small style='color:gray'>"
                "Colour and category are auto-filled from PDF. "
                "尺寸 dimension drawings are embedded automatically from the PDF. "
                "Upload real-life 图片 photos manually if needed.</small>",
                unsafe_allow_html=True,
            )

            for brand in brands_order:
                indices = by_brand[brand]
                st.markdown(f"---\n**🏷 {brand}**")

                brand_disc = st.number_input(
                    f"Brand discount for {brand}  (e.g. 0.85 = 15% off list price)",
                    min_value=0.0, max_value=1.0, value=1.0, step=0.05, format="%.2f",
                    key=f"brand_disc_{brand}",
                )

                hc = st.columns([4, 1, 1])
                hc[0].markdown("**Product / Code / Price**")
                hc[1].markdown("**数量 Qty**")
                hc[2].markdown("**折扣 Disc**")

                for i in indices:
                    prod      = products_for_xl[i]
                    codes_str = ", ".join(str(c) for c in prod.get("codes", []))
                    name_str  = prod.get("name", "?")
                    price_str = f"¥{prod.get('price', '—')}"

                    col = st.columns([4, 1, 1])
                    col[0].markdown(f"{i+1}. **{name_str}**  `{codes_str}`  {price_str}")
                    qty  = col[1].number_input("", min_value=0, value=1, key=f"qty_{i}",
                                               label_visibility="collapsed")
                    disc = col[2].number_input("", min_value=0.0, max_value=1.0,
                                               value=float(brand_disc), step=0.05, format="%.2f",
                                               key=f"disc_{i}", label_visibility="collapsed")

                    with st.expander(f"  ↳ Details & images for #{i+1}", expanded=True):
                        dc = st.columns([2, 2, 3])

                        # 颜色: pre-filled from PDF, fallback to "如图"
                        color = dc[0].text_input(
                            "颜色 Color",
                            value=prod.get("color") or "如图",
                            key=f"color_{i}", placeholder="如图",
                        )

                        delivery = dc[1].text_input(
                            "到货时间 Delivery",
                            value=xi_delivery_default,
                            key=f"delivery_{i}", placeholder="现货",
                        )

                        # 种类: auto-detect from type field FIRST (most reliable),
                        # then description, then product name as last resort
                        raw_desc = prod.get("description") or ""
                        auto_cat = (_auto_zh(prod.get("type", ""))
                                    or _auto_zh(raw_desc)
                                    or _auto_zh(prod.get("name", "")))
                        category = dc[2].text_input(
                            "种类 Category",
                            value=auto_cat or raw_desc,
                            key=f"category_{i}", placeholder="e.g. 吊灯",
                        )

                        # 图片: manual upload of real-life product photo
                        st.markdown("**图片** (real-life product photo — upload manually):")
                        custom_file = st.file_uploader(
                            "Upload 图片 (JPG / PNG)",
                            type=["jpg", "jpeg", "png"], key=f"custom_img_{i}",
                        )
                        custom_pil = Image.open(custom_file).convert("RGB") if custom_file else None
                        if custom_pil:
                            st.image(custom_pil, width=120, caption="图片 ✓")
                        img_idx = -1   # unused; custom_pil is the only source

                    per_product[i] = {
                        "qty": qty, "discount": disc,
                        "img_idx": img_idx,
                        "color": color, "delivery": delivery, "category": category,
                        "custom_pil": custom_pil,
                    }

            st.markdown("---")
            if st.button("📊 Generate Excel", type="primary"):
                xl_products = []
                xl_prod_imgs = {}
                xl_dim_imgs  = {}
                for i, prod in enumerate(products_for_xl):
                    p = dict(prod)
                    if not p.get("pdfs"):
                        p["pdfs"] = {"name": pdf_name_for_xl}
                    pp = per_product[i] or {
                        "qty": 1, "discount": 1.0, "img_idx": -1,
                        "color": "如图", "delivery": "现货", "category": "",
                        "custom_pil": None,
                    }
                    p["_qty"]      = pp["qty"]
                    p["_discount"] = pp["discount"]
                    p["_color"]    = pp["color"]
                    p["_delivery"] = pp["delivery"]
                    p["_category"] = pp["category"]
                    xl_products.append(p)

                    # 图片 image: manual upload only
                    if pp["custom_pil"] is not None:
                        xl_prod_imgs[i] = pp["custom_pil"]

                    # 尺寸 dim image: match by product_index (not list position).
                    # Accessories never get a dimension drawing.
                    if dim_images_for_xl and not prod.get("is_accessory"):
                        pidx = prod.get("product_index", 0)
                        if pidx < len(dim_images_for_xl):
                            xl_dim_imgs[i] = dim_images_for_xl[pidx]
                        else:
                            xl_dim_imgs[i] = dim_images_for_xl[-1]

                xl_bytes = xl.build_excel_from_template(
                    xl_products,
                    order_info={
                        "order_number":   xi_order_num,
                        "customer_name":  xi_customer,
                        "contact_person": xi_contact,
                        "phone":          xi_phone,
                    },
                    product_images=xl_prod_imgs,
                    dim_images=xl_dim_imgs,
                )
                st.success(f"✅ Excel generated with {len(xl_products)} products!")
                st.download_button(
                    label="💾 Download Excel",
                    data=xl_bytes,
                    file_name="order_test.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    else:
        st.info("⬆️ Upload a PDF above to unlock sections 3, 4 and 5.")
