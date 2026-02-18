"""
Lighting Catalog App
────────────────────
• Upload PDFs and convert prices
• Extract product data with AI (Qwen via OpenRouter)
• Search by product code or image
• Generate customer quotes as Excel files
"""

import io
import streamlit as st
from PIL import Image

import database as db
import pdf_processor as pdf
import ai_extractor as ai
import image_search as imgs
import excel_export as xl


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: render product cards  (defined first so all pages can use it)
# ═══════════════════════════════════════════════════════════════════════════════
def _render_product_cards(products: list, show_similarity: bool = False):
    for p in products:
        with st.container():
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            col_img, col_info = st.columns([1, 3])

            with col_img:
                images = p.get("product_images") or []
                if images and images[0].get("image_url"):
                    try:
                        st.image(images[0]["image_url"], use_container_width=True)
                    except Exception:
                        st.caption("Image unavailable")
                else:
                    st.caption("No image")

            with col_info:
                codes = p.get("codes") or []
                badges = " ".join(f'<span class="badge">{c}</span>' for c in codes)
                st.markdown(badges, unsafe_allow_html=True)

                if show_similarity and p.get("similarity"):
                    st.caption(f"Similarity: {p['similarity']}%")

                if p.get("name"):
                    st.subheader(p["name"])

                info_cols = st.columns(3)
                fields = [
                    ("Color", p.get("color")),
                    ("Light Source", p.get("light_source")),
                    ("Dimensions", p.get("dimensions")),
                    ("Wattage", p.get("wattage")),
                    ("Price", f"{p.get('currency','')}{p.get('price')}" if p.get("price") else None),
                    ("Catalog", (p.get("pdfs") or {}).get("name")),
                ]
                shown = [(k, v) for k, v in fields if v]
                for i, (k, v) in enumerate(shown):
                    with info_cols[i % 3]:
                        st.metric(k, v)

                if p.get("description"):
                    with st.expander("Description"):
                        st.write(p["description"])

                ef = p.get("extra_fields") or {}
                if ef:
                    with st.expander("More specifications"):
                        for k, v in ef.items():
                            st.write(f"**{k.title()}:** {v}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.divider()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lighting Catalog",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stButton>button { border-radius: 8px; }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: #fafafa;
    }
    .badge {
        display: inline-block;
        background: #1F3864;
        color: white;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8em;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💡 Lighting Catalog")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📤 Upload & Convert PDFs", "🔍 Search by Code",
         "🖼️ Search by Image", "💰 Pricing & Export", "📚 Manage Catalogs"],
        label_visibility="collapsed"
    )
    st.divider()
    client = db.get_client()
    catalogs = db.list_pdfs(client)
    st.caption(f"**{len(catalogs)}** catalog(s) loaded")
    for c in catalogs[:8]:
        st.caption(f"• {c['name']} ({c.get('page_count', '?')} pages)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload & Convert
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload & Convert PDFs":
    st.header("📤 Upload & Convert PDFs")
    tab1, tab2 = st.tabs(["Upload & Extract Products", "Convert Prices Only"])

    # ── Tab 1: Full upload + AI extraction ───────────────────────────────────
    with tab1:
        st.markdown("Upload a catalog PDF. The AI will read every page and extract all products.")
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"], key="upload_extract")

        if uploaded:
            pdf_bytes = uploaded.read()
            page_count = pdf.get_page_count(pdf_bytes)
            st.info(f"**{uploaded.name}** — {page_count} pages")

            col1, col2 = st.columns(2)
            with col1:
                extract_images_flag = st.checkbox("Extract product images", value=True)
            with col2:
                dpi = st.select_slider("Render quality", [100, 150, 200], value=150)

            if st.button("🚀 Upload & Extract All Products", type="primary"):
                api_key = st.secrets.get("OPENROUTER_API_KEY", "")
                if not api_key:
                    st.error("OpenRouter API key not configured. Add it to Streamlit secrets.")
                    st.stop()

                ai_client = ai.get_client(api_key)

                with st.spinner("Uploading PDF to cloud storage…"):
                    file_url = db.upload_pdf(client, pdf_bytes, uploaded.name)
                    pdf_id = db.create_pdf_record(client, uploaded.name, file_url, page_count)

                progress = st.progress(0, text="Starting…")
                status = st.empty()
                total_products = 0

                for page_num, page_img in enumerate(pdf.render_pages(pdf_bytes, dpi=dpi)):
                    pct = (page_num + 1) / page_count
                    progress.progress(pct, text=f"Processing page {page_num + 1} / {page_count}…")
                    status.caption(f"Reading page {page_num + 1}…")

                    # AI extraction
                    products = ai.extract_products_from_page(ai_client, page_img, page_num)

                    # Extract embedded images for this page
                    page_images = []
                    if extract_images_flag:
                        page_images = pdf.extract_images_from_page(pdf_bytes, page_num)

                    for prod in products:
                        prod_id = db.save_product(client, pdf_id, prod, page_num)
                        total_products += 1

                        # Attach images to this product
                        if page_images:
                            for pil_img in page_images[:3]:  # max 3 images per product
                                img_url = db.upload_image(client, pil_img)
                                img_hash = imgs.compute_hash(pil_img)
                                description = ai.describe_image(ai_client, pil_img)
                                db.save_product_image(client, prod_id, img_url, img_hash, description)

                progress.progress(1.0, text="Done!")
                st.success(f"✅ Extracted **{total_products} products** from {page_count} pages!")
                st.cache_resource.clear()

    # ── Tab 2: Price conversion only ─────────────────────────────────────────
    with tab2:
        st.markdown("Convert prices in a PDF without changing anything else.")
        uploaded_conv = st.file_uploader("Choose a PDF", type=["pdf"], key="upload_convert")

        if uploaded_conv:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                from_sym = st.text_input("From currency symbol", value="€")
            with col2:
                to_sym = st.text_input("To currency symbol", value="$")
            with col3:
                multiplier = st.number_input("Multiplier", min_value=0.01, value=1.08, step=0.01,
                                             help="e.g. 1.08 = multiply by 1.08")
            with col4:
                st.metric("Example", f"{to_sym}{100 * multiplier:.2f}", f"was {from_sym}100.00")

            if st.button("🔄 Convert Prices", type="primary"):
                with st.spinner("Converting prices…"):
                    original_bytes = uploaded_conv.read()
                    converted_bytes = pdf.convert_prices(original_bytes, from_sym, multiplier, to_sym)

                st.success("Done! Download your converted PDF below.")
                st.download_button(
                    "⬇️ Download Converted PDF",
                    data=converted_bytes,
                    file_name=f"converted_{uploaded_conv.name}",
                    mime="application/pdf"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Search by Code
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Search by Code":
    st.header("🔍 Search by Product Code")

    query = st.text_input("Enter a product code", placeholder="e.g. AB1234 or part of a code…")

    if query:
        with st.spinner("Searching…"):
            results = db.search_by_code(client, query)

        if not results:
            st.warning(f"No products found for **{query}**")
        else:
            st.success(f"Found **{len(results)}** product(s)")
            _render_product_cards(results)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Search by Image
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Search by Image":
    st.header("🖼️ Search by Image")
    st.markdown("Upload a photo of a light fitting to find matching products.")

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_img:
        query_img = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(query_img, caption="Your search image", use_container_width=True)
        with col2:
            threshold = st.slider("Matching sensitivity", 5, 40, 20,
                                  help="Higher = finds more results but less precise")
            search_btn = st.button("🔍 Find Matching Products", type="primary")

        if search_btn:
            with st.spinner("Comparing against all catalog images…"):
                all_hashes = db.get_all_image_hashes(client)
                matches = imgs.find_similar(query_img, all_hashes, threshold=threshold)

            if not matches:
                st.warning("No similar products found. Try increasing the sensitivity slider.")

                # Offer AI description search as fallback
                if st.button("🤖 Try AI Description Search"):
                    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
                    if api_key:
                        ai_client = ai.get_client(api_key)
                        with st.spinner("Asking AI to describe your image…"):
                            desc = ai.describe_image(ai_client, query_img)
                        st.info(f"**AI description:** {desc}")
                        st.caption("Use keywords from the description in the Code Search tab.")
            else:
                st.success(f"Found **{len(matches)}** similar product(s)")
                # Build result list from product data inside matches
                prod_results = []
                for m in matches:
                    p = m.get("products")
                    if p:
                        p["product_images"] = [{"image_url": m.get("image_url"), "image_hash": m.get("image_hash")}]
                        p["similarity"] = m.get("similarity_score", 0)
                        prod_results.append(p)
                _render_product_cards(prod_results, show_similarity=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Pricing & Export
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Pricing & Export":
    st.header("💰 Customer Pricing & Excel Export")

    col1, col2 = st.columns([3, 1])
    with col1:
        codes_input = st.text_area(
            "Enter product codes (one per line)",
            height=200,
            placeholder="AB1234\nCD5678\nEF9012"
        )
    with col2:
        discount_input = st.number_input(
            "Discount factor",
            min_value=0.01, max_value=1.0, value=0.7, step=0.01,
            help="0.7 = customer pays 70% (30% off). 1.0 = no discount."
        )
        discount_pct = round((1 - discount_input) * 100, 1)
        st.metric("Customer discount", f"{discount_pct}% off")
        include_imgs = st.checkbox("Include images in Excel", value=True)

    if st.button("📊 Look Up Products & Generate Excel", type="primary"):
        codes = [c.strip() for c in codes_input.splitlines() if c.strip()]
        if not codes:
            st.warning("Please enter at least one product code.")
            st.stop()

        with st.spinner(f"Looking up {len(codes)} code(s)…"):
            products = db.get_products_by_codes(client, codes)

        if not products:
            st.error("None of the entered codes were found in the catalog.")
            st.stop()

        not_found = [c for c in codes if not any(
            c.upper() in [x.upper() for x in (p.get("codes") or [])] for p in products
        )]
        if not_found:
            st.warning(f"Not found: {', '.join(not_found)}")

        st.success(f"Found **{len(products)}** product(s). Preview:")

        # Preview table
        rows = []
        for p in products:
            orig = p.get("price")
            cust = round(orig * discount_input, 2) if orig else None
            rows.append({
                "Code(s)": ", ".join(p.get("codes") or []),
                "Name": p.get("name") or "",
                "Color": p.get("color") or "",
                "Light Source": p.get("light_source") or "",
                "Original Price": f"{p.get('currency','')}{orig}" if orig else "—",
                "Customer Price": f"{p.get('currency','')}{cust}" if cust else "—",
            })
        st.dataframe(rows, use_container_width=True)

        with st.spinner("Building Excel file…"):
            excel_bytes = xl.build_excel(products, discount_input, include_images=include_imgs)

        st.download_button(
            "⬇️ Download Excel Quote",
            data=excel_bytes,
            file_name="customer_quote.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Manage Catalogs
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Manage Catalogs":
    st.header("📚 Manage Catalogs")
    catalogs = db.list_pdfs(client)

    if not catalogs:
        st.info("No catalogs uploaded yet. Go to Upload & Convert PDFs to add one.")
    else:
        for cat in catalogs:
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(f"**{cat['name']}**")
                st.caption(f"{cat.get('page_count','?')} pages · Uploaded {cat['uploaded_at'][:10]}")
            with col2:
                if cat.get("file_url"):
                    st.link_button("View PDF", cat["file_url"])
            with col3:
                if st.button("🗑️ Delete", key=f"del_{cat['id']}"):
                    db.delete_pdf(client, cat["id"])
                    st.rerun()


