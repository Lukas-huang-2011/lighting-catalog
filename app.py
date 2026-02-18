"""
Lighting Catalog App
• Upload PDFs and extract products with AI
• Convert prices between currencies
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

st.set_page_config(page_title="Lighting Catalog", page_icon="💡", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stButton>button { border-radius: 8px; }
  .product-card { border:1px solid #e0e0e0; border-radius:12px; padding:16px; margin-bottom:12px; background:#fafafa; }
  .badge { display:inline-block; background:#1F3864; color:white; border-radius:6px; padding:2px 8px; font-size:0.8em; margin:2px; }
</style>
""", unsafe_allow_html=True)


# ── Helper: product cards ─────────────────────────────────────────────────────
def _render_cards(products: list, show_similarity: bool = False):
    for p in products:
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
            st.markdown(" ".join(f'<span class="badge">{c}</span>' for c in codes),
                        unsafe_allow_html=True)
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
                ("Catalog", (p.get("pdfs") or {}).get("name")),
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💡 Lighting Catalog")
    st.divider()
    page = st.radio("Navigation", [
        "📤 Upload & Extract",
        "🔄 Convert Prices",
        "🔍 Search by Code",
        "🖼️ Search by Image",
        "💰 Pricing & Export",
        "📚 Manage Catalogs"
    ], label_visibility="collapsed")
    st.divider()
    client = db.get_client()
    catalogs = db.list_pdfs(client)
    st.caption(f"**{len(catalogs)}** catalog(s) loaded")
    for c in catalogs[:8]:
        st.caption(f"• {c['name']} ({c.get('page_count','?')} pages)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload & Extract
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload & Extract":
    st.header("📤 Upload Catalog & Extract Products")
    st.markdown("Upload a PDF catalog. The AI reads every page and saves all products to the database.")

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        pdf_bytes = uploaded.read()
        page_count = pdf.get_page_count(pdf_bytes)
        st.info(f"**{uploaded.name}** — {page_count} pages")

        col1, col2, col3 = st.columns(3)
        with col1:
            extract_images_flag = st.checkbox("Extract product images", value=True)
        with col2:
            dpi = st.select_slider("Render quality", [100, 150, 200], value=150,
                                   help="Higher = better AI reading but slower")
        with col3:
            st.caption("Currency is detected automatically from the PDF column headers.")

        if st.button("🚀 Upload & Extract All Products", type="primary"):
            api_key = st.secrets.get("OPENROUTER_API_KEY", "")
            if not api_key:
                st.error("OpenRouter API key not configured in Streamlit secrets.")
                st.stop()

            ai_client = ai.get_client(api_key)

            with st.spinner("Uploading PDF…"):
                file_url = db.upload_pdf(client, pdf_bytes, uploaded.name)
                pdf_id = db.create_pdf_record(client, uploaded.name, file_url, page_count)

            progress = st.progress(0, text="Starting…")
            log = st.empty()
            total_products = 0
            errors = []

            for page_num, page_img in enumerate(pdf.render_pages(pdf_bytes, dpi=dpi)):
                pct = (page_num + 1) / page_count
                progress.progress(pct, text=f"Page {page_num + 1} / {page_count}…")
                log.caption(f"Reading page {page_num + 1}…")

                products = ai.extract_products_from_page(ai_client, page_img, page_num)

                page_images = []
                if extract_images_flag:
                    try:
                        page_images = pdf.extract_images_from_page(pdf_bytes, page_num)
                    except Exception as e:
                        errors.append(f"Page {page_num+1} image extract: {e}")

                for prod in products:
                    try:
                        prod_id = db.save_product(client, pdf_id, prod, page_num)
                        total_products += 1

                        for pil_img in page_images[:3]:
                            try:
                                img_url = db.upload_image(client, pil_img)
                                img_hash = imgs.compute_hash(pil_img)
                                description = ai.describe_image(ai_client, pil_img)
                                db.save_product_image(client, prod_id, img_url, img_hash, description)
                            except Exception as e:
                                errors.append(f"Image upload: {e}")
                    except Exception as e:
                        errors.append(f"Save product: {e}")

            progress.progress(1.0, text="Done!")
            st.success(f"✅ Extracted **{total_products} products** from {page_count} pages!")
            if errors:
                with st.expander(f"⚠️ {len(errors)} minor warnings"):
                    for e in errors[:20]:
                        st.caption(e)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Convert Prices
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Convert Prices":
    st.header("🔄 Convert Prices in a PDF")
    st.markdown("Converts all prices in the PDF. Images, layout, and everything else stays exactly the same.")

    uploaded_conv = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_conv:
        st.subheader("Conversion settings")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original currency in the PDF**")
            from_type = st.radio("How are prices marked?", [
                "Currency symbol before price (e.g. € 149,00)",
                "No symbol — currency is in column header (e.g. RMB, EUR)"
            ], label_visibility="collapsed")

            if "symbol" in from_type:
                from_currency = st.text_input("Symbol", value="€",
                                              help="The symbol that appears before each price")
            else:
                from_currency = st.text_input("Currency label in column header", value="RMB",
                                              help="The text in the column header, e.g. RMB, EUR, USD")

        with col2:
            st.markdown("**Convert to**")
            to_currency = st.text_input("New currency label/symbol", value="€",
                                        help="This will be written before each converted price")
            multiplier = st.number_input("Multiplier", min_value=0.0001, value=0.13, step=0.01,
                                         help="New price = original × multiplier. E.g. RMB→EUR ≈ 0.13")

        # Live example
        example_price = 14469.00
        st.info(f"**Example:** {from_currency} {example_price:,.2f} → {to_currency} {example_price * multiplier:,.2f}")

        if st.button("🔄 Convert & Download", type="primary"):
            pdf_bytes = uploaded_conv.read()
            with st.spinner("Converting prices…"):
                converted = pdf.convert_prices(pdf_bytes, from_currency, multiplier, to_currency)
            st.success("Done!")
            st.download_button(
                "⬇️ Download Converted PDF",
                data=converted,
                file_name=f"converted_{uploaded_conv.name}",
                mime="application/pdf"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Search by Code
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Search by Code":
    st.header("🔍 Search by Product Code")

    query = st.text_input("Enter a product code", placeholder="e.g. 21019/DIM/AR or part of a code…")

    if query:
        with st.spinner("Searching…"):
            results = db.search_by_code(client, query)
        if not results:
            st.warning(f"No products found for **{query}**")
        else:
            st.success(f"Found **{len(results)}** result(s)")
            _render_cards(results)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Search by Image
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Search by Image":
    st.header("🖼️ Search by Image")
    st.markdown("Upload a photo of a light fitting to find matching products.")

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_img:
        query_img = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(query_img, caption="Your image", use_container_width=True)
        with col2:
            threshold = st.slider("Sensitivity", 5, 40, 20,
                                  help="Higher = more results but less precise")
            if st.button("🔍 Find Matches", type="primary"):
                with st.spinner("Comparing against all catalog images…"):
                    all_hashes = db.get_all_image_hashes(client)
                    matches = imgs.find_similar(query_img, all_hashes, threshold=threshold)

                if not matches:
                    st.warning("No similar images found. Try increasing the sensitivity.")
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

    col1, col2 = st.columns([3, 1])
    with col1:
        codes_input = st.text_area("Product codes (one per line)", height=200,
                                   placeholder="21019/DIM/AR\n21019/DIM/AZ\n12345/BI")
    with col2:
        discount = st.number_input("Discount factor", min_value=0.01, max_value=1.0,
                                   value=0.7, step=0.01,
                                   help="0.7 = customer pays 70% (30% off)")
        st.metric("Discount", f"{round((1-discount)*100,1)}% off")
        include_imgs = st.checkbox("Include images in Excel", value=True)

    if st.button("📊 Generate Excel Quote", type="primary"):
        codes = [c.strip() for c in codes_input.splitlines() if c.strip()]
        if not codes:
            st.warning("Please enter at least one product code.")
            st.stop()

        with st.spinner(f"Looking up {len(codes)} code(s)…"):
            products = db.get_products_by_codes(client, codes)

        if not products:
            st.error("None of the entered codes were found.")
            st.stop()

        not_found = [c for c in codes if not any(
            c.upper() in [x.upper() for x in (p.get("codes") or [])] for p in products
        )]
        if not_found:
            st.warning(f"Not found: {', '.join(not_found)}")

        st.success(f"Found **{len(products)}** product(s). Preview:")

        rows = []
        for p in products:
            orig = p.get("price")
            cust = round(orig * discount, 2) if orig else None
            rows.append({
                "Code(s)": ", ".join(p.get("codes") or []),
                "Name": p.get("name") or "",
                "Color": p.get("color") or "",
                "Original Price": f"{p.get('currency','')} {orig}" if orig else "—",
                "Customer Price": f"{p.get('currency','')} {cust}" if cust else "—",
            })
        st.dataframe(rows, use_container_width=True)

        with st.spinner("Building Excel…"):
            excel_bytes = xl.build_excel(products, discount, include_images=include_imgs)

        st.download_button(
            "⬇️ Download Excel Quote", data=excel_bytes,
            file_name="customer_quote.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
                st.write(f"**{cat['name']}**")
                st.caption(f"{cat.get('page_count','?')} pages · {cat['uploaded_at'][:10]}")
            with col2:
                if cat.get("file_url"):
                    st.link_button("View PDF", cat["file_url"])
            with col3:
                if st.button("🗑️ Delete", key=f"del_{cat['id']}"):
                    db.delete_pdf(client, cat["id"])
                    st.rerun()
