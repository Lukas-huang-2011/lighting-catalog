"""All Supabase database and storage operations."""

import io
import uuid
import streamlit as st
from supabase import create_client, Client
from PIL import Image


@st.cache_resource
def get_client() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except KeyError as e:
        st.error(f"Missing Streamlit secret: {e}. Go to app Settings → Secrets and add it.")
        st.stop()
    except Exception as e:
        st.error(f"Could not connect to Supabase: {e}\n\nMake sure SUPABASE_KEY is the anon/public key starting with eyJ...")
        st.stop()


def _upload_bytes(client: Client, path: str, data: bytes, mime: str) -> str:
    client.storage.from_("catalog-files").upload(
        path, data, {"content-type": mime, "upsert": "true"}
    )
    return client.storage.from_("catalog-files").get_public_url(path)


def upload_pdf(client: Client, pdf_bytes: bytes, filename: str) -> str:
    path = f"pdfs/{uuid.uuid4()}_{filename}"
    return _upload_bytes(client, path, pdf_bytes, "application/pdf")


def upload_image(client: Client, image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    path = f"images/{uuid.uuid4()}.png"
    return _upload_bytes(client, path, buf.getvalue(), "image/png")


def create_pdf_record(client: Client, name: str, file_url: str, page_count: int) -> str:
    res = client.table("pdfs").insert({
        "name": name, "file_url": file_url, "page_count": page_count
    }).execute()
    return res.data[0]["id"]


def list_pdfs(client: Client) -> list:
    try:
        return client.table("pdfs").select("*").order("uploaded_at", desc=True).execute().data
    except Exception:
        return []


def delete_pdf(client: Client, pdf_id: str):
    client.table("pdfs").delete().eq("id", pdf_id).execute()


def save_product(client: Client, pdf_id: str, data: dict, page_num: int) -> str:
    res = client.table("products").insert({
        "pdf_id": pdf_id,
        "codes": data.get("codes") or [],
        "name": data.get("name"),
        "description": data.get("description"),
        "color": data.get("color"),
        "light_source": data.get("light_source"),
        "dimensions": data.get("dimensions"),
        "wattage": data.get("wattage"),
        "price": data.get("price"),
        "currency": data.get("currency"),
        "page_number": page_num,
        "raw_text": str(data),
        "extra_fields": data.get("extra_fields") or {}
    }).execute()
    return res.data[0]["id"]


def save_product_image(client: Client, product_id: str, image_url: str,
                       image_hash: str, description: str):
    client.table("product_images").insert({
        "product_id": product_id,
        "image_url": image_url,
        "image_hash": image_hash,
        "image_description": description
    }).execute()


def search_by_code(client: Client, query: str) -> list:
    q = query.strip()
    if not q:
        return []

    seen_ids = set()
    results = []

    # 1. Exact code match (e.g. user typed full code "21019/DIM/AR")
    exact = client.table("products") \
        .select("*, pdfs(name), product_images(image_url, image_hash)") \
        .contains("codes", [q.upper()]) \
        .limit(50).execute()
    for row in (exact.data or []):
        if row["id"] not in seen_ids:
            seen_ids.add(row["id"])
            results.append(row)

    # 2. Partial match — searches codes, name, color, description, anything in raw_text
    partial = client.table("products") \
        .select("*, pdfs(name), product_images(image_url, image_hash)") \
        .ilike("raw_text", f"%{q}%") \
        .limit(100).execute()
    for row in (partial.data or []):
        if row["id"] not in seen_ids:
            seen_ids.add(row["id"])
            results.append(row)

    return results[:100]


def get_all_image_hashes(client: Client) -> list:
    res = client.table("product_images") \
        .select("id, image_hash, image_url, product_id, products(codes, name, description, color, light_source, price, currency, wattage, dimensions, extra_fields)") \
        .not_.is_("image_hash", "null").execute()
    return res.data


def get_products_by_codes(client: Client, codes: list) -> list:
    results = []
    seen_ids = set()
    for code in codes:
        rows = search_by_code(client, code.strip())
        for row in rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                results.append(row)
    return results
