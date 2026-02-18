"""
All Supabase database and storage operations.
"""

import io
import uuid
import streamlit as st
from supabase import create_client, Client
from PIL import Image


@st.cache_resource
def get_client() -> Client:
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


# ── Storage ──────────────────────────────────────────────────────────────────

def _upload_bytes(client: Client, path: str, data: bytes, mime: str) -> str:
    """Upload bytes to the catalog-files bucket, return public URL."""
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


# ── PDF records ───────────────────────────────────────────────────────────────

def create_pdf_record(client: Client, name: str, file_url: str, page_count: int) -> str:
    res = client.table("pdfs").insert({
        "name": name,
        "file_url": file_url,
        "page_count": page_count
    }).execute()
    return res.data[0]["id"]


def list_pdfs(client: Client) -> list:
    return client.table("pdfs").select("*").order("uploaded_at", desc=True).execute().data


def delete_pdf(client: Client, pdf_id: str):
    client.table("pdfs").delete().eq("id", pdf_id).execute()


# ── Products ──────────────────────────────────────────────────────────────────

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


# ── Search ────────────────────────────────────────────────────────────────────

def search_by_code(client: Client, query: str) -> list:
    """Search products by code (exact array match or raw_text contains)."""
    query = query.strip().upper()

    # Try exact match in codes array
    res = client.table("products") \
        .select("*, pdfs(name), product_images(image_url, image_hash, image_description)") \
        .contains("codes", [query]) \
        .execute()

    if res.data:
        return res.data

    # Fallback: partial match anywhere in raw_text
    res = client.table("products") \
        .select("*, pdfs(name), product_images(image_url, image_hash, image_description)") \
        .ilike("raw_text", f"%{query}%") \
        .execute()

    return res.data


def get_all_image_hashes(client: Client) -> list:
    """Fetch all image hashes + linked product info for similarity search."""
    res = client.table("product_images") \
        .select("id, image_hash, image_url, product_id, products(codes, name, description, color, light_source, price, currency, wattage, dimensions, extra_fields)") \
        .not_.is_("image_hash", "null") \
        .execute()
    return res.data


def get_products_by_codes(client: Client, codes: list) -> list:
    """Fetch product details for a list of codes (for pricing page)."""
    results = []
    seen_ids = set()
    for code in codes:
        rows = search_by_code(client, code.strip())
        for row in rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                results.append(row)
    return results
