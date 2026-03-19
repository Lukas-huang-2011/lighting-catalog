-- ============================================================
-- Run this in Supabase → SQL Editor (one-time setup)
-- ============================================================

-- 1. PDFs table
CREATE TABLE IF NOT EXISTS pdfs (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name        TEXT NOT NULL,
    file_url    TEXT,
    page_count  INTEGER DEFAULT 0,
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Products table
CREATE TABLE IF NOT EXISTS products (
    id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    pdf_id       UUID REFERENCES pdfs(id) ON DELETE CASCADE,
    codes        TEXT[] DEFAULT '{}',
    name         TEXT,
    description  TEXT,
    color        TEXT,
    light_source TEXT,
    dimensions   TEXT,
    wattage      TEXT,
    price        NUMERIC,
    currency     TEXT,
    page_number  INTEGER DEFAULT 0,
    raw_text     TEXT,
    extra_fields JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Product images table
CREATE TABLE IF NOT EXISTS product_images (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_id        UUID REFERENCES products(id) ON DELETE CASCADE,
    image_url         TEXT NOT NULL,
    image_hash        TEXT,
    image_description TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Indexes for fast search
CREATE INDEX IF NOT EXISTS idx_products_codes
    ON products USING GIN (codes);

CREATE INDEX IF NOT EXISTS idx_product_images_hash
    ON product_images (image_hash);

CREATE INDEX IF NOT EXISTS idx_products_pdf
    ON products (pdf_id);

-- 5. Disable Row Level Security (internal tool — simplest setup)
ALTER TABLE pdfs           DISABLE ROW LEVEL SECURITY;
ALTER TABLE products       DISABLE ROW LEVEL SECURITY;
ALTER TABLE product_images DISABLE ROW LEVEL SECURITY;

-- ============================================================
-- Also do this in Supabase → Storage:
--   Create a bucket named:  catalog-files
--   Set it to PUBLIC
-- ============================================================
