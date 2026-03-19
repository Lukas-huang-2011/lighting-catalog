# Lighting Catalog App — Setup Instructions

## What you need (all free)
- GitHub account
- Supabase account (you have this)
- OpenRouter account (you have this)
- Streamlit Cloud account (free at share.streamlit.io)

---

## Step 1 — Set up Supabase (5 minutes)

1. Go to **supabase.com** and log in
2. Open your project
3. Click **SQL Editor** in the left sidebar
4. Copy the entire contents of `supabase_setup.sql` and paste it in
5. Click **Run** — this creates all the database tables
6. Now click **Storage** in the left sidebar
7. Click **New bucket**
8. Name it exactly: `catalog-files`
9. Turn on **Public bucket** → click Save

---

## Step 2 — Create a GitHub repository (3 minutes)

1. Go to **github.com** and log in
2. Click the **+** button (top right) → **New repository**
3. Name it: `lighting-catalog`
4. Keep it **Private**
5. Click **Create repository**
6. Click **uploading an existing file**
7. Drag and drop ALL the files from this folder into GitHub
   - app.py, database.py, pdf_processor.py, ai_extractor.py
   - image_search.py, excel_export.py, requirements.txt
   - The `.streamlit` folder with secrets.toml inside
8. Click **Commit changes**

---

## Step 3 — Deploy on Streamlit Cloud (5 minutes)

1. Go to **share.streamlit.io** and sign in with your GitHub account
2. Click **New app**
3. Select your `lighting-catalog` repository
4. Main file path: `app.py`
5. Click **Advanced settings**
6. In the **Secrets** box, paste this (with your real values):

```toml
SUPABASE_URL = "https://atuepfyupezwuhywwbpx.supabase.co"
SUPABASE_KEY = "sb_publishable_ysJ1Gu_9hZu1l58Y2drUfg_3YLMNpdA"
OPENROUTER_API_KEY = "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"
```

7. Click **Deploy** — it will take ~2 minutes to start

Once deployed, you get a URL like `https://your-app.streamlit.app` — share this with your team!

---

## How to use the app

### Upload a catalog
- Go to **Upload & Convert PDFs** → first tab
- Upload a PDF and click **Upload & Extract All Products**
- The AI reads every page and saves all products to the database
- First upload may take a few minutes depending on PDF size

### Convert prices
- Go to **Upload & Convert PDFs** → second tab
- Upload a PDF, set the currency symbols and multiplier
- Download the converted PDF

### Search by code
- Go to **Search by Code**
- Type any product code (or part of one)

### Search by image
- Go to **Search by Image**
- Upload a photo of a light fitting
- Adjust sensitivity if needed

### Create a customer quote
- Go to **Pricing & Export**
- Paste product codes (one per line)
- Set the discount factor (0.7 = 30% off)
- Download the Excel file

---

## Costs
- Streamlit Cloud: **free**
- Supabase: **free** (up to 500MB storage, 50,000 rows)
- OpenRouter/Qwen: roughly **€0.002 per PDF page** for extraction
  - 100-page catalog ≈ €0.20
  - After extraction, all searches and exports are **free**
