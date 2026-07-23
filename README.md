# Document Extract Web

Standalone desktop project (not in `techno-core`). Full stack: React UI + FastAPI backend + vendored sectioning scripts under `vendor/mllm-scripts/`.

Local web app for uploading PDFs or images, running OCR + section layout detection, filtering boilerplate sections, and extracting schema-free label→value pairs via Gemini.

## Pipeline

```
Upload → PDF rasterize (300 DPI) → Google Document AI OCR
      → auto sectioning (horizontal gaps + HV column splits + table-row merge)
      → disclaimer/boilerplate filter → **local small LLM section gate** → per-section Gemini extraction → merged JSON
```

Reuses modules from `vendor/mllm-scripts/` (synced from company `mllm-invoker/scripts` when needed):

- `ocr_word_to_line_boxes` / `ocr_line_to_sections` — layout detection
- `section_preprocess` — filter disclaimers / legal prose
- `app/section_crop` — section crops for Gemini (no gallery deps)

## Prerequisites

- Python 3.11+ (`/opt/homebrew/bin/python3.11` on macOS)
- Node 18+ (for frontend)
- `pdftoppm` (poppler) optional — falls back to PyMuPDF
- Google Cloud credentials for **Document AI** and **Gemini**

### Credentials policy

| Context | Credentials |
|---------|-------------|
| **Doc-extract-web site** (`./run.sh`, local) | Personal `.env.local` in this repo for OCR; optional company Vertex overlay only if you point `MLLM_DIR` at company checkout |
| **Render deploy** (`docparse-api`) | **Personal dashboard env vars only** — never reads company `mllm-invoker/.env.local` or Vertex |
| **User emails** | **SQLite** (stdlib) — no Redshift/AWS DB; optional `DOC_EXTRACT_USERS_DB` path |
| **Everything else** (demos, galleries, ad-hoc scripts) | Company `mllm-invoker/.env.local` only |

Personal creds are never loaded outside the website. Use `scripts/sectioning_demo.py` for ad-hoc sectioning — it always uses company creds.

### Render (production API)

Set these in the **Render dashboard** only — your personal keys, never company service accounts:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Personal Google AI Studio key (required for extraction) |
| `GOOGLE_CLOUD_API_KEY` | Personal Vision API key for OCR |
| `DOC_EXTRACT_CRED_MODE` | `personal_only` (set in `render.yaml`; blocks company Vertex) |

Render auto-sets `RENDER=true`, which triggers personal-only mode. Company keyfiles and `mllm-invoker/.env.local` are **not** copied into the Docker image and are **never** loaded at runtime.

Verify after deploy: `curl https://docparse-api.onrender.com/api/health` → `"cred_mode": "personal_deploy"`, `"gemini_auth": "api_key"`, `"gemini_project": null`

### Personal account (local website)

Create **`.env.local`** in this repo — the website loads this and does **not** load company `mllm-invoker/.env.local` on Render.

```bash
cd ~/Desktop/classification/nikhil_desktop/doc-extract-web
cp .env.local.example .env.local
# edit .env.local — add your personal GCP_PROJECT_ID + GEMINI_API_KEY
./run.sh
```

Verify personal creds: `curl http://127.0.0.1:8000/api/health` → `"credentials": "personal"`

| What | Your personal setup |
|------|---------------------|
| **Google OCR** | Personal GCP project + [Cloud Vision API](https://console.cloud.google.com/apis/library/vision.googleapis.com) + `gcloud auth application-default login` |
| **Extraction** | [Google AI Studio](https://aistudio.google.com/apikey) API key (personal Gmail) |

### Company credentials (demos + local website Gemini overlay)

- **Demos / scripts:** `python scripts/sectioning_demo.py <image>` — always company creds
- **Local website:** if `.env.local` exists, OCR uses personal creds only (Render uses dashboard env vars)

### Environment variables reference

Loads optional company overlay only when `DOC_EXTRACT_GEMINI_*` or paths are configured locally.

| Variable | Purpose |
|----------|---------|
| `GCP_PROJECT_NUMBER` or `GCP_PROJECT_ID` | Document AI + Vertex |
| `DOCUMENT_AI_PROCESSOR_ID` or `PROCESSOR_ID` | OCR processor |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account JSON path |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Gemini API (or use Vertex via project creds) |
| `DOC_EXTRACT_GEMINI_MODEL` | Default `gemini-3.1-flash-lite` |
| `DOC_EXTRACT_SECTION_GATE` | `1` = classify each section before full extract; `0` = off |
| `DOC_EXTRACT_GATE_PROVIDER` | `local` (default): Ollama/vLLM; `gemini` for cloud gate only |
| `DOC_EXTRACT_GATE_BASE_URL` | OpenAI-compatible URL (default `http://127.0.0.1:11434/v1`) |
| `DOC_EXTRACT_GATE_MODEL` | Tag on the server, e.g. `qwen2.5:3b-instruct`, `mistral:7b-instruct` |
| `DOC_EXTRACT_GATE_USE_IMAGE` | `1`/`0` force vision; if unset, **auto-on** for model names containing `vl` (e.g. `qwen2.5vl:3b`) |
| `DOC_EXTRACT_GATE_JSON_MODE` | `1` = request JSON object from server (default) |
| `DOC_EXTRACT_USERS_DB` | Optional path to SQLite file (default: `../data/users.sqlite` under job root) |
| `DOC_EXTRACT_DATA_DIR` | Directory for default `users.sqlite` |
| `EXTRACT_PROVIDER` | `gemini` (default); `qwen` reserved for future |

### Sign-in, credits, and usage (SQLite, free)

- **Email/password:** `POST /api/auth/register` and `POST /api/auth/login` (bcrypt hashes in `password_hash`).
- **Google:** `POST /api/auth/google` (optional; links to the same row by email).
- **Credits:** new accounts get **2 credits** (`DOC_EXTRACT_INITIAL_CREDITS`). Each upload job costs **2 credits** (`DOC_EXTRACT_CREDITS_PER_DOC`). Insufficient balance returns **402** with a clear message.
- **Ledger:** `credit_ledger` stores signup bonuses and per-`job_id` charges (`delta`, `balance_after`, `reason`).

```bash
sqlite3 /path/to/users.sqlite "SELECT email, credits FROM docparse_users;"
sqlite3 /path/to/users.sqlite "SELECT * FROM credit_ledger ORDER BY id DESC LIMIT 10;"
```

`/api/health` includes `"users_db": { "backend": "sqlite", "initial_credits": 2, "credits_per_document": 2, ... }`.

On Render’s free tier the filesystem is ephemeral unless you set `DOC_EXTRACT_USERS_DB` to a persistent disk path. Later you can point the same app at a **free** hosted Postgres (Neon/Supabase) if you outgrow SQLite — still no AWS Redshift bill.

## Setup

### Backend

```bash
cd doc-extract-web/backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Frontend

```bash
cd doc-extract-web/frontend
npm install
```

## Run

**One command (recommended):**

```bash
cd doc-extract-web
./run.sh
```

Open http://127.0.0.1:5173

Or two terminals:

```bash
# Terminal 1 — API
cd doc-extract-web
./run-backend.sh

# Terminal 2 — UI
cd doc-extract-web/frontend
npm run dev
```

### OCR backend

- If `DOCUMENT_AI_PROCESSOR_ID` is set → uses **Google Document AI**
- Otherwise → falls back to **Google Cloud Vision API** using `GOOGLE_CLOUD_KEYFILE_JSON` (already in mllm-invoker `.env.local`)

Check config: `curl http://127.0.0.1:8000/api/health`

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/jobs` | Upload file (multipart `file`) |
| `GET` | `/api/jobs/{id}` | Job status + result JSON |
| `GET` | `/api/jobs/{id}/pages/{n}/overlay.png` | Section overlay (green=kept, red=filtered) |
| `GET` | `/api/jobs/{id}/pages/{n}/sections/{s}/crop.png` | Section crop |

Jobs are stored under `/tmp/doc-extract-web/{job_id}/`.

## Skip LLM (pipeline test)

```bash
curl -X POST "http://127.0.0.1:8000/api/jobs?skip_llm=true" -F "file=@page.png"
```

## Self-hosted section gate (no Gemini API for classify)

The gate decides **extractable vs disclaimer** per section. Default is a **local** OpenAI-compatible server (Ollama is the usual setup) — **no training**, pick any small instruct model (Qwen, Mistral, Phi, etc.).

```bash
# Text-only gate (fast, smaller):
ollama pull qwen2.5:3b
export DOC_EXTRACT_GATE_MODEL=qwen2.5:3b
export DOC_EXTRACT_GATE_USE_IMAGE=0

# Vision gate (section crop + OCR) — Qwen2.5-VL:
ollama pull qwen2.5vl:3b
export DOC_EXTRACT_GATE_MODEL=qwen2.5vl:3b
# USE_IMAGE defaults on when the model name contains "vl"
```

Works with **vLLM**, **llama.cpp server**, or **Mistral/LM Studio** — any `/v1/chat/completions` endpoint. Set `DOC_EXTRACT_GATE_BASE_URL` and `DOC_EXTRACT_GATE_MODEL` to match.

If Ollama is down, the gate **defaults to extractable** (full Gemini extract still runs) so jobs do not silently drop data.

Full field extraction still uses Gemini today (`DOC_EXTRACT_GEMINI_MODEL` / company creds). Only the gate is local unless you later wire `EXTRACT_PROVIDER=qwen`.

## Future: Qwen 2.5 VL

Set `EXTRACT_PROVIDER=qwen` once `app/extract/qwen.py` is implemented (Ollama/vLLM endpoint). v1 uses Gemini only.
