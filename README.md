# DocExtract API

Backend for [extracteverythiing.vercel.app](https://extracteverythiing.vercel.app) — document OCR, section layout, hybrid table detection, and Gemini extraction.

## Deploy (Render)

- **Service:** `docparse-api` on Render
- **Runtime:** Docker (`Dockerfile` at repo root)
- **Required env vars (Render dashboard):**
  - `GEMINI_API_KEY` — personal Google AI Studio key
  - `GOOGLE_CLOUD_API_KEY` — personal Vision API key for OCR
  - `DOC_EXTRACT_CRED_MODE=personal_only` (set in Dockerfile + render.yaml)
  - `CORS_ORIGINS` — Vercel frontend URLs

Health check: `GET /api/health` → `"cred_mode": "personal_deploy"`, `"gemini_auth": "api_key"`

## Local dev

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
export GEMINI_API_KEY=...
export GOOGLE_CLOUD_API_KEY=...
export DOC_EXTRACT_CRED_MODE=personal_only
uvicorn app.main:app --reload --port 8000
```
