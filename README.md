# Potutrely Python Service — Deploy Guide

## Stack
- FastAPI + MediaPipe + OpenCV
- Gambar disimpan ke **Cloudinary** (bukan disk lokal)
- Deploy ke **Render.com** (free tier)

---

## 1. Setup Cloudinary (5 menit)

1. Daftar di https://cloudinary.com (free, no CC)
2. Buka **Dashboard** → catat:
   - `Cloud Name`
   - `API Key`
   - `API Secret`
3. Tidak perlu setting tambahan apa-apa

---

## 2. Push ke GitHub

```bash
# Pastikan .env tidak ikut ke-commit
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/USERNAME/potutrely-python.git
git push -u origin main
```

> ⚠️ Pastikan `.env` ada di `.gitignore` sebelum push!

---

## 3. Deploy ke Render.com

1. Daftar di https://render.com (free, no CC)
2. Klik **New → Web Service**
3. Connect repo GitHub `potutrely-python`
4. Settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Scroll ke **Environment Variables**, tambahkan:

| Key | Value |
|-----|-------|
| `CLOUDINARY_CLOUD_NAME` | dari Cloudinary dashboard |
| `CLOUDINARY_API_KEY` | dari Cloudinary dashboard |
| `CLOUDINARY_API_SECRET` | dari Cloudinary dashboard |
| `ALLOWED_ORIGINS` | `https://your-laravel.koyeb.app,https://your-fe.vercel.app` |

6. Klik **Create Web Service**
7. Tunggu build ~5-10 menit (download MediaPipe + model)
8. URL service kamu: `https://potutrely-python.onrender.com`

---

## 4. Test Endpoint

```bash
curl -X POST https://potutrely-python.onrender.com/analyze-posture \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://res.cloudinary.com/CLOUD/image/upload/sample.jpg",
    "view": "FRONT"
  }'
```

Response:
```json
{
  "score": 80.0,
  "category": "FAIR",
  "metrics": { ... },
  "summary": "...",
  "overlay_image_url": "https://res.cloudinary.com/...",
  "crop_images": [...]
}
```

---

## 5. Update URL di Laravel

Di `.env` Laravel:
```
PYTHON_SERVICE_URL=https://potutrely-python.onrender.com
```

Di code Laravel yang manggil Python:
```php
$response = Http::post(env('PYTHON_SERVICE_URL') . '/analyze-posture', [
    'image_url' => $imageUrl,  // URL Cloudinary dari upload FE
    'view'      => $request->view,
]);
```

---

## ⚠️ Catatan Free Tier Render

- Service **sleep setelah 1 jam idle** — cold start ~30 detik
- Gambar **tidak hilang** karena tersimpan di Cloudinary
- Kalau mau hindari sleep: set ping ke `/health` tiap 50 menit
  - Pakai https://cron-job.org (free) untuk ping otomatis

### Setup Ping Anti-Sleep (Opsional tapi recommended)

1. Daftar https://cron-job.org (free)
2. New Cronjob:
   - URL: `https://potutrely-python.onrender.com/health`
   - Schedule: setiap 50 menit
3. Done — service tidak akan sleep selama ada traffic

---

## Struktur File

```
potutrely-python/
├── main.py            ← FastAPI app utama
├── requirements.txt   ← dependencies
├── render.yaml        ← config auto-deploy Render
├── .env.example       ← template env vars
├── .env               ← JANGAN di-commit!
└── .gitignore
```
