import os
import uuid
import urllib.request
from io import BytesIO
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import requests
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import cloudinary
import cloudinary.uploader

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY    = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
ALLOWED_ORIGINS       = os.getenv("ALLOWED_ORIGINS", "*").split(",")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True,
)

# ─── MODEL DOWNLOAD ───────────────────────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[MODEL] Downloading pose_landmarker_full.task ...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[MODEL] Download complete.", flush=True)

ensure_model()

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Potutrely Posture AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_url: str
    view: Optional[str] = "FRONT"

class AnalyzeResponse(BaseModel):
    score: float
    category: str
    metrics: dict
    summary: str
    overlay_image_url: Optional[str] = None
    crop_images: Optional[List[Dict]] = None

# ─── POSE LANDMARKER ──────────────────────────────────────────────────────────

def build_pose_landmarker():
    BaseOptions          = mp.tasks.BaseOptions
    PoseLandmarker       = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode    = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
    )
    return PoseLandmarker.create_from_options(options)

pose_landmarker = build_pose_landmarker()

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def download_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    file_bytes = np.asarray(bytearray(BytesIO(resp.content).read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal decode gambar.")
    return img

def upload_to_cloudinary(image_np: np.ndarray, folder: str = "potutrely") -> str:
    _, buffer = cv2.imencode(".png", image_np, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    result = cloudinary.uploader.upload(
        buffer.tobytes(),
        folder=folder,
        resource_type="image",
        public_id=uuid.uuid4().hex,
    )
    return result["secure_url"]

def extract_landmarks(image: np.ndarray):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result    = pose_landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]

# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_front_back_metrics(landmarks):
    ls   = landmarks[11]
    rs   = landmarks[12]
    lh   = landmarks[23]
    rh   = landmarks[24]
    nose = landmarks[0]

    shoulder_tilt  = abs(ls.y - rs.y) * 100.0
    hip_tilt       = abs(lh.y - rh.y) * 100.0
    shoulder_ctr_x = (ls.x + rs.x) / 2.0
    shoulder_w     = abs(ls.x - rs.x) + 1e-6
    forward_head   = abs(nose.x - shoulder_ctr_x) / shoulder_w

    return {
        "shoulder_tilt_index": float(shoulder_tilt),
        "hip_tilt_index":      float(hip_tilt),
        "forward_head_index":  float(forward_head),
    }

def compute_side_view_metrics(landmarks):
    ear_l = landmarks[7]
    ear_r = landmarks[8]
    ear   = ear_l if ear_l.visibility >= ear_r.visibility else ear_r
    shoulder = landmarks[11]
    hip      = landmarks[23]

    def angle_with_vertical(p1, p2):
        dx, dy = p2.x - p1.x, p2.y - p1.y
        if dy == 0:
            return 90.0
        return float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    return {
        "neck_inclination_deg":  angle_with_vertical(shoulder, ear),
        "torso_inclination_deg": angle_with_vertical(hip, shoulder),
    }

# ─── SCORING ──────────────────────────────────────────────────────────────────

def _finding(area, severity, detail, value):
    return {"area": area, "severity": severity, "detail": detail, "metric_value": value}

def rule_based_scoring_front(m):
    sh, hi, hd = m["shoulder_tilt_index"], m["hip_tilt_index"], m["forward_head_index"]
    findings   = []

    for val, mild, mod, label, unit in [
        (sh, 2.0, 5.0, "Bahu",    "%"),
        (hi, 2.0, 5.0, "Panggul", "%"),
    ]:
        if val >= mod:
            findings.append(_finding(label, "Sedang", f"Kemiringan {label.lower()} {val:.1f}{unit}. Asimetri cukup jelas.", val))
        elif val >= mild:
            findings.append(_finding(label, "Ringan", f"Kemiringan {label.lower()} {val:.1f}{unit}. Sedikit berbeda.", val))

    if hd >= 0.25:
        findings.append(_finding("Kepala", "Sedang", f"Kepala maju index {hd:.2f}. Indikasi text neck.", hd))
    elif hd >= 0.15:
        findings.append(_finding("Kepala", "Ringan", f"Kepala sedikit maju index {hd:.2f}.", hd))

    sh_mod = sh >= 5.0; hi_mod = hi >= 5.0; hd_mod = hd >= 0.35
    sh_mil = sh >= 2.0; hi_mil = hi >= 2.0; hd_mil = hd >= 0.2
    mod_c  = sum([sh_mod, hi_mod, hd_mod])
    mil_c  = sum([sh_mil, hi_mil, hd_mil])

    if mod_c >= 2 or (mod_c == 1 and mil_c >= 2):
        score, cat = 60.0, "ATTENTION"
        summary = f"Skor postur {score:.0f}/100 — PERLU PERHATIAN. Ditemukan {len(findings)} area deviasi."
    elif mil_c >= 1:
        score, cat = 80.0, "FAIR"
        summary = f"Skor postur {score:.0f}/100 — CUKUP BAIK. {len(findings)} area perlu dipantau."
    else:
        score, cat = 95.0, "GOOD"
        summary = "Skor postur 95/100 — BAIK. Postur simetris dan seimbang."
        findings.append(_finding("Postur Keseluruhan", "Baik", "Tidak ditemukan deviasi bermakna.", 0))

    return score, cat, summary, findings

def rule_based_scoring_back(m):
    sh, hi = m["shoulder_tilt_index"], m["hip_tilt_index"]
    findings = []
    score    = 100.0 - sh * 12.0 - hi * 10.0
    score    = max(0.0, min(100.0, score))

    for val, mild, mod, label in [(sh, 1.5, 3.5, "Bahu"), (hi, 1.5, 3.5, "Panggul")]:
        if val >= mod:
            findings.append(_finding(label, "Sedang", f"Kemiringan {label.lower()} {val:.1f}% dari belakang.", val))
        elif val >= mild:
            findings.append(_finding(label, "Ringan", f"Sedikit asimetri {label.lower()} {val:.1f}%.", val))

    if score >= 80:
        cat, summary = "GOOD", f"Skor {score:.0f}/100 — BAIK dari belakang."
        if not findings:
            findings.append(_finding("Postur Keseluruhan", "Baik", "Simetris dari belakang.", 0))
    elif score >= 73:
        cat, summary = "FAIR", f"Skor {score:.0f}/100 — CUKUP BAIK. {len(findings)} area dipantau."
    else:
        cat, summary = "ATTENTION", f"Skor {score:.0f}/100 — PERLU PERHATIAN."

    return score, cat, summary, findings

def rule_based_scoring_side(m):
    neck, torso = m["neck_inclination_deg"], m["torso_inclination_deg"]
    findings = []
    score    = 100.0

    if neck >= 20:
        findings.append(_finding("Leher", "Sedang", f"Leher menunduk {neck:.1f}°.", neck))
        score -= (neck - 20) * 2.0
    elif neck >= 10:
        findings.append(_finding("Leher", "Ringan", f"Leher sedikit menunduk {neck:.1f}°.", neck))
        score -= (neck - 10) * 1.0

    if torso >= 20:
        findings.append(_finding("Batang Tubuh", "Sedang", f"Punggung bungkuk {torso:.1f}°.", torso))
        score -= (torso - 15) * 2.0
    elif torso >= 10:
        findings.append(_finding("Batang Tubuh", "Ringan", f"Punggung sedikit bungkuk {torso:.1f}°.", torso))
        score -= (torso - 10) * 1.0

    score = max(0.0, min(100.0, score))

    if score >= 85:
        cat, summary = "GOOD", f"Skor {score:.0f}/100 — BAIK dari samping."
        if not findings:
            findings.append(_finding("Postur Keseluruhan", "Baik", "Leher dan punggung tegak.", 0))
    elif score >= 70:
        cat, summary = "FAIR", f"Skor {score:.0f}/100 — CUKUP BAIK."
    else:
        cat, summary = "ATTENTION", f"Skor {score:.0f}/100 — PERLU PERHATIAN."

    return score, cat, summary, findings

# ─── OVERLAY ──────────────────────────────────────────────────────────────────

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

def draw_skeleton(overlay, landmarks, w, h, color):
    for s, e in POSE_CONNECTIONS:
        if s >= len(landmarks) or e >= len(landmarks):
            continue
        sl, el = landmarks[s], landmarks[e]
        if sl.visibility > 0.2 and el.visibility > 0.2:
            sp = (max(0, min(w-1, int(sl.x*w))), max(0, min(h-1, int(sl.y*h))))
            ep = (max(0, min(w-1, int(el.x*w))), max(0, min(h-1, int(el.y*h))))
            cv2.line(overlay, sp, ep, color, 3)
            cv2.circle(overlay, sp, 5, color, -1)
            cv2.circle(overlay, ep, 5, color, -1)

def create_posture_overlay(image, view, metrics, landmarks):
    overlay = image.copy()
    h, w    = overlay.shape[:2]
    score   = metrics.get("raw_score", 95)

    skel_color = (0, 255, 0)   if score >= 80 else (0, 165, 255)
    hi_color   = (0, 255, 255) if score >= 80 else (0, 0, 255)

    if landmarks:
        try:
            draw_skeleton(overlay, landmarks, w, h, skel_color)

            ls = landmarks[11]; rs = landmarks[12]
            lh = landmarks[23]; rh = landmarks[24]

            for a, b, c in [(ls, rs, hi_color), (lh, rh, hi_color)]:
                ap = (int(a.x*w), int(a.y*h))
                bp = (int(b.x*w), int(b.y*h))
                cv2.line(overlay, ap, bp, c, 8)
                cv2.circle(overlay, ap, 12, c, -1)
                cv2.circle(overlay, bp, 12, c, -1)

            mid_x = (int(ls.x*w) + int(rs.x*w)) // 2
            cv2.line(overlay, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)
        except Exception as ex:
            print(f"[DRAW ERROR] {ex}", flush=True)

    # Info box
    cv2.rectangle(overlay, (10, 10), (320, 190), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (320, 190), (255, 255, 255), 2)
    cv2.putText(overlay, f"Score: {score:.0f}", (25, 50),  cv2.FONT_HERSHEY_DUPLEX, 1.3, (255,255,255), 3)
    cv2.putText(overlay, f"View: {view}",       (25, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    cv2.putText(overlay, f"Shoulder: {metrics.get('shoulder_tilt_index',0):.1f}%", (25,125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(overlay, f"Hip: {metrics.get('hip_tilt_index',0):.1f}%",            (25,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    return overlay

def create_cropped_regions(image, landmarks, metrics, w, h):
    crops = []

    def crop_and_draw(lm_indices, color, label, value, text):
        pts_x = [int(landmarks[i].x * w) for i in lm_indices]
        pts_y = [int(landmarks[i].y * h) for i in lm_indices]
        pad   = 100
        x1, y1 = max(0, min(pts_x)-pad), max(0, min(pts_y)-pad)
        x2, y2 = min(w, max(pts_x)+pad), min(h, max(pts_y)+pad)
        if x2 <= x1 or y2 <= y1:
            return
        crop = image[y1:y2, x1:x2].copy()
        for i in lm_indices:
            pt = (int(landmarks[i].x*w - x1), int(landmarks[i].y*h - y1))
            cv2.circle(crop, pt, 12, color, -1)
        if len(lm_indices) == 2:
            p1 = (int(landmarks[lm_indices[0]].x*w - x1), int(landmarks[lm_indices[0]].y*h - y1))
            p2 = (int(landmarks[lm_indices[1]].x*w - x1), int(landmarks[lm_indices[1]].y*h - y1))
            cv2.line(crop, p1, p2, color, 6)
        cv2.putText(crop, text, (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
        crops.append((label, crop))

    sh = metrics.get("shoulder_tilt_index", 0)
    hi = metrics.get("hip_tilt_index", 0)
    hd = metrics.get("forward_head_index", 0)
    nk = metrics.get("neck_inclination_deg", 0)
    to = metrics.get("torso_inclination_deg", 0)

    if sh > 1.0:
        crop_and_draw([11, 12], (0,255,255), "shoulder", sh, f"BAHU MIRING {sh:.1f}%")
    if hi > 1.0:
        crop_and_draw([23, 24], (0,255,255), "hip",      hi, f"PANGGUL MIRING {hi:.1f}%")
    if hd > 0.2:
        crop_and_draw([0, 11, 12], (0,165,255), "head",  hd, f"KEPALA MAJU {hd:.2f}")
    if nk > 15:
        crop_and_draw([7, 11],  (0,165,255), "neck",  nk, f"LEHER MENUNDUK {nk:.1f} deg")
    if to > 15:
        crop_and_draw([11, 23], (0,100,255), "torso", to, f"PUNGGUNG BUNGKUK {to:.1f} deg")

    return crops

# ─── ANALYZE ──────────────────────────────────────────────────────────────────

def analyze_posture(image: np.ndarray, view: str = "FRONT"):
    landmarks = extract_landmarks(image)
    if landmarks is None:
        return 0.0, "UNKNOWN", {}, "Pose tidak terdeteksi, mohon unggah foto yang lebih jelas.", None

    view = (view or "FRONT").upper()

    if view == "BACK":
        metrics = compute_front_back_metrics(landmarks)
        score, category, summary, findings = rule_based_scoring_back(metrics)
    elif view == "SIDE":
        metrics = compute_side_view_metrics(landmarks)
        score, category, summary, findings = rule_based_scoring_side(metrics)
    else:
        metrics = compute_front_back_metrics(landmarks)
        score, category, summary, findings = rule_based_scoring_front(metrics)

    metrics["raw_score"] = score
    metrics["findings"]  = findings

    # Draw overlay
    h, w    = image.shape[:2]
    overlay = create_posture_overlay(image, view, metrics, landmarks)

    # Upload overlay ke Cloudinary
    overlay_url = upload_to_cloudinary(overlay, folder="potutrely/overlays")

    # Crop regions → upload masing-masing
    crop_results = []
    crops = create_cropped_regions(image, landmarks, metrics, w, h)
    for region_name, crop_img in crops:
        url = upload_to_cloudinary(crop_img, folder="potutrely/crops")
        crop_results.append({"region": region_name, "url": url})

    return score, category, metrics, summary, {
        "overlay_url": overlay_url,
        "crops": crop_results,
    }

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Potutrely Posture AI v1.0 — healthy ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze-posture", response_model=AnalyzeResponse)
def analyze_posture_endpoint(payload: AnalyzeRequest):
    print(f"[REQUEST] image_url={payload.image_url} view={payload.view}", flush=True)
    try:
        image = download_image(payload.image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal download gambar: {e}")

    try:
        score, category, metrics, summary, overlay_result = analyze_posture(image, view=payload.view)
    except Exception as e:
        print(f"[ERROR] analyze: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Gagal memproses gambcompute_side_view_metricsar.")

    return {
        "score":             score,
        "category":          category,
        "metrics":           metrics,
        "summary":           summary,
        "overlay_image_url": overlay_result["overlay_url"] if overlay_result else None,
        "crop_images":       overlay_result["crops"]       if overlay_result else [],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
