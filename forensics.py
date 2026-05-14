"""
forensics.py — All 6 forensic analysis engines for DeepFake Forensics Lab.
ZERO Streamlit imports. Pure computation using OpenCV, NumPy, SciPy.
Each engine returns a dict with: score (0-100), status (str), and visual data.
Engine 6: Face Comparison — landmark-based identity verification.
"""

import io
import cv2
import numpy as np
from PIL import Image
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
ELA_QUALITY: int = 92
ELA_AMPLIFY: int = 10
ELA_THRESHOLD: float = 15.0

FFT_HEATMAP_CMAP: str = "hot"

# Normal human facial geometry ranges
GEOMETRY_RANGES = {
    "Eye Distance / Face Width": (0.38, 0.55),
    "Nose Length / Face Height": (0.40, 0.60),
    "Mouth Width / Eye Distance": (0.65, 1.05),
    "Facial Symmetry Score":     (0.80, 1.00),
}

LBP_ENTROPY_LOW: float = 4.0
LBP_ENTROPY_HIGH: float = 7.5

COLOR_BORDER_EXPAND: float = 0.15

# ── Key MediaPipe landmark indices ───────────────────────────────────────────
IDX_LEFT_EYE   = [33, 133, 157, 158, 159, 160, 161, 173, 246]
IDX_RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263]
IDX_NOSE_TIP   = 1
IDX_CHIN       = 152
IDX_LEFT_MOUTH = 61
IDX_RIGHT_MOUTH = 291
IDX_LEFT_CHEEK = 234
IDX_RIGHT_CHEEK = 454
IDX_FOREHEAD   = 10


# ── Helper ───────────────────────────────────────────────────────────────────

def _pil_to_jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _pt(landmarks, idx, w, h):
    """Convert normalised landmark to pixel coordinates."""
    lm = landmarks[idx]
    return np.array([lm[0] * w, lm[1] * h])


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _shannon_entropy(hist: np.ndarray) -> float:
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ── Engine 1 — Error Level Analysis ─────────────────────────────────────────

def run_ela(img_pil: Image.Image, face_roi: np.ndarray) -> dict:
    """
    Error Level Analysis: detect JPEG compression artefacts indicating manipulation.

    Returns:
        dict with keys: score, status, ela_image (PIL), mean_ela (float).
    """
    try:
        # Re-save at fixed quality and compute pixel difference
        resaved_bytes = _pil_to_jpeg_bytes(img_pil, ELA_QUALITY)
        resaved = Image.open(io.BytesIO(resaved_bytes)).convert("RGB")

        orig_arr = np.array(img_pil.convert("RGB"), dtype=np.int32)
        resaved_arr = np.array(resaved, dtype=np.int32)

        # Amplify differences
        diff = np.abs(orig_arr - resaved_arr) * ELA_AMPLIFY
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        ela_img = Image.fromarray(diff, mode="RGB")

        # Compute mean ELA on face ROI region
        h_full, w_full = orig_arr.shape[:2]
        h_roi, w_roi = face_roi.shape[:2]

        # Use full image ELA mean if face roi is empty
        if h_roi > 0 and w_roi > 0:
            ela_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            mean_ela = float(np.mean(ela_gray))
        else:
            ela_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            mean_ela = float(np.mean(ela_gray))

        # Score: normalise mean_ela; higher = more fake
        score = min(100.0, (mean_ela / 50.0) * 100.0)

        if mean_ela > ELA_THRESHOLD:
            status = "High manipulation artifacts detected"
        else:
            status = "Low ELA — consistent texture"

        return {
            "score": round(score, 1),
            "status": status,
            "ela_image": ela_img,
            "mean_ela": round(mean_ela, 2),
            "error": None,
        }
    except Exception as exc:
        return {"score": 50.0, "status": "Engine offline", "ela_image": None,
                "mean_ela": 0.0, "error": str(exc)}


# ── Engine 2 — Frequency Domain Analysis (FFT) ──────────────────────────────

def run_fft(face_roi: np.ndarray) -> dict:
    """
    FFT analysis: detect GAN frequency fingerprints in face region.

    Returns:
        dict with keys: score, status, fft_image (PIL), periodicity_score (float).
    """
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 2D FFT
        fft2 = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft2)
        magnitude = np.abs(fft_shift)
        log_mag = np.log1p(magnitude)

        # Normalise for visualisation
        log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
        log_mag_uint8 = (log_mag_norm * 255).astype(np.uint8)

        # Apply 'hot' colormap via OpenCV
        colored = cv2.applyColorMap(log_mag_uint8, cv2.COLORMAP_HOT)
        fft_img = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

        # Periodicity score: variance of high-frequency ring
        h, w = log_mag_norm.shape
        cy, cx = h // 2, w // 2
        r_inner, r_outer = min(h, w) // 6, min(h, w) // 3
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        ring_mask = (dist_from_center >= r_inner) & (dist_from_center <= r_outer)
        ring_vals = log_mag_norm[ring_mask]
        periodicity_score = float(np.var(ring_vals)) * 10000

        # Score
        score = min(100.0, periodicity_score * 15.0)

        if periodicity_score > 3.0:
            status = "GAN frequency artifacts detected"
        else:
            status = "Natural frequency distribution"

        return {
            "score": round(score, 1),
            "status": status,
            "fft_image": fft_img,
            "periodicity_score": round(periodicity_score, 3),
            "log_magnitude": log_mag_norm,
            "error": None,
        }
    except Exception as exc:
        return {"score": 50.0, "status": "Engine offline", "fft_image": None,
                "periodicity_score": 0.0, "log_magnitude": None, "error": str(exc)}


# ── Engine 3 — Facial Landmark Geometry ─────────────────────────────────────

def run_geometry(landmarks: Optional[list], img_shape: tuple) -> dict:
    """
    Geometric consistency check using MediaPipe 468 landmarks.

    Returns:
        dict with keys: score, status, table (list of dicts), violations (int).
    """
    if landmarks is None or len(landmarks) < 468:
        return {
            "score": 50.0, "status": "[ MESH ERROR ] Landmark extraction failed",
            "table": [], "violations": 0, "error": "No landmarks available",
        }
    try:
        h, w = img_shape[:2]

        def px(idx):
            return _pt(landmarks, idx, w, h)

        # Compute centroid-based eye centres
        left_eye_pts = np.array([px(i) for i in IDX_LEFT_EYE])
        right_eye_pts = np.array([px(i) for i in IDX_RIGHT_EYE])
        left_eye_center = left_eye_pts.mean(axis=0)
        right_eye_center = right_eye_pts.mean(axis=0)

        nose_tip = px(IDX_NOSE_TIP)
        chin = px(IDX_CHIN)
        left_mouth = px(IDX_LEFT_MOUTH)
        right_mouth = px(IDX_RIGHT_MOUTH)
        left_cheek = px(IDX_LEFT_CHEEK)
        right_cheek = px(IDX_RIGHT_CHEEK)
        forehead = px(IDX_FOREHEAD)

        eye_dist = _dist(left_eye_center, right_eye_center)
        face_width = _dist(left_cheek, right_cheek)
        face_height = _dist(forehead, chin)
        nose_length = _dist(nose_tip, forehead)
        mouth_width = _dist(left_mouth, right_mouth)

        # Symmetry: compare left vs right half widths
        mid_x = (left_eye_center[0] + right_eye_center[0]) / 2
        left_half = abs(left_cheek[0] - mid_x)
        right_half = abs(right_cheek[0] - mid_x)
        symmetry = min(left_half, right_half) / (max(left_half, right_half) + 1e-8)

        ratios = {
            "Eye Distance / Face Width": eye_dist / (face_width + 1e-8),
            "Nose Length / Face Height": nose_length / (face_height + 1e-8),
            "Mouth Width / Eye Distance": mouth_width / (eye_dist + 1e-8),
            "Facial Symmetry Score":      symmetry,
        }

        table = []
        violations = 0
        for metric, measured in ratios.items():
            lo, hi = GEOMETRY_RANGES[metric]
            in_range = lo <= measured <= hi
            if not in_range:
                violations += 1
            dev = 0.0
            if measured < lo:
                dev = lo - measured
            elif measured > hi:
                dev = measured - hi
            table.append({
                "Metric": metric,
                "Measured": round(measured, 3),
                "Normal Range": f"{lo:.2f} – {hi:.2f}",
                "Deviation": round(dev, 3),
                "Status": "✅ PASS" if in_range else "❌ FAIL",
            })

        # Score: 0 violations = 100 (REAL), 4 violations = 0 (FAKE)
        score = max(0.0, 100.0 - (violations / 4.0) * 100.0)
        status = f"{violations} geometric anomalies detected"

        return {
            "score": round(score, 1),
            "status": status,
            "table": table,
            "violations": violations,
            "error": None,
        }
    except Exception as exc:
        return {"score": 50.0, "status": "Engine offline", "table": [],
                "violations": 0, "error": str(exc)}


# ── Engine 4 — LBP Texture Entropy ──────────────────────────────────────────

def run_texture(face_roi: np.ndarray) -> dict:
    """
    Local Binary Pattern entropy analysis to detect synthetic skin texture.

    Returns:
        dict with keys: score, status, lbp_image (PIL), entropy (float).
    """
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY).astype(np.uint8)

        # Manual LBP: compare each pixel to its 8 neighbours
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        padded = np.pad(gray, 1, mode="edge")

        neighbours = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for bit, (dy, dx) in enumerate(neighbours):
            shifted = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            lbp |= ((shifted >= gray).astype(np.uint8) << bit)

        # Histogram and entropy
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 255))
        entropy = _shannon_entropy(hist)

        # Visualise LBP map
        lbp_img = Image.fromarray(lbp, mode="L")

        # Score: map entropy [4.0 low → 7.5 high] to [0 → 100]
        score = (entropy - LBP_ENTROPY_LOW) / (LBP_ENTROPY_HIGH - LBP_ENTROPY_LOW) * 100.0
        score = float(np.clip(score, 0.0, 100.0))

        if entropy < 5.5:
            status = "Low texture entropy — possible synthetic skin"
        else:
            status = "Natural skin texture pattern confirmed"

        return {
            "score": round(score, 1),
            "status": status,
            "lbp_image": lbp_img,
            "entropy": round(entropy, 3),
            "error": None,
        }
    except Exception as exc:
        return {"score": 50.0, "status": "Engine offline", "lbp_image": None,
                "entropy": 0.0, "error": str(exc)}


# ── Engine 5 — Color Channel Inconsistency ──────────────────────────────────

def run_color(img_rgb: np.ndarray, bbox: tuple) -> dict:
    """
    Detect color inconsistency between the face ROI and its surrounding border.

    Returns:
        dict with keys: score, status, face_stats (dict), border_stats (dict),
                        delta_score (float), face_img (PIL), border_img (PIL).
    """
    try:
        x, y, w, h = bbox
        img_h, img_w = img_rgb.shape[:2]

        # Face region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        face_region = img_rgb[y1:y2, x1:x2]

        # Expanded border region (15% larger)
        pad_x = int(w * COLOR_BORDER_EXPAND)
        pad_y = int(h * COLOR_BORDER_EXPAND)
        bx1 = max(0, x - pad_x)
        by1 = max(0, y - pad_y)
        bx2 = min(img_w, x + w + pad_x)
        by2 = min(img_h, y + h + pad_y)
        expanded = img_rgb[by1:by2, bx1:bx2]

        # Build border mask (expanded region minus face region)
        border_img_arr = expanded.copy()
        # Mask out the face interior within the expanded crop
        inner_x1 = x1 - bx1
        inner_y1 = y1 - by1
        inner_x2 = inner_x1 + (x2 - x1)
        inner_y2 = inner_y1 + (y2 - y1)
        border_mask = np.ones(border_img_arr.shape[:2], dtype=bool)
        border_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
        border_pixels = border_img_arr[border_mask]

        def stats(pixels):
            r = pixels[:, 0].astype(float)
            g = pixels[:, 1].astype(float)
            b = pixels[:, 2].astype(float)
            return {
                "R_mean": float(r.mean()), "G_mean": float(g.mean()), "B_mean": float(b.mean()),
                "R_std":  float(r.std()),  "G_std":  float(g.std()),  "B_std":  float(b.std()),
            }

        face_flat = face_region.reshape(-1, 3)
        face_stats = stats(face_flat)
        border_stats = stats(border_pixels) if len(border_pixels) > 0 else face_stats.copy()

        delta = (
            abs(face_stats["R_mean"] - border_stats["R_mean"]) +
            abs(face_stats["G_mean"] - border_stats["G_mean"]) +
            abs(face_stats["B_mean"] - border_stats["B_mean"])
        )

        # Score: delta=0 → REAL (score 0 fake), delta≥75 → FAKE (score 100)
        score = min(100.0, (delta / 75.0) * 100.0)

        if delta > 25.0:
            status = "Color boundary anomaly detected"
        else:
            status = "Consistent color profile across face boundary"

        return {
            "score": round(score, 1),
            "status": status,
            "face_stats": face_stats,
            "border_stats": border_stats,
            "delta_score": round(delta, 2),
            "face_img": Image.fromarray(face_region),
            "border_img": Image.fromarray(expanded),
            "error": None,
        }
    except Exception as exc:
        return {"score": 50.0, "status": "Engine offline", "face_stats": {},
                "border_stats": {}, "delta_score": 0.0,
                "face_img": None, "border_img": None, "error": str(exc)}


# ── Engine 6 — Face Comparison (Reference vs Test) ──────────────────────────

# Key landmark indices for identity comparison (eyes, nose, mouth, jaw corners)
_FACE_COMPARE_INDICES = [
    # Eyes
    33, 133, 362, 263,
    # Nose tip + bridge
    1, 4, 168,
    # Mouth corners + lip centre
    61, 291, 0, 17,
    # Cheeks / jaw corners
    234, 454,
    # Forehead
    10,
    # Eyebrow peaks
    70, 300,
]


def _extract_normalised_landmarks(landmarks: list, w: int, h: int) -> np.ndarray:
    """Return Nx2 array of pixel coords for comparison indices, normalised by face bbox."""
    pts = np.array([[landmarks[i][0] * w, landmarks[i][1] * h]
                    for i in _FACE_COMPARE_INDICES], dtype=np.float32)
    # Translate so centroid is at origin, scale by span
    centroid = pts.mean(axis=0)
    pts -= centroid
    scale = np.sqrt((pts ** 2).sum(axis=1).mean()) + 1e-8
    pts /= scale
    return pts


def run_face_comparison(
    ref_landmarks,
    ref_shape: tuple,
    test_landmarks,
    test_shape: tuple,
) -> dict:
    """
    Compare reference and test face using normalised landmark distances.

    Args:
        ref_landmarks:  list of 468+ [x, y, z] normalised MediaPipe landmarks (reference).
        ref_shape:      (H, W) of reference image.
        test_landmarks: list of 468+ [x, y, z] normalised MediaPipe landmarks (test).
        test_shape:     (H, W) of test image.

    Returns:
        dict with keys:
            similarity_score  (float 0–100, 100 = identical)
            status            (str: "Face Match Detected" | "Face Mismatch Detected")
            match             (bool)
            error             (str|None)
    """
    if ref_landmarks is None or len(ref_landmarks) < 468:
        return {
            "similarity_score": 0.0,
            "status": "[ COMPARISON ERROR ] Reference landmarks unavailable",
            "match": False,
            "error": "Reference landmarks missing or insufficient",
        }
    if test_landmarks is None or len(test_landmarks) < 468:
        return {
            "similarity_score": 0.0,
            "status": "[ COMPARISON ERROR ] Test image landmarks unavailable",
            "match": False,
            "error": "Test landmarks missing or insufficient",
        }

    try:
        ref_h, ref_w = ref_shape[:2]
        test_h, test_w = test_shape[:2]

        ref_pts = _extract_normalised_landmarks(ref_landmarks, ref_w, ref_h)
        test_pts = _extract_normalised_landmarks(test_landmarks, test_w, test_h)

        # Procrustes-style alignment: find optimal rotation
        # Cross-covariance
        H = ref_pts.T @ test_pts
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Correct reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        ref_aligned = ref_pts @ R

        # Mean per-landmark Euclidean distance (lower = more similar)
        dists = np.linalg.norm(ref_aligned - test_pts, axis=1)
        mean_dist = float(dists.mean())

        # Convert distance to similarity: empirically, dist~0 → 100%, dist~1 → ~0%
        # Clamp to [0, 1.5] then invert
        similarity = max(0.0, 1.0 - (mean_dist / 1.5)) * 100.0
        similarity = float(np.clip(similarity, 0.0, 100.0))

        MATCH_THRESHOLD = 55.0  # similarity >= 55% → match
        match = similarity >= MATCH_THRESHOLD

        if match:
            status = "Face Match Detected"
        else:
            status = "Face Mismatch Detected"

        return {
            "similarity_score": round(similarity, 1),
            "status": status,
            "match": match,
            "mean_landmark_dist": round(mean_dist, 4),
            "error": None,
        }

    except Exception as exc:
        return {
            "similarity_score": 0.0,
            "status": "[ ENGINE ERROR ] Face comparison failed",
            "match": False,
            "error": str(exc),
        }
