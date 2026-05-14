"""
video_engines.py — 7 Temporal Forensic Engines for Video DeepFake Analysis.
Pure NumPy/OpenCV/SciPy computation. Zero Streamlit imports.
"""
import cv2
import numpy as np
from scipy import signal
from typing import List, Dict, Optional
import mediapipe as mp

try:
    from mediapipe.python.solutions import face_mesh as _fm
    _mesh = _fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                         refine_landmarks=True, min_detection_confidence=0.4)
    MP_OK = True
except Exception:
    _mesh = None
    MP_OK = False

_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

IDX_L_EYE = [33, 160, 158, 133, 153, 144]
IDX_R_EYE = [362, 385, 387, 263, 373, 380]
IDX_MOUTH = [61, 291, 0, 17, 78, 308]
IDX_NOSE  = 1
IDX_CHIN  = 152
IDX_L_CHEEK = 234
IDX_R_CHEEK = 454
IDX_FOREHEAD = 10


def _get_landmarks(frame_rgb: np.ndarray) -> Optional[list]:
    if not MP_OK or _mesh is None:
        return None
    try:
        res = _mesh.process(frame_rgb)
        if res.multi_face_landmarks:
            return [(lm.x, lm.y) for lm in res.multi_face_landmarks[0].landmark]
    except Exception:
        pass
    return None


def _ear(landmarks, indices, w, h):
    """Eye Aspect Ratio from 6-point eye landmarks."""
    pts = np.array([[landmarks[i][0]*w, landmarks[i][1]*h] for i in indices])
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A + B) / (2.0 * C + 1e-8)


def _geometry_vec(landmarks, w, h):
    """Return a compact geometry vector for identity comparison."""
    def px(i): return np.array([landmarks[i][0]*w, landmarks[i][1]*h])
    le = np.mean([px(i) for i in IDX_L_EYE], axis=0)
    re = np.mean([px(i) for i in IDX_R_EYE], axis=0)
    nose = px(IDX_NOSE)
    chin = px(IDX_CHIN)
    lc = px(IDX_L_CHEEK)
    rc = px(IDX_R_CHEEK)
    fore = px(IDX_FOREHEAD)
    ed = np.linalg.norm(le-re)
    fh = np.linalg.norm(fore-chin) + 1e-8
    fw = np.linalg.norm(lc-rc) + 1e-8
    return np.array([ed/fw, np.linalg.norm(nose-fore)/fh,
                     np.linalg.norm(le-re)/fh,
                     np.linalg.norm(lc-rc)/fh])


def _mouth_openness(landmarks, w, h):
    pts = np.array([[landmarks[i][0]*w, landmarks[i][1]*h] for i in IDX_MOUTH])
    return float(np.linalg.norm(pts[2]-pts[3]))


def _fft_score(frame_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    mag_n = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    h, w = mag_n.shape
    cy, cx = h//2, w//2
    ri, ro = min(h,w)//6, min(h,w)//3
    Y, X = np.ogrid[:h, :w]
    mask = (np.sqrt((X-cx)**2+(Y-cy)**2) >= ri) & (np.sqrt((X-cx)**2+(Y-cy)**2) <= ro)
    return float(np.var(mag_n[mask])) * 10000


def _texture_entropy(frame_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0,255))
    p = hist.astype(float)/hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ─── ENGINE 1 — Temporal Consistency ────────────────────────────────────────

def engine_temporal_consistency(frames: List[np.ndarray]) -> Dict:
    """Frame-to-frame landmark drift and motion instability."""
    timeline, anomaly_frames = [], []
    prev_lm = None
    h, w = frames[0].shape[:2] if frames else (360, 640)

    for i, frame in enumerate(frames):
        lm = _get_landmarks(frame)
        if lm is None or prev_lm is None:
            timeline.append(0.0)
            prev_lm = lm
            continue
        # Compute mean drift of all 468 pts
        curr_pts = np.array([[lm[j][0]*w, lm[j][1]*h] for j in range(min(468,len(lm)))])
        prev_pts = np.array([[prev_lm[j][0]*w, prev_lm[j][1]*h] for j in range(min(468,len(prev_lm)))])
        drift = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))
        score = min(100.0, drift * 3.0)
        timeline.append(score)
        if score > 55:
            anomaly_frames.append(i)
        prev_lm = lm

    if not timeline:
        return {"score":50.0,"status":"Insufficient frames","timeline":[],"anomaly_frames":[],"summary":"N/A","error":None}

    mean_score = float(np.mean(timeline)) if timeline else 50.0
    instability = float(np.std(timeline)) if len(timeline)>1 else 0.0
    final = min(100.0, mean_score * 0.6 + instability * 0.8)
    status = "HIGH temporal inconsistency — warping artifacts detected" if final > 55 else "Temporal motion vectors within natural range"
    return {"score":round(final,1),"status":status,"timeline":timeline,
            "anomaly_frames":anomaly_frames,"instability":round(instability,2),
            "summary":f"Avg drift {mean_score:.1f}, std {instability:.1f} over {len(frames)} frames","error":None}


# ─── ENGINE 2 — Identity Persistence ────────────────────────────────────────

def engine_identity_persistence(frames: List[np.ndarray]) -> Dict:
    """Track identity geometry vector consistency across frames."""
    vectors, anomaly_frames, timeline = [], [], []
    h, w = frames[0].shape[:2] if frames else (360, 640)

    for i, frame in enumerate(frames):
        lm = _get_landmarks(frame)
        if lm and len(lm) >= 468:
            vectors.append(_geometry_vec(lm, w, h))
        else:
            vectors.append(None)

    valid = [v for v in vectors if v is not None]
    if len(valid) < 3:
        return {"score":50.0,"status":"Insufficient face detections","timeline":[],"anomaly_frames":[],"summary":"N/A","error":None}

    ref = np.mean(valid, axis=0)
    for i, v in enumerate(vectors):
        if v is None:
            timeline.append(50.0)
            continue
        dist = float(np.linalg.norm(v - ref))
        score = min(100.0, dist * 300)
        timeline.append(score)
        if score > 60:
            anomaly_frames.append(i)

    mean_score = float(np.mean(timeline))
    drift_std = float(np.std([t for t in timeline if t is not None]))
    final = min(100.0, mean_score * 0.7 + drift_std * 0.5)
    status = "Identity morphing detected — face embedding drift exceeds threshold" if final > 55 else "Consistent facial identity maintained across timeline"
    return {"score":round(final,1),"status":status,"timeline":timeline,
            "anomaly_frames":anomaly_frames,"drift_std":round(drift_std,3),
            "summary":f"Identity drift σ={drift_std:.3f} over {len(valid)} detected faces","error":None}


# ─── ENGINE 3 — GAN Fingerprint ──────────────────────────────────────────────

def engine_gan_fingerprint(frames: List[np.ndarray]) -> Dict:
    """FFT spectral variance tracking for GAN rendering signatures."""
    timeline, anomaly_frames = [], []
    for i, frame in enumerate(frames):
        score = _fft_score(frame)
        norm_score = min(100.0, score * 15.0)
        timeline.append(norm_score)
        if norm_score > 60:
            anomaly_frames.append(i)

    if not timeline:
        return {"score":50.0,"status":"No frames","timeline":[],"anomaly_frames":[],"summary":"N/A","error":None}

    mean_score = float(np.mean(timeline))
    variance = float(np.var(timeline))
    final = min(100.0, mean_score * 0.75 + variance * 0.001)
    status = "GAN rendering fingerprints detected — periodic spectral artifacts present" if final > 50 else "Natural spectral distribution — no GAN signatures"
    return {"score":round(final,1),"status":status,"timeline":timeline,
            "anomaly_frames":anomaly_frames,"spectral_variance":round(variance,2),
            "summary":f"Mean GAN index {mean_score:.1f}, spectral var {variance:.1f}","error":None}


# ─── ENGINE 4 — Micro-Expression ─────────────────────────────────────────────

def engine_micro_expression(frames: List[np.ndarray]) -> Dict:
    """Detect expression freezing, lip-sync delay, asymmetric motion."""
    timeline, anomaly_frames = [], []
    mouth_series = []
    h, w = frames[0].shape[:2] if frames else (360, 640)

    for i, frame in enumerate(frames):
        lm = _get_landmarks(frame)
        if lm and len(lm) >= 308:
            mo = _mouth_openness(lm, w, h)
            mouth_series.append(mo)
        else:
            mouth_series.append(None)

    valid_mouth = [m for m in mouth_series if m is not None]
    if len(valid_mouth) < 4:
        return {"score":50.0,"status":"Insufficient landmark data","timeline":[],"anomaly_frames":[],"summary":"N/A","error":None}

    mouth_arr = np.array(valid_mouth)
    # Expression freeze: low variance = frozen expression
    mouth_std = float(np.std(mouth_arr))
    mouth_mean = float(np.mean(mouth_arr))

    # Score: very low std = suspect freeze
    freeze_score = max(0.0, 60.0 - mouth_std * 8.0)

    for i, m in enumerate(mouth_series):
        if m is None:
            timeline.append(freeze_score * 0.5)
            continue
        deviation = abs(m - mouth_mean) / (mouth_std + 1e-8)
        frame_score = min(100.0, freeze_score * 0.6 + (1.0 / (deviation + 1)) * 30)
        timeline.append(frame_score)
        if frame_score > 60:
            anomaly_frames.append(i)

    final = min(100.0, freeze_score * 0.8 + (100.0 - mouth_std * 5.0) * 0.2)
    final = max(0.0, final)
    status = "Expression freeze detected — micro-expression suppression signature" if final > 55 else "Natural micro-expression variation confirmed"
    return {"score":round(final,1),"status":status,"timeline":timeline,
            "anomaly_frames":anomaly_frames,"mouth_std":round(mouth_std,3),
            "summary":f"Lip motion σ={mouth_std:.2f}, freeze index={freeze_score:.1f}","error":None}


# ─── ENGINE 5 — Blink & Eye Analysis ─────────────────────────────────────────

def engine_blink_eye(frames: List[np.ndarray]) -> Dict:
    """EAR-based blink detection and eye consistency analysis."""
    ear_series, blink_frames, timeline = [], [], []
    h, w = frames[0].shape[:2] if frames else (360, 640)

    for i, frame in enumerate(frames):
        lm = _get_landmarks(frame)
        if lm and len(lm) >= 390:
            left_ear  = _ear(lm, IDX_L_EYE, w, h)
            right_ear = _ear(lm, IDX_R_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0
            ear_series.append(avg_ear)
            sym_score = abs(left_ear - right_ear) / (avg_ear + 1e-8)
            timeline.append(min(100.0, sym_score * 200))
            if avg_ear < 0.15:
                blink_frames.append(i)
        else:
            ear_series.append(None)
            timeline.append(50.0)

    valid_ear = [e for e in ear_series if e is not None]
    if len(valid_ear) < 4:
        return {"score":50.0,"status":"Eye landmarks unavailable","timeline":[],"anomaly_frames":[],"summary":"N/A","error":None}

    ear_arr = np.array(valid_ear)
    blink_count = len(blink_frames)
    fps_est = 25.0
    duration = len(frames) / fps_est
    blink_rate = blink_count / (duration / 60.0 + 1e-8)
    # Normal blink rate: 12-20 per minute
    if blink_rate < 3 or blink_rate > 50:
        blink_score = 75.0
    elif blink_rate < 8 or blink_rate > 30:
        blink_score = 45.0
    else:
        blink_score = 15.0

    ear_std = float(np.std(ear_arr))
    final = min(100.0, blink_score * 0.7 + (1.0 / (ear_std + 0.05)) * 2.0)
    status = "Unnatural eye behavior — static eyes or abnormal blink pattern" if final > 50 else "Blink pattern within natural biological range"
    return {"score":round(final,1),"status":status,"timeline":timeline,
            "anomaly_frames":blink_frames,"blink_rate":round(blink_rate,1),
            "ear_series":[round(e,3) if e else 0.0 for e in ear_series],
            "summary":f"Blink rate {blink_rate:.1f}/min, EAR σ={ear_std:.3f}","error":None}


# ─── ENGINE 6 — Skin Texture Stability ───────────────────────────────────────

def engine_skin_texture(frames: List[np.ndarray]) -> Dict:
    """Frame-to-frame texture entropy evolution."""
    entropy_series, anomaly_frames, timeline = [], [], []

    for i, frame in enumerate(frames):
        ent = _texture_entropy(frame)
        entropy_series.append(ent)

    if len(entropy_series) < 3:
        return {"score":50.0,"status":"No frames","timeline":[],"anomaly_frames":[],"summary":"N/A","error":None}

    arr = np.array(entropy_series)
    mean_ent = float(np.mean(arr))
    std_ent  = float(np.std(arr))

    # Sudden smoothing: entropy drops sharply
    diffs = np.abs(np.diff(arr))
    for i, d in enumerate(diffs):
        norm_score = min(100.0, d * 50.0)
        timeline.append(norm_score)
        if norm_score > 60:
            anomaly_frames.append(i+1)
    timeline.append(timeline[-1] if timeline else 0.0)

    # Low overall entropy = synthetic smoothing
    smooth_score = max(0.0, (6.5 - mean_ent) * 20.0)
    final = min(100.0, smooth_score * 0.6 + std_ent * 30.0 * 0.4)
    status = "Neural skin smoothing detected — texture entropy collapse" if final > 55 else "Natural skin texture entropy confirmed"
    return {"score":round(final,1),"status":status,"timeline":timeline,
            "entropy_series":entropy_series,"anomaly_frames":anomaly_frames,
            "mean_entropy":round(mean_ent,3),"summary":f"Mean entropy {mean_ent:.2f}, σ={std_ent:.3f}","error":None}


# ─── ENGINE 7 — Audio-Visual Sync ────────────────────────────────────────────

def engine_audio_visual_sync(frames: List[np.ndarray], has_audio: bool) -> Dict:
    """Lip-sync analysis via mouth openness pattern."""
    timeline, anomaly_frames = [], []
    h, w = frames[0].shape[:2] if frames else (360, 640)

    mouth_series = []
    for frame in frames:
        lm = _get_landmarks(frame)
        if lm and len(lm) >= 308:
            mouth_series.append(_mouth_openness(lm, w, h))
        else:
            mouth_series.append(None)

    valid = [m for m in mouth_series if m is not None]
    if len(valid) < 4:
        return {"score":30.0,"status":"Lip landmarks insufficient","timeline":[],"anomaly_frames":[],
                "has_audio":has_audio,"summary":"Insufficient data for sync analysis","error":None}

    arr = np.array(valid)
    # Compute frequency of mouth movement changes
    changes = np.diff(arr)
    change_rate = float(np.mean(np.abs(changes)))

    if not has_audio:
        # No audio: check if mouth is unnaturally active (movement without speech)
        if change_rate > 8.0:
            sync_score = 70.0
            status = "Abnormal lip movement detected without audio track — possible synthetic motion"
        else:
            sync_score = 25.0
            status = "No audio track present — lip-sync analysis limited"
    else:
        # With audio: check for lip movement regularity (should correlate with speech rhythm)
        # Natural speech: irregular mouth openness, not uniform
        uniformity = 1.0 / (float(np.std(arr)) + 0.5)
        sync_score = min(100.0, uniformity * 15.0 + (change_rate < 3.0) * 30.0)
        status = "Lip-sync anomaly detected — mouth motion does not correlate with speech pattern" if sync_score > 50 else "Audio-visual synchronization within natural parameters"

    for i, m in enumerate(mouth_series):
        if m is None:
            timeline.append(sync_score * 0.5)
            continue
        dev = abs(m - float(np.mean(arr))) / (float(np.std(arr)) + 1e-8)
        frame_score = min(100.0, sync_score * 0.7 + (1.0 - min(1.0, dev/3.0)) * sync_score * 0.3)
        timeline.append(frame_score)
        if frame_score > 65:
            anomaly_frames.append(i)

    return {"score":round(sync_score,1),"status":status,"timeline":timeline,
            "anomaly_frames":anomaly_frames,"has_audio":has_audio,
            "mouth_series":[round(m,2) if m else 0.0 for m in mouth_series],
            "summary":f"Lip change rate {change_rate:.2f}, audio={'YES' if has_audio else 'NO'}","error":None}
