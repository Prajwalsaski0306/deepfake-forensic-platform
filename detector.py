"""
detector.py — Face detection and MediaPipe landmark mesh extraction.
Compatible with mediapipe >= 0.10.x (uses mp.solutions API with fallback).
ZERO Streamlit imports. Pure OpenCV / MediaPipe computation.
"""

import cv2
import numpy as np
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
FACE_SCALE_FACTOR: float = 1.1
FACE_MIN_NEIGHBORS: int = 5
FACE_MIN_SIZE: tuple = (60, 60)
MESH_COLOR_DOTS: tuple = (0, 255, 136)
MESH_COLOR_LINES: tuple = (0, 200, 100)
MESH_ALPHA: float = 0.55
MESH_DOT_RADIUS: int = 1
MESH_LINE_THICKNESS: int = 1

# ── MediaPipe import with version compat ─────────────────────────────────────
import mediapipe as mp

# mediapipe >= 0.10 restructured the API; solutions still works but
# must be imported carefully. Force the legacy solutions path.
try:
    from mediapipe.python.solutions import face_mesh as _fm_module
    _mp_connections = _fm_module
    _face_mesh_instance = _fm_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    )
    TESSELATION = _fm_module.FACEMESH_TESSELATION
    CONTOURS    = _fm_module.FACEMESH_CONTOURS
    try:
        IRISES = _fm_module.FACEMESH_IRISES
    except AttributeError:
        IRISES = set()
    MEDIAPIPE_OK = True
except Exception as _mp_err:
    MEDIAPIPE_OK = False
    _face_mesh_instance = None
    TESSELATION = []
    CONTOURS = []
    IRISES = []

# OpenCV Haar Cascade — loaded once
_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(img_rgb: np.ndarray) -> Optional[tuple]:
    """
    Detect the primary face using Haar Cascade.
    Returns (x, y, w, h) bounding box of largest face, or None.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    if len(faces) == 0:
        return None
    largest = max(faces, key=lambda r: r[2] * r[3])
    return (int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3]))


def extract_landmarks(img_rgb: np.ndarray) -> Optional[list]:
    """
    Extract 468 facial landmarks via MediaPipe FaceMesh.
    Returns list of (x_norm, y_norm) tuples, or None if unavailable.
    """
    if not MEDIAPIPE_OK or _face_mesh_instance is None:
        return None
    try:
        result = _face_mesh_instance.process(img_rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        return [(lm.x, lm.y) for lm in landmarks]
    except Exception:
        return None


def draw_landmark_mesh(img_rgb: np.ndarray, landmarks: list) -> np.ndarray:
    """Draw 468-point landmark mesh overlay on a copy of  the image."""
    h, w = img_rgb.shape[:2]
    overlay = img_rgb.copy()
    canvas = img_rgb.copy()
    pts = [(int(x * w), int(y * h)) for (x, y) in landmarks]

    for conn_set, color, thick in [
        (TESSELATION, MESH_COLOR_LINES, MESH_LINE_THICKNESS),
        (CONTOURS,    (0, 180, 255),    1),
        (IRISES,      (0, 200, 255),    1),
    ]:
        for (i, j) in conn_set:
            if i < len(pts) and j < len(pts):
                cv2.line(overlay, pts[i], pts[j], color, thick)

    for pt in pts:
        cv2.circle(overlay, pt, MESH_DOT_RADIUS, MESH_COLOR_DOTS, -1)

    return cv2.addWeighted(overlay, MESH_ALPHA, canvas, 1 - MESH_ALPHA, 0)


def draw_face_bbox(img_rgb: np.ndarray, bbox: tuple, is_fake: bool) -> np.ndarray:
    """Draw coloured bounding box + corner accents around the detected face."""
    result = img_rgb.copy()
    x, y, w, h = bbox
    color = (255, 51, 51) if is_fake else (0, 255, 136)
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    cl = min(w, h) // 6
    for cx, cy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1), (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
        cv2.line(result, (cx, cy), (cx + dx * cl, cy), color, 3)
        cv2.line(result, (cx, cy), (cx, cy + dy * cl), color, 3)
    cv2.putText(result, "FACE DETECTED", (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return result


def get_face_roi(img_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
    """Crop and return the face region of interest."""
    x, y, w, h = bbox
    H, W = img_rgb.shape[:2]
    return img_rgb[max(0, y):min(H, y+h), max(0, x):min(W, x+w)]
