"""
video_scorer.py — Aggregate 7 engine scores into enterprise forensic verdict.
"""
import datetime
import random
import string
from typing import Dict

# Engine weights (must sum to 1.0)
ENGINE_WEIGHTS = {
    "temporal":   0.20,
    "identity":   0.20,
    "gan":        0.18,
    "micro_expr": 0.12,
    "blink":      0.10,
    "texture":    0.10,
    "av_sync":    0.10,
}

FAKE_THRESHOLD = 52.0  # composite score above this = DEEPFAKE


def compute_video_verdict(engine_results: Dict) -> Dict:
    """
    Aggregate all engine scores into a final forensic verdict.

    Args:
        engine_results: dict keyed by engine name with 'score' field.

    Returns:
        Full verdict dict with all forensic indices.
    """
    keys = ["temporal", "identity", "gan", "micro_expr", "blink", "texture", "av_sync"]
    scores = {}
    for k in keys:
        r = engine_results.get(k, {})
        scores[k] = r.get("score", 50.0)

    # Weighted composite
    composite = sum(scores[k] * ENGINE_WEIGHTS[k] for k in keys)

    # Neural anomaly index: proportion of engines flagging > 55
    flagged = sum(1 for s in scores.values() if s > 55)
    neural_anomaly_index = round(flagged / len(scores) * 100, 1)

    # Temporal instability: from temporal engine's instability value
    temporal_instability = engine_results.get("temporal", {}).get("instability", 0.0)

    is_fake = composite >= FAKE_THRESHOLD

    # Risk classification
    if composite >= 80:
        risk = "CRITICAL"
        threat = "EXTREME"
    elif composite >= 65:
        risk = "HIGH RISK"
        threat = "HIGH"
    elif composite >= 50:
        risk = "MODERATE RISK"
        threat = "MODERATE"
    elif composite >= 35:
        risk = "LOW RISK"
        threat = "LOW"
    else:
        risk = "MINIMAL RISK"
        threat = "MINIMAL"

    # Confidence: distance from threshold, scaled to 50–99%
    dist = abs(composite - FAKE_THRESHOLD)
    confidence = min(99, int(50 + dist * 1.5))

    # Scan ID
    scan_id = "VFI-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "is_fake": is_fake,
        "composite_score": round(composite, 1),
        "confidence": confidence,
        "risk_level": risk,
        "threat_level": threat,
        "neural_anomaly_index": neural_anomaly_index,
        "temporal_instability_score": round(temporal_instability, 2),
        "flagged_engines": flagged,
        "engine_scores": scores,
        "scan_id": scan_id,
        "timestamp": timestamp,
    }
