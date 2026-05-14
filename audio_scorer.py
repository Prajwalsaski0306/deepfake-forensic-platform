"""audio_scorer.py - Compute audio forensic verdict from engine results"""
import datetime
import random

def compute_audio_verdict(engine_results: dict) -> dict:
    weights = {"spectral": 0.20, "mfcc": 0.22, "prosody": 0.18, "noise": 0.15, "biometric": 0.15, "artifact": 0.10}
    composite = 0.0
    for key, w in weights.items():
        r = engine_results.get(key, {})
        composite += r.get("score", 0) * w
    composite = round(composite, 1)
    is_fake = composite > 48
    confidence = min(99, max(51, int(50 + abs(composite - 48) * 1.4 + random.uniform(-3, 3))))
    if composite >= 75:   risk = "CRITICAL";      threat = "ALPHA"
    elif composite >= 58: risk = "HIGH RISK";     threat = "BRAVO"
    elif composite >= 42: risk = "MODERATE RISK"; threat = "CHARLIE"
    elif composite >= 25: risk = "LOW RISK";      threat = "DELTA"
    else:                 risk = "MINIMAL RISK";  threat = "ECHO"
    import uuid
    scan_id = f"AUD-{str(uuid.uuid4())[:8].upper()}"
    return {
        "is_fake": is_fake, "composite_score": int(composite), "confidence": confidence,
        "risk_level": risk, "threat_level": threat, "scan_id": scan_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
