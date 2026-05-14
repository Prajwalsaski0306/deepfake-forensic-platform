"""
scorer.py — Combine 5 engine scores into a final verdict + risk level.
ZERO Streamlit imports. Pure math and dicts.
"""

# ── Engine Weights ────────────────────────────────────────────────────────────
WEIGHTS = {
    "ela":       0.25,
    "fft":       0.20,
    "geometry":  0.20,
    "texture":   0.20,
    "color":     0.15,
}

# ── Thresholds ────────────────────────────────────────────────────────────────
FAKE_THRESHOLD: float = 50.0

RISK_LEVELS = [
    (30.0,  "LOW RISK",      "Very likely authentic"),
    (50.0,  "MODERATE RISK", "Some anomalies detected"),
    (75.0,  "HIGH RISK",     "Multiple fake signals found"),
    (100.1, "CRITICAL",      "Strong deepfake indicators"),
]


def compute_verdict(engine_scores: dict) -> dict:
    """
    Combine engine scores using weighted sum into a final forensics verdict.

    Args:
        engine_scores: dict mapping engine key → score (0-100).
                       Keys: 'ela', 'fft', 'geometry', 'texture', 'color'.
                       Missing keys are redistributed proportionally.

    Returns:
        dict with keys:
            fake_score   (float 0–100)
            verdict      (str: 'DEEPFAKE DETECTED' | 'AUTHENTIC IMAGE')
            is_fake      (bool)
            confidence   (float 0–100)
            risk_level   (str)
            risk_desc    (str)
            weights_used (dict)
    """
    # Handle missing engines — redistribute their weight equally among active ones
    active = {k: v for k, v in engine_scores.items() if k in WEIGHTS and v is not None}
    missing_weight = sum(w for k, w in WEIGHTS.items() if k not in active)

    if len(active) == 0:
        return _build_result(50.0, engine_scores)

    extra_per_engine = missing_weight / len(active)
    weights_used = {k: WEIGHTS[k] + extra_per_engine for k in active}
    total_w = sum(weights_used.values())

    fake_score = sum(active[k] * weights_used[k] for k in active) / total_w
    return _build_result(fake_score, weights_used)


def integrate_face_comparison(verdict: dict, face_cmp: dict) -> dict:
    """
    Adjust the verdict by incorporating the face comparison result.

    Rules:
      - Real image + mismatch  → escalate risk to HIGH RISK, flag as possible face swap.
      - Fake image + mismatch  → escalate risk to CRITICAL.
      - Real image + match     → keep verdict as-is (AUTHENTIC confirmed).
      - Fake image + match     → keep verdict, add note (suspicious despite match).

    Returns an updated copy of the verdict dict.
    """
    v = dict(verdict)
    is_fake = v.get("is_fake", False)
    match   = face_cmp.get("match", True)   # default True = no penalty if engine failed

    if not is_fake and not match:
        # Looks real but face mismatch → possible face swap
        v["risk_level"] = "HIGH RISK"
        v["risk_desc"]  = "Possible face swap — identity mismatch on otherwise authentic image"
        # Nudge fake_score up slightly so downstream consumers see the flag
        v["fake_score"] = min(100.0, round(v["fake_score"] + 15.0, 1))
        v["confidence"] = round(v["fake_score"], 1)
    elif is_fake and not match:
        # Both fake signals and mismatch → CRITICAL
        v["risk_level"] = "CRITICAL"
        v["risk_desc"]  = "Strong deepfake indicators AND identity mismatch detected"

    v["face_comparison"] = face_cmp
    return v


def _build_result(fake_score: float, weights_used: dict) -> dict:
    is_fake = fake_score > FAKE_THRESHOLD
    confidence = fake_score if is_fake else (100.0 - fake_score)
    verdict = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC IMAGE"

    risk_level = "CRITICAL"
    risk_desc = "Strong deepfake indicators"
    for threshold, level, desc in RISK_LEVELS:
        if fake_score <= threshold:
            risk_level = level
            risk_desc = desc
            break

    return {
        "fake_score":   round(fake_score, 1),
        "verdict":      verdict,
        "is_fake":      is_fake,
        "confidence":   round(confidence, 1),
        "risk_level":   risk_level,
        "risk_desc":    risk_desc,
        "weights_used": weights_used,
    }


def build_report_text(
    scan_id: str,
    filename: str,
    resolution: str,
    engine_results: dict,
    verdict_result: dict,
    timestamp: str,
) -> str:
    """
    Generate a plain-text forensics report suitable for download.

    Args:
        scan_id:        Forensics scan identifier string.
        filename:       Original uploaded filename.
        resolution:     Image resolution string (e.g. '512 × 512').
        engine_results: Dict of engine result dicts keyed by engine name.
        verdict_result: Output of compute_verdict().
        timestamp:      ISO-formatted timestamp string.

    Returns:
        Multi-line string of the forensics report.
    """
    sep = "─" * 45

    engines_completed = sum(1 for v in engine_results.values() if v.get("error") is None)
    total_engines = len(engine_results)

    def score_label(s):
        if s >= 70:
            return "SUSPICIOUS"
        elif s >= 50:
            return "MODERATE"
        elif s >= 30:
            return "LOW RISK"
        else:
            return "LIKELY REAL"

    engine_lines = []
    labels = {
        "ela":      "ELA Analysis",
        "fft":      "FFT Analysis",
        "geometry": "Landmark Geometry",
        "texture":  "Texture Entropy",
        "color":    "Color Consistency",
    }
    for idx, (key, label) in enumerate(labels.items(), 1):
        res = engine_results.get(key, {})
        score = res.get("score", 0)
        engine_lines.append(
            f"[{idx}] {label:<20} → Score: {score:>5.1f}/100 — {score_label(score)}"
        )

    report = f"""
DEEPFAKE FORENSICS REPORT
{sep}
Scan ID      : {scan_id}
Timestamp    : {timestamp}
File         : {filename}
Resolution   : {resolution}
Analysis     : {engines_completed}/{total_engines} Engines Completed
{sep}
ENGINE RESULTS:
{chr(10).join(engine_lines)}
{sep}
FINAL VERDICT      : {verdict_result['verdict']}
CONFIDENCE         : {verdict_result['confidence']}%
FAKE SCORE         : {verdict_result['fake_score']}/100
RISK LEVEL         : {verdict_result['risk_level']}
RISK DESCRIPTION   : {verdict_result['risk_desc']}
{sep}
⚠️  DISCLAIMER: This tool is for educational and research purposes only.
Results are based on forensic heuristics and should not be used as
sole evidence of image manipulation.
{sep}
""".strip()

    return report
