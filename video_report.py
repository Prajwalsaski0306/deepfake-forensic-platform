"""
video_report.py — Intelligence-grade HTML forensic report generator for video analysis.
"""
import base64
import datetime
from io import BytesIO
from typing import Dict, List, Optional
import numpy as np
from PIL import Image


def _img_to_b64(img) -> str:
    if img is None:
        return ""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_video_html_report(
    metadata: Dict,
    engine_results: Dict,
    verdict: Dict,
    suspicious_frames: Optional[List[np.ndarray]] = None,
) -> str:
    is_fake = verdict["is_fake"]
    v_title = "DEEPFAKE CONFIRMED" if is_fake else "AUTHENTIC — NO DEEPFAKE DETECTED"
    v_color = "#ff3333" if is_fake else "#00ff88"
    risk = verdict["risk_level"]
    threat = verdict["threat_level"]
    composite = verdict["composite_score"]
    confidence = verdict["confidence"]
    scan_id = verdict["scan_id"]
    timestamp = verdict["timestamp"]
    flagged = verdict["flagged_engines"]
    nai = verdict["neural_anomaly_index"]
    tis = verdict["temporal_instability_score"]
    engine_scores = verdict.get("engine_scores", {})

    engine_names = {
        "temporal":   "Temporal Consistency Engine",
        "identity":   "Identity Persistence Engine",
        "gan":        "GAN Fingerprint Engine",
        "micro_expr": "Micro-Expression Engine",
        "blink":      "Blink & Eye Analysis Engine",
        "texture":    "Skin Texture Stability Engine",
        "av_sync":    "Audio-Visual Sync Engine",
    }

    rows = ""
    for k, name in engine_names.items():
        sc = engine_scores.get(k, 0)
        r = engine_results.get(k, {})
        status = r.get("status", "N/A")
        anomalies = len(r.get("anomaly_frames", []))
        color = "#ff3333" if sc > 65 else "#ffcc00" if sc > 40 else "#00ff88"
        rows += f"""
        <tr>
          <td>{name}</td>
          <td style="color:{color};font-weight:bold">{sc}/100</td>
          <td>{status}</td>
          <td>{anomalies} frames</td>
        </tr>"""

    frame_imgs = ""
    if suspicious_frames:
        for i, fr in enumerate(suspicious_frames[:6]):
            b64 = _img_to_b64(fr)
            if b64:
                frame_imgs += f'<div class="frame-card"><img src="data:image/png;base64,{b64}" /><div class="frame-label">ANOMALY FRAME {i+1}</div></div>'

    if is_fake:
        executive = (
            f"Forensic analysis of the submitted video evidence has identified statistically significant "
            f"temporal inconsistencies, spectral anomalies, and neural rendering fingerprints characteristic "
            f"of synthetic media generation. A total of {flagged} out of 7 independent forensic engines "
            f"exceeded detection thresholds. The composite manipulation index of {composite}/100 "
            f"with a neural anomaly index of {nai}% strongly indicates the presence of deepfake synthesis technology. "
            f"The temporal instability score of {tis} further corroborates frame-level identity inconsistencies "
            f"consistent with GAN-based face replacement or reenactment methods. "
            f"Classification: {risk} | Threat Level: {threat}."
        )
    else:
        executive = (
            f"Forensic analysis of the submitted video evidence reveals no statistically significant "
            f"indicators of synthetic media manipulation. All 7 forensic engines report scores within "
            f"natural biological and optical tolerance ranges. The composite authenticity index of "
            f"{100 - composite}/100 and neural anomaly index of {nai}% fall below detection thresholds. "
            f"Temporal motion vectors, facial identity embeddings, and spectral frequency distributions "
            f"are consistent with authentic video capture. "
            f"Classification: {risk} | Threat Level: {threat}."
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VIDEO DEEPFAKE FORENSIC INTELLIGENCE REPORT — {scan_id}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #060b14; color: #c9d1d9; font-family: 'Courier New', monospace; padding: 50px; line-height: 1.7; }}
    .header {{ text-align: center; border-bottom: 2px solid {v_color}; padding-bottom: 30px; margin-bottom: 40px; }}
    .header h1 {{ color: {v_color}; font-size: 26px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 8px; }}
    .header .sub {{ color: #8bb4e7; font-size: 13px; letter-spacing: 2px; }}
    .badge {{ display: inline-block; background: rgba(255,255,255,0.05); border: 1px solid {v_color};
              color: {v_color}; padding: 6px 16px; border-radius: 4px; font-size: 12px; margin: 4px; }}
    .verdict-box {{ border: 2px solid {v_color}; border-radius: 8px; padding: 30px;
                    background: rgba(0,0,0,0.4); margin-bottom: 40px; text-align: center; }}
    .verdict-box h2 {{ color: {v_color}; font-size: 28px; letter-spacing: 4px; margin-bottom: 16px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
    .stat-box {{ background: rgba(0,170,255,0.05); border: 1px solid rgba(0,170,255,0.2);
                 border-radius: 6px; padding: 16px; text-align: center; }}
    .stat-box .label {{ color: #718096; font-size: 11px; letter-spacing: 2px; margin-bottom: 6px; }}
    .stat-box .val {{ color: #e2e8f0; font-size: 20px; font-weight: bold; }}
    .section-title {{ color: #00aaff; font-size: 16px; border-bottom: 1px solid rgba(0,170,255,0.3);
                      padding-bottom: 8px; margin: 40px 0 20px; text-transform: uppercase; letter-spacing: 2px; }}
    p {{ color: #a0aec0; font-size: 14px; line-height: 1.8; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th {{ background: rgba(0,170,255,0.1); color: #00aaff; padding: 12px; text-align: left;
          border: 1px solid rgba(0,170,255,0.2); letter-spacing: 1px; }}
    td {{ background: rgba(10,16,28,0.5); padding: 12px; border: 1px solid rgba(0,170,255,0.15); }}
    .frames-grid {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 20px; }}
    .frame-card {{ background: rgba(0,0,0,0.4); border: 1px solid rgba(255,51,51,0.4);
                   border-radius: 6px; padding: 10px; flex: 1 1 calc(33% - 16px); text-align: center; }}
    .frame-card img {{ max-width: 100%; border-radius: 4px; }}
    .frame-label {{ color: #ff3333; font-size: 11px; margin-top: 6px; letter-spacing: 2px; }}
    .footer {{ margin-top: 60px; border-top: 1px solid rgba(255,255,255,0.05);
               padding-top: 24px; text-align: center; color: #4a5568; font-size: 12px; }}
    .highlight {{ color: {v_color}; font-weight: bold; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🔬 VIDEO DEEPFAKE FORENSIC INTELLIGENCE REPORT</h1>
    <div class="sub">CLASSIFICATION: RESTRICTED // FORENSIC INTELLIGENCE // SCAN {scan_id}</div>
    <div style="margin-top:14px">
      <span class="badge">SCAN ID: {scan_id}</span>
      <span class="badge">TIMESTAMP: {timestamp}</span>
      <span class="badge">ENGINES: 7 ACTIVE</span>
      <span class="badge">FRAMES ANALYZED: {metadata.get('sampled_frames','N/A')}</span>
    </div>
  </div>

  <div class="verdict-box">
    <h2>FINAL ASSESSMENT: {v_title}</h2>
    <div class="stats-grid">
      <div class="stat-box"><div class="label">COMPOSITE SCORE</div><div class="val" style="color:{v_color}">{composite}/100</div></div>
      <div class="stat-box"><div class="label">CONFIDENCE</div><div class="val">{confidence}%</div></div>
      <div class="stat-box"><div class="label">RISK LEVEL</div><div class="val" style="color:{v_color}">{risk}</div></div>
      <div class="stat-box"><div class="label">THREAT LEVEL</div><div class="val">{threat}</div></div>
    </div>
    <div class="stats-grid">
      <div class="stat-box"><div class="label">NEURAL ANOMALY INDEX</div><div class="val">{nai}%</div></div>
      <div class="stat-box"><div class="label">TEMPORAL INSTABILITY</div><div class="val">{tis}</div></div>
      <div class="stat-box"><div class="label">FLAGGED ENGINES</div><div class="val">{flagged}/7</div></div>
      <div class="stat-box"><div class="label">DETECTION THRESHOLD</div><div class="val">52/100</div></div>
    </div>
  </div>

  <div class="section-title">01 // Video Evidence Metadata</div>
  <div class="meta-grid">
    <div class="stat-box"><div class="label">FILENAME</div><div class="val" style="font-size:13px">{metadata.get('filename','N/A')}</div></div>
    <div class="stat-box"><div class="label">DURATION</div><div class="val">{metadata.get('duration','N/A')}</div></div>
    <div class="stat-box"><div class="label">RESOLUTION</div><div class="val">{metadata.get('resolution','N/A')}</div></div>
    <div class="stat-box"><div class="label">FPS</div><div class="val">{metadata.get('fps','N/A')}</div></div>
    <div class="stat-box"><div class="label">CODEC</div><div class="val">{metadata.get('codec','N/A')}</div></div>
    <div class="stat-box"><div class="label">BITRATE</div><div class="val">{metadata.get('bitrate_kbps','N/A')} kbps</div></div>
    <div class="stat-box"><div class="label">TOTAL FRAMES</div><div class="val">{metadata.get('total_frames','N/A')}</div></div>
    <div class="stat-box"><div class="label">AUDIO</div><div class="val">{'PRESENT' if metadata.get('has_audio') else 'ABSENT'}</div></div>
  </div>

  <div class="section-title">02 // Executive Summary</div>
  <p>{executive}</p>

  <div class="section-title">03 // Forensic Engine Telemetry</div>
  <table>
    <thead><tr><th>ENGINE</th><th>SCORE</th><th>STATUS</th><th>ANOMALY FRAMES</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <div class="section-title">04 // Temporal Inconsistency Analysis</div>
  <p>
    The temporal forensic pipeline analyzed <span class="highlight">{metadata.get('sampled_frames','N/A')} sampled frames</span>
    across a {metadata.get('duration','N/A')} duration video. Frame-to-frame landmark drift analysis
    revealed a temporal instability score of <span class="highlight">{tis}</span>.
    {f'Significant warping artifacts detected at {len(engine_results.get("temporal",{}).get("anomaly_frames",[]))} frame positions.' if is_fake else 'Motion vectors remain consistent with natural human movement.'}
  </p>
  <p>
    Identity persistence analysis tracked facial geometry embeddings across the full timeline.
    <span class="highlight">{engine_results.get("identity",{}).get("summary","N/A")}</span>
  </p>

  <div class="section-title">05 // GAN Fingerprint Analysis</div>
  <p>
    Spectral frequency analysis via Fast Fourier Transform identified
    {'periodic GAN upsampling artifacts in the high-frequency ring, characteristic of neural rendering pipelines' if engine_scores.get('gan',0) > 50 else 'no significant GAN frequency signatures — spectral distribution consistent with authentic camera capture'}.
    GAN engine score: <span class="highlight">{engine_scores.get('gan',0)}/100</span>.
    {engine_results.get('gan',{}).get('summary','N/A')}
  </p>

  <div class="section-title">06 // Suspicious Frame Evidence</div>
  {'<div class="frames-grid">' + frame_imgs + '</div>' if frame_imgs else '<p>No suspicious frames extracted or frame capture unavailable.</p>'}

  <div class="section-title">07 // Final Forensic Assessment</div>
  <p>
    Based on the convergent findings of all 7 independent forensic engines, this analysis concludes
    with <span class="highlight">{confidence}% confidence</span> that the submitted video evidence
    {'contains synthetic manipulation consistent with deepfake generation technology' if is_fake else 'is authentic and does not contain detectable synthetic manipulation'}.
    Risk classification: <span class="highlight">{risk}</span>.
    Threat level: <span class="highlight">{threat}</span>.
  </p>
  <p>
    This report was generated by the Video DeepFake Forensic Intelligence Platform.
    Analysis is based on heuristic multi-engine temporal forensic methods and should be
    interpreted in the context of a broader forensic investigation.
  </p>

  <div class="footer">
    VIDEO DEEPFAKE FORENSIC INTELLIGENCE PLATFORM &nbsp;|&nbsp; SCAN: {scan_id} &nbsp;|&nbsp; {timestamp}
  </div>
</body>
</html>"""
    return html
