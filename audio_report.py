"""audio_report.py - Audio Forensic HTML Report Generator"""

def generate_audio_html_report(meta, engines, verdict):
    scan_id = verdict.get("scan_id","AUD-UNKNOWN")
    timestamp = verdict.get("timestamp","N/A")
    is_fake = verdict.get("is_fake", False)
    confidence = verdict.get("confidence", 0)
    risk_level = verdict.get("risk_level","UNKNOWN")
    composite = verdict.get("composite_score", 0)
    threat = verdict.get("threat_level","UNKNOWN")

    risk_colors = {"CRITICAL":"#ff3333","HIGH RISK":"#ff6600","MODERATE RISK":"#ffcc00","LOW RISK":"#88ff00","MINIMAL RISK":"#00ff88"}
    rc = risk_colors.get(risk_level,"#aaa")
    v_title = "DEEPFAKE AUDIO DETECTED" if is_fake else "AUTHENTIC AUDIO"

    if is_fake:
        summary = (f"Multi-engine forensic analysis detected significant synthetic speech artifacts. "
                   f"MFCC coefficient distributions, prosodic rhythm patterns, and spectral frequency "
                   f"profiles are inconsistent with natural human voice production. Voice cloning or "
                   f"neural TTS synthesis is highly probable.")
    else:
        summary = (f"Multi-engine forensic analysis indicates audio characteristics consistent with "
                   f"authentic human speech production. Spectral profiles, prosodic rhythm, MFCC "
                   f"distributions, and noise floor patterns fall within natural biological tolerances.")

    def eng_rows(eng_dict):
        rows = ""
        labels = {"spectral":"Voice Spectral Analysis","mfcc":"MFCC Voice Analysis","prosody":"Prosody Analysis",
                  "noise":"Noise Consistency","biometric":"Voice Biometric","artifact":"Deep Speech Artifact"}
        for key, label in labels.items():
            r = eng_dict.get(key, {})
            sc = r.get("score",0); st = r.get("status","N/A"); sm = r.get("summary","")
            color = "#ff3333" if sc > 60 else "#ffcc00" if sc > 35 else "#00ff88"
            rows += f"<tr><td>{label}</td><td style='color:{color};font-weight:bold'>{sc}/100</td><td>{st}</td><td style='font-size:12px;color:#718096'>{sm}</td></tr>"
        return rows

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Audio Forensic Report - {scan_id}</title>
<style>
body{{font-family:'Courier New',monospace;background:#060b14;color:#c9d1d9;padding:40px;line-height:1.6;}}
.header{{text-align:center;border-bottom:2px solid #00aaff;padding-bottom:20px;margin-bottom:30px;}}
.header h1{{color:#00aaff;font-size:26px;letter-spacing:3px;}}
.meta{{text-align:center;color:#58a6ff;font-size:13px;margin-bottom:35px;letter-spacing:1px;}}
.verdict-box{{background:rgba(0,170,255,0.05);border:1px solid {rc};border-left:8px solid {rc};padding:25px;border-radius:8px;margin-bottom:35px;}}
.verdict-box h2{{margin-top:0;color:{rc};font-size:22px;letter-spacing:2px;}}
.sec{{color:#00aaff;font-size:16px;border-bottom:1px solid rgba(0,170,255,0.3);padding-bottom:8px;margin:35px 0 18px;text-transform:uppercase;font-weight:bold;letter-spacing:1px;}}
.meta-grid{{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:20px;}}
.meta-card{{background:rgba(10,16,28,0.8);border:1px solid rgba(0,170,255,0.2);border-radius:8px;padding:14px 20px;flex:1 1 140px;text-align:center;}}
.meta-card .lbl{{color:#4a6070;font-size:11px;letter-spacing:2px;display:block;margin-bottom:4px;}}
.meta-card .val{{color:#e2e8f0;font-size:15px;font-weight:bold;}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px;}}
th,td{{border:1px solid rgba(0,170,255,0.2);padding:11px;text-align:left;}}
th{{background:rgba(0,170,255,0.1);color:#00aaff;}}
td{{background:rgba(10,16,28,0.5);}}
.verdict-label{{font-size:28px;font-weight:900;color:{rc};letter-spacing:4px;margin-bottom:12px;}}
</style>
</head>
<body>
<div class="header"><h1>🎧 AUDIO DEEPFAKE FORENSIC REPORT</h1></div>
<div class="meta">SCAN ID: {scan_id} &nbsp;|&nbsp; TIMESTAMP: {timestamp} &nbsp;|&nbsp; CLASSIFICATION: MULTI-ENGINE VOICE FORENSICS</div>
<div class="verdict-box">
  <div class="verdict-label">{v_title}</div>
  <p><strong>Confidence:</strong> {confidence}% &nbsp;|&nbsp; <strong>Composite Score:</strong> {composite}/100 &nbsp;|&nbsp; <strong>Risk Level:</strong> {risk_level} &nbsp;|&nbsp; <strong>Threat Level:</strong> {threat}</p>
  <p><strong>Executive Summary:</strong> {summary}</p>
</div>
<div class="sec">01 // AUDIO METADATA</div>
<div class="meta-grid">
  <div class="meta-card"><span class="lbl">FILENAME</span><span class="val">{meta.get('filename','N/A')[:18]}</span></div>
  <div class="meta-card"><span class="lbl">FORMAT</span><span class="val">{meta.get('format','N/A')}</span></div>
  <div class="meta-card"><span class="lbl">DURATION</span><span class="val">{meta.get('duration','N/A')}</span></div>
  <div class="meta-card"><span class="lbl">SAMPLE RATE</span><span class="val">{meta.get('sample_rate','N/A')}</span></div>
  <div class="meta-card"><span class="lbl">BITRATE</span><span class="val">{meta.get('bitrate','N/A')}</span></div>
  <div class="meta-card"><span class="lbl">FILE SIZE</span><span class="val">{meta.get('file_size','N/A')}</span></div>
</div>
<div class="sec">02 // ENGINE TELEMETRY</div>
<table>
  <thead><tr><th>ENGINE</th><th>ANOMALY SCORE</th><th>STATUS</th><th>SUMMARY</th></tr></thead>
  <tbody>{eng_rows(engines)}</tbody>
</table>
<div class="sec">03 // THREAT CLASSIFICATION</div>
<table>
  <thead><tr><th>PARAMETER</th><th>VALUE</th></tr></thead>
  <tbody>
    <tr><td>Synthetic Speech Probability</td><td style="color:{rc};font-weight:bold">{composite}%</td></tr>
    <tr><td>Voice Authenticity Score</td><td>{100-composite}/100</td></tr>
    <tr><td>Risk Classification</td><td style="color:{rc}">{risk_level}</td></tr>
    <tr><td>Threat Level</td><td style="color:{rc}">{threat}</td></tr>
    <tr><td>Final Determination</td><td style="color:{rc};font-weight:bold">{v_title}</td></tr>
  </tbody>
</table>
</body></html>"""
    return html
