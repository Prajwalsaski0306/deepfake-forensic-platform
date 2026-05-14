"""
app.py — Unified Multi-Modal AI Forensic Platform (Image, Video, & Audio)
"""
import os, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# Video Modules
import video_ingest as vi
import video_engines as ve
import video_scorer as vs
import video_report as vr

# Image Modules
import utils
import detector
import forensics
import visualizer
import scorer

# Audio Modules
import audio_engines as ae
import audio_scorer as asco
import audio_report as ar
import cross_modal_engine as cme
import pdf_generator
import app_styles as styles

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Forensic Terminal",
    page_icon="👁️‍🗨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ────────────────────────────────────────────────────────
for k in ["scan_active", "scan_done", "results", "media_type", "media_path", "metadata", "ref_img_bytes", "test_img_bytes", "audio_bytes"]:
    if k not in st.session_state:
        st.session_state[k] = None if k not in ["scan_active", "scan_done"] else False

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_b64(img_obj):
    if img_obj is None: return ""
    if isinstance(img_obj, np.ndarray):
        img_obj = Image.fromarray(img_obj)
    import base64
    from io import BytesIO
    buf = BytesIO()
    if img_obj.mode == "RGBA":
        img_obj = img_obj.convert("RGB")
    img_obj.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_image_html_report(scan_id, timestamp, engines, verdict, images, geo_table, face_cmp=None):
    risk_colors = {"LOW RISK": "#00ff88", "MODERATE RISK": "#ffcc00", "HIGH RISK": "#ff8800", "CRITICAL": "#ff3333"}
    r_color = risk_colors.get(verdict["risk_level"], "#aaa")
    v_title = "DEEPFAKE DETECTED" if verdict["is_fake"] else "AUTHENTIC"

    fc = face_cmp or {}
    sim = fc.get("similarity_score", "N/A")
    match_str = "MATCH" if fc.get("match", False) else "MISMATCH"

    if verdict["is_fake"]:
        summary = (f"Analysis indicates statistically significant anomalies in frequency distribution, "
                   f"texture entropy, and error level patterns consistent with synthetic image generation. "
                   f"Identity match result: {match_str} (Similarity: {sim}%).")
    else:
        summary = (f"Analysis indicates spatial, spectral, and geometric heuristics fall within natural "
                   f"biological and optical tolerances. No definitive evidence of synthetic manipulation found. "
                   f"Identity match result: {match_str} (Similarity: {sim}%).")

    table_rows = ""
    if geo_table:
        for row in geo_table:
            status_cls = "fail" if "FAIL" in str(row['Status']) else "pass"
            table_rows += f"<tr><td>{row['Metric']}</td><td>{row['Measured']}</td><td>{row['Normal Range']}</td><td>{row['Deviation']}</td><td class='{status_cls}'>{row['Status']}</td></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Image Forensic Report - {scan_id}</title>
    <style>
        body {{ font-family: 'Courier New', monospace; background-color: #060b14; color: #c9d1d9; padding: 40px; line-height:1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #00ff88; padding-bottom: 20px; margin-bottom: 30px; }}
        .header h1 {{ color: #00ff88; font-size: 28px; letter-spacing: 2px; text-transform: uppercase; }}
        .meta {{ text-align: center; color: #58a6ff; font-size: 14px; margin-bottom: 40px; letter-spacing: 1px; }}
        .verdict-box {{ background: rgba(0, 255, 136, 0.05); border: 1px solid {r_color}; border-left: 8px solid {r_color}; padding: 25px; border-radius: 8px; margin-bottom: 40px; }}
        .verdict-box h2 {{ margin-top: 0; color: {r_color}; font-size: 24px; letter-spacing: 2px; }}
        .section-title {{ color: #00aaff; font-size: 18px; border-bottom: 1px solid rgba(0, 170, 255, 0.3); padding-bottom: 8px; margin: 40px 0 20px; text-transform: uppercase; font-weight: bold; letter-spacing: 1px; }}
        .grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .card {{ background: rgba(10, 16, 28, 0.8); border: 1px solid rgba(0, 170, 255, 0.2); border-radius: 8px; padding: 20px; flex: 1 1 calc(33.333% - 20px); text-align: center; }}
        .card h3 {{ color: #e2e8f0; font-size: 15px; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }}
        .card p {{ font-size: 13px; color: #a0aec0; }}
        .card img {{ max-width: 100%; border-radius: 6px; margin-top: 15px; border: 1px solid rgba(0,170,255,0.3); box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        th, td {{ border: 1px solid rgba(0, 170, 255, 0.2); padding: 12px; text-align: left; }}
        th {{ background-color: rgba(0, 170, 255, 0.1); color: #00aaff; }}
        td {{ background-color: rgba(10, 16, 28, 0.5); }}
        .fail {{ color: #ff3333; font-weight: bold; }}
        .pass {{ color: #00ff88; }}
    </style>
</head>
<body>
    <div class="header"><h1>IMAGE DEEPFAKE FORENSIC REPORT</h1></div>
    <div class="meta">SCAN ID: {scan_id} &nbsp;|&nbsp; TIMESTAMP: {timestamp}</div>
    <div class="verdict-box">
        <h2>FINAL ASSESSMENT: {v_title}</h2>
        <p><strong>Confidence:</strong> {verdict['confidence']}% &nbsp;|&nbsp; <strong>Risk Level:</strong> {verdict['risk_level']} &nbsp;|&nbsp; <strong>System Score:</strong> {verdict['fake_score']}/100</p>
        <p><strong>Executive Summary:</strong> {summary}</p>
    </div>
    <div class="section-title">01 // Spatial Analysis Evidence</div>
    <div class="grid">
        <div class="card"><h3>Source + Bounding Box</h3><img src="data:image/png;base64,{images.get('bbox','')}" /></div>
        <div class="card"><h3>Geometric Mesh (468-PT)</h3><img src="data:image/png;base64,{images.get('mesh','')}" /></div>
        <div class="card"><h3>Error Level Heatmap</h3><img src="data:image/png;base64,{images.get('ela','')}" /></div>
    </div>
    <div class="section-title">02 // Heuristic Engine Telemetry</div>
    <div class="grid">
        <div class="card"><h3>FFT Spectrum Signature</h3><p><strong>Score: {engines.get('fft',{{}})['score']}/100</strong><br>{engines.get('fft',{{}})['status']}</p><img src="data:image/png;base64,{images.get('fft','')}" /></div>
        <div class="card"><h3>LBP Texture Entropy</h3><p><strong>Score: {engines.get('tex',{{}})['score']}/100</strong><br>{engines.get('tex',{{}})['status']}</p><img src="data:image/png;base64,{images.get('lbp','')}" /></div>
        <div class="card"><h3>Color Boundary Delta</h3><p><strong>Score: {engines.get('col',{{}})['score']}/100</strong><br>{engines.get('col',{{}})['status']}</p><img src="data:image/png;base64,{images.get('col','')}" /></div>
    </div>
    <div class="section-title">03 // Geometric Violations Map</div>
    <table><thead><tr><th>Metric</th><th>Measured</th><th>Normal Range</th><th>Deviation</th><th>Status</th></tr></thead><tbody>{table_rows}</tbody></table>
</body></html>"""
    return html

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Orbitron,sans-serif;color:#00ff88;font-size:1.2rem;
      letter-spacing:4px;text-align:center;border-bottom:1px solid rgba(0,255,136,.3);
      padding-bottom:16px;margin-bottom:20px;text-shadow:0 0 10px rgba(0,255,136,.6);'>
      VFI // MULTI-MODAL
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='color:#8bb4e7;font-size:.82rem;line-height:1.7;margin-bottom:20px;'>
    Unified AI Forensic Intelligence Platform.<br>
    Supports Image, Video, and Audio forensic modes automatically.
    </div>""", unsafe_allow_html=True)
    
    if os.path.exists("test_deepfake.mp4"):
        with open("test_deepfake.mp4", "rb") as file:
            btn = st.download_button(
                label="📥 DOWNLOAD SAMPLE DEEPFAKE (MP4)",
                data=file,
                file_name="test_deepfake.mp4",
                mime="video/mp4",
                use_container_width=True
            )

    st.markdown("<div style='color:#e2e8f0;font-family:Orbitron,sans-serif;font-size:.8rem;letter-spacing:2px;margin-bottom:12px;'>IMAGE ENGINES</div>", unsafe_allow_html=True)
    for icon, name in [("🔴","Error Level Analysis"), ("🌊","FFT Spectrum"), ("📐","Geometric Ratios"), ("🧬","Texture Entropy"), ("🟣","Color Boundary"), ("🆔","Face Comparison")]:
        st.markdown(f"<div style='color:#718096;font-size:.75rem;margin-bottom:6px;'>{icon} {name}</div>", unsafe_allow_html=True)
    
    st.markdown("<br><div style='color:#e2e8f0;font-family:Orbitron,sans-serif;font-size:.8rem;letter-spacing:2px;margin-bottom:12px;'>VIDEO ENGINES</div>", unsafe_allow_html=True)
    for icon, name in [("⏱","Temporal Consistency"), ("🆔","Identity Persistence"), ("📡","GAN Fingerprint"), ("😶","Micro-Expression"), ("👁","Blink & Eye Analysis"), ("🧬","Skin Stability"), ("🔊","Audio-Visual Sync")]:
        st.markdown(f"<div style='color:#718096;font-size:.75rem;margin-bottom:6px;'>{icon} {name}</div>", unsafe_allow_html=True)

    st.markdown("<br><div style='color:#e2e8f0;font-family:Orbitron,sans-serif;font-size:.8rem;letter-spacing:2px;margin-bottom:12px;'>AUDIO ENGINES</div>", unsafe_allow_html=True)
    for icon, name in [("〰","Voice Spectral Analysis"), ("🧠","MFCC Voice Analysis"), ("🎵","Prosody Analysis"), ("🔇","Noise Consistency"), ("🧬","Voice Biometric"), ("⚡","Deep Speech Artifact")]:
        st.markdown(f"<div style='color:#718096;font-size:.75rem;margin-bottom:6px;'>{icon} {name}</div>", unsafe_allow_html=True)

    st.markdown("<br><div style='color:#e2e8f0;font-family:Orbitron,sans-serif;font-size:.8rem;letter-spacing:2px;margin-bottom:12px;'>CROSS-MODAL ENGINES</div>", unsafe_allow_html=True)
    for icon, name in [("🔄","Audio-Visual Sync"), ("🗣","Phoneme-Lip Alignment"), ("🎭","Voice-Face Identity"), ("🎤","Speaker Consistency"), ("⏱","Temporal Audio Delay"), ("🧬","Cross-Modal Biometrics")]:
        st.markdown(f"<div style='color:#718096;font-size:.75rem;margin-bottom:6px;'>{icon} {name}</div>", unsafe_allow_html=True)

    if st.session_state.scan_done and st.session_state.results:
        st.markdown("---")
        if st.button("🔄 NEW SCAN"):
            for k in ["scan_active", "scan_done", "results", "media_type", "media_path", "metadata", "ref_img_bytes", "test_img_bytes", "audio_bytes"]:
                st.session_state[k] = None if k not in ["scan_active", "scan_done"] else False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CINEMATIC SCAN SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.scan_active and not st.session_state.scan_done:
    st.markdown(styles.GLOBAL_CSS + styles.SCAN_CSS + styles.AUDIO_SCAN_CSS, unsafe_allow_html=True)
    st.markdown('<div class="scan-wrap">', unsafe_allow_html=True)
    
    mtype = st.session_state.media_type
    if mtype == "video":
        hud_title = "⬡ VIDEO FORENSIC SCAN ACTIVE ⬡"
    elif mtype == "audio":
        hud_title = "⬡ AUDIO FORENSIC SCAN ACTIVE ⬡"
    else:
        hud_title = "⬡ IMAGE FORENSIC SCAN ACTIVE ⬡"
        
    st.markdown(f'<div class="scan-hud-title">{hud_title}</div>', unsafe_allow_html=True)
    
    sub_ph = st.empty()
    
    if mtype == "audio":
        st.markdown(styles.voice_scan_orb_html(), unsafe_allow_html=True)
        st.markdown(styles.audio_waveform_html(80), unsafe_allow_html=True)
    else:
        st.markdown(styles.FACE_SVG_HTML, unsafe_allow_html=True)
        st.markdown(styles.waveform_html(48), unsafe_allow_html=True)
        
    prog = st.progress(0)
    log_ph = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    logs = []
    def stage(s, e, dur, msg, sub):
        sub_ph.markdown(f'<div class="scan-hud-sub">[ {sub} ]</div>', unsafe_allow_html=True)
        logs.append(f"[✔] {msg}")
        log_ph.markdown(styles.log_box_html(logs), unsafe_allow_html=True)
        steps = max(1, int(dur * 8))
        for i in range(steps):
            prog.progress(int(s + (e - s) * i / steps))
            time.sleep(dur / steps)

    if mtype == "video":
        meta = st.session_state.metadata or {}
        stage(0,  8,  6,  "Initializing neural video engine...", "CORE INIT")
        stage(8,  16, 7,  f"Ingesting video — {meta.get('total_frames','?')} frames @ {meta.get('fps','?')} FPS", "VIDEO INGESTION")
        stage(16, 26, 9,  "Running Temporal Consistency Engine...", "ENGINE 1/7")
        stage(26, 36, 9,  "Identity Persistence Engine...", "ENGINE 2/7")
        stage(36, 48, 10, "GAN Fingerprint Engine...", "ENGINE 3/7")
        stage(48, 58, 8,  "Micro-Expression Engine...", "ENGINE 4/7")
        stage(58, 68, 8,  "Blink & Eye EAR Engine...", "ENGINE 5/7")
        stage(68, 78, 9,  "Skin Texture Stability Engine...", "ENGINE 6/7")
        stage(78, 83, 6,  "Audio-Visual Sync Engine...", "ENGINE 7/7")
        if meta.get('has_audio'):
            stage(83, 86, 5, "Extracting Audio for Cross-Modal Analysis...", "AUDIO EXTRACT")
            stage(86, 89, 5, "Running Phoneme-Lip Alignment...", "CROSS-MODAL 1/6")
            stage(89, 93, 5, "Running Voice-Face Identity Match...", "CROSS-MODAL 2/6")
            stage(93, 96, 5, "Running Temporal Dubbing Analysis...", "CROSS-MODAL 3/6")
        stage(96, 98, 4,  "Building forensic reconstruction timeline...", "RECONSTRUCTION")
        stage(98, 100,4,  "Compiling intelligence report...", "REPORT GEN")
    elif mtype == "audio":
        stage(0,  15, 6, "Initializing neural audio engine...", "CORE INIT")
        stage(15, 30, 8, "Extracting speech features...", "FEATURE EXTRACTION")
        stage(30, 45, 9, "Running Voice Spectral Analysis...", "ENGINE 1/6")
        stage(45, 60, 9, "Running MFCC Voice Analysis...", "ENGINE 2/6")
        stage(60, 70, 8, "Analyzing Prosody & Rhythm...", "ENGINE 3/6")
        stage(70, 80, 8, "Evaluating Noise Consistency...", "ENGINE 4/6")
        stage(80, 88, 8, "Verifying Voice Biometrics...", "ENGINE 5/6")
        stage(88, 96, 8, "Detecting Deep Speech Artifacts...", "ENGINE 6/6")
        stage(96, 100,4, "Compiling intelligence report...", "REPORT GEN")
    else:
        stage(0,  15, 6, "Initializing neural image engine...", "CORE INIT")
        stage(15, 30, 6, "Extracting geometric mesh (468-PT)...", "LANDMARK EXTRACTION")
        stage(30, 45, 6, "Running Error Level Analysis (ELA)...", "ENGINE 1/5")
        stage(45, 60, 6, "Running FFT Spectral Analysis...", "ENGINE 2/5")
        stage(60, 75, 6, "Analyzing Geometric Consistency...", "ENGINE 3/5")
        stage(75, 85, 6, "Evaluating LBP Texture Entropy...", "ENGINE 4/5")
        stage(85, 95, 6, "Detecting Color Boundary Anomalies...", "ENGINE 5/5")
        stage(95, 100,4, "Compiling intelligence report...", "REPORT GEN")

    prog.progress(100)
    time.sleep(0.6)
    st.markdown("""<style>.flash{position:fixed;top:0;left:0;width:100vw;height:100vh;
      background:#fff;z-index:9999999;animation:fo 1s forwards;}
      @keyframes fo{0%{opacity:1;}100%{opacity:0;pointer-events:none;}}</style>
      <div class='flash'></div>""", unsafe_allow_html=True)
    time.sleep(0.5)
    st.session_state.scan_done = True
    st.session_state.scan_active = False
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD (PRE-SCAN UPLOAD)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.scan_active and not st.session_state.scan_done:
    st.markdown(styles.GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:30px 0 20px;'>
      <div style='font-family:Orbitron,sans-serif;font-size:2.6rem;font-weight:900;
        color:#00ff88;letter-spacing:5px;text-shadow:0 0 30px rgba(0,255,136,.5);'>
        UNIFIED FORENSIC INTELLIGENCE
      </div>
      <div style='color:#00aaff;font-size:.9rem;letter-spacing:5px;margin-top:8px;opacity:.8;'>
        [ MULTI-MODAL DEEPFAKE ANALYSIS COMMAND CENTER ]
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">00 // EVIDENCE INGESTION</div>', unsafe_allow_html=True)
    st.markdown("""<div style='color:#8bb4e7;font-size:.83rem;margin-bottom:16px;'>
    System auto-detects input modality. Supported Formats: <b>JPG, PNG, WEBP, MP4, MOV, AVI, MKV, MP3, WAV, FLAC</b>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image, Video, or Audio Evidence", type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv", "mp3", "wav", "flac"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""<div class='g-card' style='text-align:center;padding:60px;'>
          <div style='font-size:3rem;margin-bottom:14px;opacity:.5;'>🛡️</div>
          <div style='font-family:Orbitron,sans-serif;color:#8bb4e7;font-size:1.2rem;letter-spacing:3px;'>AWAITING EVIDENCE</div>
          <div style='color:#4a6070;font-size:.83rem;margin-top:10px;'>Drop a file to initiate smart routing</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    ext = uploaded.name.split('.')[-1].lower()
    if ext in ["mp4", "mov", "avi", "mkv"]:
        media_type = "video"
    elif ext in ["mp3", "wav", "flac"]:
        media_type = "audio"
    else:
        media_type = "image"
        
    st.session_state.media_type = media_type

    if media_type == "video":
        file_bytes = uploaded.getvalue()
        val = vi.validate_video(file_bytes, uploaded.name)
        if not val["valid"]:
            st.error(f"[ VIDEO VALIDATION FAILED ] {val['error']}")
            st.stop()

        tmp_path = val["path"]
        meta = vi.extract_metadata(tmp_path, uploaded.name)
        st.session_state.media_path = tmp_path
        st.session_state.metadata = meta

        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">01 // VIDEO METADATA (SMART ROUTING: VIDEO MODE)</div>', unsafe_allow_html=True)
        cols = st.columns(8)
        chip_data = [
            ("FILENAME", meta["filename"][:10]+"…"), ("DURATION", meta["duration"]),
            ("FPS", f"{meta['fps']}"), ("RESOLUTION", meta["resolution"]),
            ("CODEC", meta["codec"]), ("BITRATE", f"{meta['bitrate_kbps']}"),
            ("FRAMES", str(meta["total_frames"])), ("AUDIO", "YES" if meta["has_audio"] else "NO")
        ]
        for col, (label, val_) in zip(cols, chip_data):
            col.markdown(styles.meta_chip(label, val_), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif media_type == "audio":
        file_bytes = uploaded.getvalue()
        st.session_state.audio_bytes = file_bytes
        meta = ae.extract_audio_metadata(file_bytes, uploaded.name)
        st.session_state.metadata = meta

        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">01 // AUDIO METADATA (SMART ROUTING: AUDIO MODE)</div>', unsafe_allow_html=True)
        cols = st.columns(6)
        for col, label, val_ in zip(cols, ["FILENAME", "FORMAT", "DURATION", "SAMPLE RATE", "BITRATE", "SIZE"], 
                                    [meta["filename"][:15]+"...", meta["format"], meta["duration"], meta["sample_rate"], meta["bitrate"], meta["file_size"]]):
            col.markdown(styles.meta_chip(label, val_), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif media_type == "image":
        file_bytes = uploaded.getvalue()
        val = utils.validate_upload(file_bytes, uploaded.name)
        if not val["valid"]:
            st.error(f"[ IMAGE VALIDATION FAILED ] {val['error']}")
            st.stop()
        
        st.session_state.test_img_bytes = file_bytes
        meta = utils.get_image_metadata(file_bytes, uploaded.name)
        st.session_state.metadata = meta

        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">01 // IMAGE METADATA (SMART ROUTING: IMAGE MODE)</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for col, label, val_ in zip(cols, ["FILENAME", "RESOLUTION", "SIZE", "CHANNELS", "FORMAT"], 
                                    [meta["filename"][:15]+"...", meta["resolution"], meta["file_size"], meta["color_mode"], meta["format"]]):
            col.markdown(styles.meta_chip(label, val_), unsafe_allow_html=True)
        
        st.markdown('<div class="sec-title" style="margin-top:20px">OPTIONAL // REFERENCE IDENTITY COMPARISON</div>', unsafe_allow_html=True)
        ref_upload = st.file_uploader("Upload Reference Image (Optional)", type=["jpg", "jpeg", "png", "webp"])
        if ref_upload:
            st.session_state.ref_img_bytes = ref_upload.getvalue()
        else:
            st.session_state.ref_img_bytes = None
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("▶ INITIATE FORENSIC SCAN"):
            with st.spinner("Processing neural engines..."):
                if media_type == "video":
                    frames = vi.sample_frames(st.session_state.media_path, max_frames=60)
                    has_audio = st.session_state.metadata.get("has_audio", False)
                    e_temporal = ve.engine_temporal_consistency(frames)
                    e_identity = ve.engine_identity_persistence(frames)
                    e_gan      = ve.engine_gan_fingerprint(frames)
                    e_micro    = ve.engine_micro_expression(frames)
                    e_blink    = ve.engine_blink_eye(frames)
                    e_texture  = ve.engine_skin_texture(frames)
                    e_avsync   = ve.engine_audio_visual_sync(frames, has_audio)

                    engine_results = {
                        "temporal": e_temporal, "identity": e_identity, "gan": e_gan, 
                        "micro_expr": e_micro, "blink": e_blink, "texture": e_texture, "av_sync": e_avsync
                    }
                    verdict = vs.compute_video_verdict(engine_results)
                    
                    cm_results = None
                    if has_audio:
                        audio_bytes = vi.extract_audio_bytes(st.session_state.media_path)
                        if audio_bytes:
                            cm_results = cme.run_all_cross_modal(frames, audio_bytes)
                            audio_results = {
                                "spectral": ae.engine_voice_spectral(audio_bytes),
                                "mfcc": ae.engine_mfcc_analysis(audio_bytes),
                                "prosody": ae.engine_prosody_analysis(audio_bytes),
                                "noise": ae.engine_noise_consistency(audio_bytes),
                                "biometric": ae.engine_voice_biometric(audio_bytes),
                                "artifact": ae.engine_deep_speech_artifact(audio_bytes)
                            }
                            audio_verdict = asco.compute_audio_verdict(audio_results)
                            cm_verdict = cme.compute_cross_modal_verdict(cm_results, verdict, audio_verdict)
                            
                            verdict["verdict_title"] = cm_verdict["verdict_title"]
                            verdict["is_fake"] = cm_verdict["is_fake"]
                            verdict["risk_level"] = cm_verdict["risk_level"]
                            verdict["cm_score"] = cm_verdict["cm_score"]

                    sus_indices = sorted(set(e_temporal.get("anomaly_frames",[])[:3] + e_identity.get("anomaly_frames",[])[:3] + e_gan.get("anomaly_frames",[])[:3]))[:6]
                    sus_frames = [frames[i] for i in sus_indices if i < len(frames)]
                    html_report = vr.generate_video_html_report(st.session_state.metadata, engine_results, verdict, sus_frames)

                    st.session_state.results = {"type": "video", "engine_results": engine_results, "cm_results": cm_results, "verdict": verdict, "sus_frames": sus_frames, "html_report": html_report}
                
                elif media_type == "audio":
                    fbytes = st.session_state.audio_bytes
                    e_spec = ae.engine_voice_spectral(fbytes)
                    e_mfcc = ae.engine_mfcc_analysis(fbytes)
                    e_pros = ae.engine_prosody_analysis(fbytes)
                    e_nois = ae.engine_noise_consistency(fbytes)
                    e_biom = ae.engine_voice_biometric(fbytes)
                    e_arti = ae.engine_deep_speech_artifact(fbytes)
                    
                    engine_results = {
                        "spectral": e_spec, "mfcc": e_mfcc, "prosody": e_pros,
                        "noise": e_nois, "biometric": e_biom, "artifact": e_arti
                    }
                    verdict = asco.compute_audio_verdict(engine_results)
                    html_report = ar.generate_audio_html_report(st.session_state.metadata, engine_results, verdict)
                    
                    st.session_state.results = {"type": "audio", "engine_results": engine_results, "verdict": verdict, "html_report": html_report}
                
                elif media_type == "image":
                    img_pil = utils.resize_image(utils.load_image(st.session_state.test_img_bytes))
                    img_arr = utils.pil_to_numpy(img_pil)
                    bbox = detector.detect_face(img_arr)
                    if bbox is None:
                        st.error("CRITICAL: No face detected in test image.")
                        st.stop()
                    
                    landmarks = detector.extract_landmarks(img_arr)
                    face_roi = detector.get_face_roi(img_arr, bbox)
                    
                    ela_result = forensics.run_ela(img_pil, face_roi)
                    fft_result = forensics.run_fft(face_roi)
                    geo_result = forensics.run_geometry(landmarks, img_arr.shape)
                    tex_result = forensics.run_texture(face_roi)
                    col_result = forensics.run_color(img_arr, bbox)

                    face_cmp = {"similarity_score": 0, "status": "N/A", "match": False}
                    if st.session_state.ref_img_bytes:
                        ref_pil = utils.resize_image(utils.load_image(st.session_state.ref_img_bytes))
                        ref_arr = utils.pil_to_numpy(ref_pil)
                        ref_landmarks = detector.extract_landmarks(ref_arr)
                        if ref_landmarks:
                            face_cmp = forensics.run_face_comparison(ref_landmarks, ref_arr.shape, landmarks, img_arr.shape)

                    engine_scores = {"ela": ela_result["score"], "fft": fft_result["score"], "geometry": geo_result["score"], "texture": tex_result["score"], "color": col_result["score"]}
                    verdict = scorer.compute_verdict(engine_scores)
                    verdict = scorer.integrate_face_comparison(verdict, face_cmp)
                    verdict["scan_id"] = utils.generate_scan_id()
                    verdict["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    img_b64 = {
                        "bbox": to_b64(detector.draw_face_bbox(img_arr, bbox, verdict["is_fake"])),
                        "mesh": to_b64(detector.draw_landmark_mesh(img_arr, landmarks) if landmarks else img_arr),
                        "ela": to_b64(visualizer.render_ela_heatmap(ela_result["ela_image"], ela_result["mean_ela"]) if ela_result["ela_image"] else None),
                        "fft": to_b64(visualizer.render_fft_spectrum(fft_result.get("log_magnitude"), fft_result["periodicity_score"])),
                        "lbp": to_b64(visualizer.render_lbp_visual(tex_result.get("lbp_image"), tex_result["entropy"])),
                        "col": to_b64(visualizer.render_color_chart(col_result["face_stats"], col_result["border_stats"], col_result["delta_score"]))
                    }

                    engines = {"ela": ela_result, "fft": fft_result, "geo": geo_result, "tex": tex_result, "col": col_result}
                    html_report = generate_image_html_report(verdict["scan_id"], verdict["timestamp"], engines, verdict, img_b64, geo_result.get("table", []), face_cmp)

                    st.session_state.results = {
                        "type": "image", "img_pil": img_pil, "img_arr": img_arr, "bbox": bbox, "landmarks": landmarks,
                        "engines": engines, "verdict": verdict, "face_cmp": face_cmp, "html_report": html_report, "img_b64": img_b64
                    }
            
            st.session_state.scan_active = True
            st.session_state.scan_done = False
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD RENDERING (POST-SCAN)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.scan_done and st.session_state.results:
    st.markdown(styles.GLOBAL_CSS, unsafe_allow_html=True)
    res = st.session_state.results
    mtype = res["type"]
    verdict = res["verdict"]
    is_fake = verdict["is_fake"]

    v_cls = "v-fake" if is_fake else "v-real"
    v_title_cls = "v-title-fake" if is_fake else "v-title-real"
    
    if verdict.get("verdict_title"):
        v_label = "⚠ " + verdict["verdict_title"] if is_fake else "✔ " + verdict["verdict_title"]
    else:
        if mtype == "video":
            v_label = "⚠ DEEPFAKE VIDEO DETECTED" if is_fake else "✔ AUTHENTIC VIDEO"
        elif mtype == "audio":
            v_label = "⚠ AI GENERATED VOICE DETECTED" if is_fake else "✔ AUTHENTIC HUMAN VOICE"
        else:
            v_label = "⚠ DEEPFAKE IMAGE DETECTED" if is_fake else "✔ AUTHENTIC IMAGE"
        
    risk_colors = {"CRITICAL":"#ff3333","HIGH RISK":"#ff6600","MODERATE RISK":"#ffcc00","LOW RISK":"#88ff00","MINIMAL RISK":"#00ff88"}
    rc = risk_colors.get(verdict.get("risk_level","LOW RISK"), "#aaa")

    # ── COMMON VERDICT PANEL ──
    st.markdown(f'<div class="g-card {v_cls}" style="text-align:center;padding:40px;">', unsafe_allow_html=True)
    st.markdown(f"<div style='color:#8bb4e7;font-size:.8rem;letter-spacing:3px;margin-bottom:14px;'>SCAN_ID: {verdict['scan_id']} &nbsp;|&nbsp; {verdict['timestamp']}</div><div class='{v_title_cls}'>{v_label}</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="mbox"><span class="mlabel">CONFIDENCE</span><span class="mval">{verdict["confidence"]}%</span></div>', unsafe_allow_html=True)
    
    if mtype in ["video", "audio"]:
        c2.markdown(f'<div class="mbox"><span class="mlabel">COMPOSITE SCORE</span><span class="mval" style="color:{rc}">{verdict["composite_score"]}/100</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="mbox"><span class="mlabel">RISK LEVEL</span><span class="mval" style="color:{rc}">{verdict["risk_level"]}</span></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="mbox"><span class="mlabel">THREAT LEVEL</span><span class="mval" style="color:{rc}">{verdict["threat_level"]}</span></div>', unsafe_allow_html=True)
    else:
        c2.markdown(f'<div class="mbox"><span class="mlabel">SYSTEM SCORE</span><span class="mval" style="color:{rc}">{verdict["fake_score"]}/100</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="mbox"><span class="mlabel">RISK LEVEL</span><span class="mval" style="color:{rc}">{verdict["risk_level"]}</span></div>', unsafe_allow_html=True)
        fc = res.get("face_cmp",{})
        c4.markdown(f'<div class="mbox"><span class="mlabel">IDENTITY MATCH</span><span class="mval">{"YES" if fc.get("match") else "NO"} ({fc.get("similarity_score",0)}%)</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── VIDEO DASHBOARD ──
    if mtype == "video":
        er = res["engine_results"]
        st.markdown('<div class="g-card"><div class="sec-title">02 // VIDEO TIMELINE INTELLIGENCE</div>', unsafe_allow_html=True)
        timelines = [er[k].get("timeline",[]) for k in ["temporal","identity","gan","micro_expr","blink","texture","av_sync"] if er[k].get("timeline",[])]
        if timelines:
            min_len = min(len(t) for t in timelines)
            avg_tl = [float(np.mean([t[i] for t in timelines if i < len(t)])) for i in range(min_len)]
            fig_tl = go.Figure(go.Scatter(x=list(range(min_len)), y=avg_tl, mode="lines", line=dict(color="#00aaff", width=2), fill="tozeroy", fillcolor="rgba(0,170,255,0.08)"))
            fig_tl.add_hline(y=52, line=dict(color="#ff3333", dash="dash", width=1.5))
            fig_tl.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,5,12,0.6)", font=dict(color="#8bb4e7"), margin=dict(l=40,r=20,t=20,b=40), height=260)
            st.plotly_chart(fig_tl, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="sec-title">03 // TEMPORAL ENGINE TELEMETRY</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, (name, key, desc) in enumerate([("⏱ TEMPORAL CONSISTENCY","temporal","Frame drift"), ("🆔 IDENTITY PERSISTENCE","identity","Embeddings"), ("📡 GAN FINGERPRINT","gan","Spectral"), ("😶 MICRO-EXPRESSION","micro_expr","Freeze"), ("👁 BLINK & EYE EAR","blink","Blink rate"), ("🧬 SKIN TEXTURE","texture","Entropy"), ("🔊 AUDIO-VISUAL SYNC","av_sync","Lip-sync")]):
            r = er.get(key, {})
            (c1 if i%2==0 else c2).markdown(styles.engine_card_html(name, r.get("score",0), r.get("status",""), r.get("summary","")), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if res.get("cm_results"):
            cmr = res["cm_results"]
            st.markdown('<div class="g-card"><div class="sec-title">04 // CROSS-MODAL AUDIO-VISUAL TELEMETRY</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            for i, (name, key, desc) in enumerate([("🔄 AUDIO-VISUAL SYNC","av_sync","Lip-sync correlation"), ("🗣 PHONEME-LIP ALIGNMENT","phoneme","Syllable matching"), ("🎭 VOICE-FACE IDENTITY","identity","Embeddings match"), ("🎤 SPEAKER CONSISTENCY","speaker","Timbre drift"), ("⏱ TEMPORAL AUDIO DELAY","delay","Dubbing lag"), ("🧬 CROSS-MODAL BIOMETRICS","biometric","Unified anomaly")]):
                r = cmr.get(key, {})
                (cc1 if i%2==0 else cc2).markdown(styles.engine_card_html(name, r.get("score",0), r.get("status",""), r.get("summary","")), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if res["sus_frames"]:
            st.markdown('<div class="g-card"><div class="sec-title">05 // SUSPICIOUS FRAMES</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(res["sus_frames"]), 6))
            for i, (c, fr) in enumerate(zip(cols, res["sus_frames"])):
                c.image(fr, use_container_width=True)
                c.markdown('<div style="color:#ff3333;font-size:.7rem;text-align:center;">FLAGGED</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── AUDIO DASHBOARD ──
    elif mtype == "audio":
        er = res["engine_results"]
        st.markdown('<div class="g-card"><div class="sec-title">02 // AUDIO FORENSIC VISUALIZERS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        # Spectrogram representation (mock using spectral data)
        spec = er.get("spectral", {}).get("spectral_data", [])
        if spec:
            fig_sp = go.Figure(data=go.Heatmap(z=[spec], colorscale="Viridis", showscale=False))
            fig_sp.update_layout(title="Voice Spectral Distribution Map", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,5,12,0.6)", font=dict(color="#00aaff"), margin=dict(l=20,r=20,t=40,b=20), height=180)
            c1.plotly_chart(fig_sp, use_container_width=True)
            
        mfcc = er.get("mfcc", {}).get("mfcc_heatmap", [])
        if mfcc:
            fig_mf = go.Figure(data=go.Heatmap(z=mfcc, colorscale="Magma", showscale=False))
            fig_mf.update_layout(title="MFCC Feature Fingerprint", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,5,12,0.6)", font=dict(color="#00aaff"), margin=dict(l=20,r=20,t=40,b=20), height=180)
            c2.plotly_chart(fig_mf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="sec-title">03 // AUDIO ENGINE TELEMETRY</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        idx = 0
        for name, key, desc in [("〰 VOICE SPECTRAL ANALYSIS", "spectral", "Frequency anomalies"), 
                                ("🧠 MFCC VOICE ANALYSIS", "mfcc", "Cloning signatures"), 
                                ("🎵 PROSODY ANALYSIS", "prosody", "Speech rhythm & emotion"), 
                                ("🔇 NOISE CONSISTENCY", "noise", "Stitching artifacts"), 
                                ("🧬 VOICE BIOMETRIC", "biometric", "Identity consistency"), 
                                ("⚡ DEEP SPEECH ARTIFACT", "artifact", "Vocoder fingerprints")]:
            r = er.get(key, {})
            (c1 if idx%2==0 else c2).markdown(styles.engine_card_html(name, r.get("score",0), r.get("status",""), r.get("summary","")), unsafe_allow_html=True)
            idx += 1
        st.markdown('</div>', unsafe_allow_html=True)

    # ── IMAGE DASHBOARD ──
    else:
        eng = res["engines"]
        st.markdown('<div class="g-card"><div class="sec-title">02 // SPATIAL ANALYSIS VISUALS</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        b64 = res["img_b64"]
        c1.markdown(f'<div style="text-align:center"><img src="data:image/png;base64,{b64["bbox"]}" style="width:100%;border-radius:8px;border:1px solid rgba(0,170,255,0.3)"/><div style="color:#00aaff;font-size:.8rem;margin-top:8px;font-weight:bold;">SOURCE + BBOX</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div style="text-align:center"><img src="data:image/png;base64,{b64["mesh"]}" style="width:100%;border-radius:8px;border:1px solid rgba(0,170,255,0.3)"/><div style="color:#00aaff;font-size:.8rem;margin-top:8px;font-weight:bold;">468-PT MESH</div></div>', unsafe_allow_html=True)
        if b64["ela"]:
            c3.markdown(f'<div style="text-align:center"><img src="data:image/png;base64,{b64["ela"]}" style="width:100%;border-radius:8px;border:1px solid rgba(0,170,255,0.3)"/><div style="color:#00aaff;font-size:.8rem;margin-top:8px;font-weight:bold;">ELA HEATMAP</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="sec-title">03 // IMAGE FORENSIC ENGINES</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        idx = 0
        for title, key, desc in [("🔴 ERROR LEVEL ANALYSIS", "ela", "JPEG compression artifacts"), ("🌊 FFT SPECTRUM", "fft", "GAN frequency periodicity"), ("📐 GEOMETRY RATIOS", "geo", "Biological constraints"), ("🧬 LBP ENTROPY", "tex", "Micro-texture synthetic smoothing"), ("🟣 COLOR DELTA", "col", "Face/background blend boundary")]:
            r = eng.get(key, {})
            (cc1 if idx%2==0 else cc2).markdown(styles.engine_card_html(title, r.get("score",0), r.get("status",""), desc), unsafe_allow_html=True)
            idx += 1
        st.markdown('</div>', unsafe_allow_html=True)

    # ── REPORT EXPORT ──
    st.markdown('<div class="g-card"><div class="sec-title">EXPORT PROTOCOL</div>', unsafe_allow_html=True)
    
    c_btn1, c_btn2 = st.columns(2)
    
    c_btn1.download_button(
        label="⬇ DOWNLOAD FORENSIC REPORT (HTML)",
        data=res["html_report"].encode("utf-8"),
        file_name=f"FORENSIC_REPORT_{verdict['scan_id']}.html",
        mime="text/html",
        use_container_width=True,
    )
    
    # Generate PDF on the fly
    pdf_data = pdf_generator.generate_pdf_from_html(res["html_report"])
    if pdf_data:
        c_btn2.download_button(
            label="⬇ DOWNLOAD FORENSIC REPORT (PDF)",
            data=pdf_data,
            file_name=f"FORENSIC_REPORT_{verdict['scan_id']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        c_btn2.button("⚠️ PDF GENERATION FAILED", disabled=True, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
