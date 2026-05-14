# 🔬 DeepFake Forensics Lab

A professional-grade deepfake detection forensics tool built with Python and Streamlit.
Uses **5 real image forensics techniques** — no GPU, no model downloads, runs fully locally.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

---

## ⚙️ Detection Engines

| # | Engine | Technique | Signal |
|---|--------|-----------|--------|
| 1 | **ELA Analysis** | JPEG re-compression artefacts | High mean ELA → manipulation |
| 2 | **FFT Spectrum** | Frequency domain GAN fingerprint | Periodic variance → GAN |
| 3 | **Landmark Geometry** | 468-point facial ratio consistency | Ratio violations → anomaly |
| 4 | **Texture Entropy** | LBP skin micro-texture analysis | Low entropy → synthetic skin |
| 5 | **Color Boundary** | Face/background color delta | High delta → paste boundary |

---

## 📁 File Structure

```
deepfake detection/
├── app.py            # Streamlit UI (layout only)
├── detector.py       # OpenCV + MediaPipe face detection
├── forensics.py      # 5 forensic analysis engines
├── visualizer.py     # Heatmaps, FFT plots, color charts
├── scorer.py         # Weighted verdict + report generator
├── utils.py          # Image helpers and validation
├── requirements.txt  # Pinned dependencies
└── README.md         # This file
```

---

## 🛡️ Verdict Logic

```
FAKE_SCORE = ELA(25%) + FFT(20%) + Geometry(20%) + Texture(20%) + Color(15%)

> 50  → DEEPFAKE DETECTED 🚨
≤ 50  → AUTHENTIC IMAGE  ✅

Risk:  0–30 LOW | 31–50 MODERATE | 51–75 HIGH | 76–100 CRITICAL
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.
Results are based on forensic heuristics and should **not** be used as sole evidence of image manipulation.
