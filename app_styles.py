"""app_styles.py — All CSS and HTML strings for the video forensic platform."""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Courier+Prime:wght@400;700&display=swap');

html,body,.stApp{background:#020912!important;color:#e0e6ed!important;font-family:'Courier Prime',monospace!important;}
#MainMenu,footer{visibility:hidden;}
header{background:transparent!important;}
.block-container{padding-top:1.5rem!important;}

/* Cyber grid background */
.stApp::before{content:'';position:fixed;top:0;left:0;width:100vw;height:100vh;
  background-image:linear-gradient(rgba(0,170,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,170,255,.04) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0;}

/* Glassmorphism card */
.g-card{background:rgba(8,15,30,.7);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border:1px solid rgba(0,255,136,.18);border-radius:16px;padding:24px;margin-bottom:20px;
  box-shadow:0 8px 40px rgba(0,0,0,.8),inset 0 0 20px rgba(0,255,136,.04);}

/* Section titles */
.sec-title{font-family:'Orbitron',sans-serif;font-size:1.1rem;color:#00aaff;letter-spacing:3px;
  text-transform:uppercase;border-bottom:1px solid rgba(0,170,255,.25);padding-bottom:8px;margin-bottom:20px;}

/* Verdict panels */
.v-fake{border:2px solid #ff3333!important;box-shadow:0 0 60px rgba(255,51,51,.4),inset 0 0 40px rgba(255,51,51,.1)!important;}
.v-real{border:2px solid #00ff88!important;box-shadow:0 0 60px rgba(0,255,136,.3),inset 0 0 40px rgba(0,255,136,.08)!important;}
.v-title-fake{font-family:'Orbitron',sans-serif;font-size:3rem;font-weight:900;color:#ff3333;
  text-shadow:0 0 30px #ff3333;letter-spacing:6px;animation:glitch 2.5s infinite;}
.v-title-real{font-family:'Orbitron',sans-serif;font-size:3rem;font-weight:900;color:#00ff88;
  text-shadow:0 0 30px #00ff88;letter-spacing:6px;}

/* Metric box */
.mbox{background:rgba(0,0,0,.5);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:14px 18px;text-align:center;}
.mlabel{color:#718096;font-size:.72rem;letter-spacing:2px;text-transform:uppercase;display:block;margin-bottom:5px;}
.mval{color:#e2e8f0;font-family:'Orbitron',sans-serif;font-size:1.1rem;font-weight:700;}

/* Engine badge */
.eng-card{background:rgba(6,12,24,.8);border:1px solid rgba(0,170,255,.2);border-left:4px solid #00aaff;
  border-radius:10px;padding:16px;margin-bottom:12px;}
.eng-name{color:#00aaff;font-family:'Orbitron',sans-serif;font-size:.9rem;font-weight:700;letter-spacing:1px;}
.eng-score-ok{color:#00ff88;font-family:'Orbitron',sans-serif;font-size:1.2rem;font-weight:700;}
.eng-score-warn{color:#ffcc00;font-family:'Orbitron',sans-serif;font-size:1.2rem;font-weight:700;}
.eng-score-bad{color:#ff3333;font-family:'Orbitron',sans-serif;font-size:1.2rem;font-weight:700;}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,rgba(0,255,136,.15),rgba(0,170,255,.1))!important;
  border:1px solid #00ff88!important;color:#00ff88!important;font-family:'Orbitron',sans-serif!important;
  letter-spacing:3px!important;padding:16px 32px!important;border-radius:8px!important;
  width:100%;font-weight:700;box-shadow:0 0 20px rgba(0,255,136,.2)!important;
  transition:all .25s!important;text-transform:uppercase;}
.stButton>button:hover{background:rgba(0,255,136,.25)!important;box-shadow:0 0 40px rgba(0,255,136,.5)!important;transform:scale(1.02);}

/* Upload zone */
[data-testid="stFileUploadDropzone"]{background:rgba(8,15,30,.6)!important;
  border:2px dashed rgba(0,170,255,.5)!important;border-radius:14px!important;padding:36px!important;}

/* Sidebar */
section[data-testid="stSidebar"]{background:rgba(2,4,10,.98)!important;border-right:1px solid rgba(0,255,136,.2);}

/* Progress */
.stProgress>div>div>div{background:linear-gradient(90deg,#00ff88,#00aaff)!important;box-shadow:0 0 12px #00ff88;}

/* Scrollbar */
::-webkit-scrollbar{width:6px;} ::-webkit-scrollbar-track{background:#020912;}
::-webkit-scrollbar-thumb{background:rgba(0,170,255,.4);border-radius:3px;}

/* Pulsing dot */
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.8);}}
@keyframes glitch{0%,100%{text-shadow:0 0 30px #ff3333;}25%{text-shadow:-3px 0 #ff6666,3px 0 #cc0000,0 0 30px #ff3333;}75%{text-shadow:3px 0 #ff6666,-3px 0 #cc0000,0 0 30px #ff3333;}}
@keyframes holo-glow{0%,100%{box-shadow:0 0 20px rgba(0,170,255,.3);}50%{box-shadow:0 0 50px rgba(0,170,255,.7),0 0 80px rgba(0,255,136,.3);}}
@keyframes scan-line{0%{top:-5px;}100%{top:105%;}}
@keyframes fade-in{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
</style>
"""

SCAN_CSS = """
<style>
[data-testid="stSidebar"]{display:none!important;}
header{display:none!important;}
.block-container{padding-top:0!important;padding-left:2rem!important;padding-right:2rem!important;max-width:1100px!important;}
.stApp{background:radial-gradient(ellipse at center top,#010d1a 0%,#000000 100%)!important;}

.scan-wrap{display:flex;flex-direction:column;align-items:center;padding:30px 0;}
.scan-hud-title{font-family:'Orbitron',sans-serif;font-size:2.4rem;color:#00ff88;
  letter-spacing:6px;text-shadow:0 0 20px #00ff88,0 0 40px rgba(0,255,136,.4);
  margin-bottom:4px;text-align:center;}
.scan-hud-sub{font-family:'Courier Prime',monospace;color:#00aaff;font-size:1rem;
  letter-spacing:3px;opacity:.8;margin-bottom:28px;text-align:center;}

/* Face scan box */
.face-scan-box{position:relative;width:340px;height:380px;margin:0 auto 24px;
  background:rgba(0,15,30,.5);border-radius:24px;overflow:hidden;
  border:1px solid rgba(0,170,255,.3);
  box-shadow:0 0 60px rgba(0,170,255,.2),inset 0 0 60px rgba(0,255,136,.05);}

/* Corner brackets */
.face-scan-box::before,.face-scan-box::after{content:'';position:absolute;width:40px;height:40px;border-color:#00ff88;border-style:solid;z-index:20;}
.face-scan-box::before{top:12px;left:12px;border-width:3px 0 0 3px;box-shadow:-4px -4px 10px rgba(0,255,136,.5);}
.face-scan-box::after{bottom:12px;right:12px;border-width:0 3px 3px 0;box-shadow:4px 4px 10px rgba(0,255,136,.5);}

.grid-overlay{position:absolute;inset:0;
  background-image:linear-gradient(rgba(0,170,255,.2) 1px,transparent 1px),linear-gradient(90deg,rgba(0,170,255,.2) 1px,transparent 1px);
  background-size:28px 28px;}
.laser-line{position:absolute;left:0;width:100%;height:3px;
  background:linear-gradient(90deg,transparent,#00ff88,transparent);
  box-shadow:0 0 16px #00ff88,0 0 32px rgba(0,255,136,.6);
  animation:scan-line 2.8s linear infinite;z-index:15;}

.face-svg{position:absolute;top:8%;left:8%;width:84%;height:84%;
  stroke:rgba(0,170,255,.85);fill:none;stroke-width:1.5;
  stroke-dasharray:900;stroke-dashoffset:900;
  animation:draw-face 3.5s ease forwards,pulse-face 3s ease-in-out 3.5s infinite;}

/* Waveform row */
.wave-row{display:flex;align-items:center;gap:3px;height:50px;margin:10px 0;}
.wave-bar{width:5px;border-radius:3px;background:linear-gradient(180deg,#00aaff,#00ff88);
  animation:wave-anim var(--dur,.6s) ease-in-out infinite alternate;}

/* Log terminal */
.log-terminal{background:rgba(0,5,12,.8);border:1px solid rgba(0,170,255,.3);border-left:3px solid #00aaff;
  border-radius:0 10px 10px 0;padding:18px 20px;min-height:180px;font-family:'Courier Prime',monospace;
  font-size:.95rem;color:#8bb4e7;overflow:hidden;}
.log-line{margin-bottom:6px;opacity:.9;text-shadow:0 0 4px rgba(139,180,231,.3);animation:fade-in .3s ease;}
.log-cursor{color:#00ff88;font-weight:bold;animation:pulse-dot 1s infinite;}

/* HUD stats row */
.hud-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0;}
.hud-stat{background:rgba(0,0,0,.5);border:1px solid rgba(0,170,255,.15);border-radius:8px;padding:10px;text-align:center;}
.hud-stat-label{color:#4a6070;font-size:.65rem;letter-spacing:2px;text-transform:uppercase;}
.hud-stat-val{color:#00aaff;font-family:'Orbitron',sans-serif;font-size:.9rem;font-weight:700;}

@keyframes draw-face{to{stroke-dashoffset:0;}}
@keyframes pulse-face{0%,100%{filter:drop-shadow(0 0 4px #00aaff);}50%{filter:drop-shadow(0 0 18px #00aaff) drop-shadow(0 0 30px rgba(0,255,136,.4));}}
@keyframes wave-anim{from{height:10%;}to{height:90%;}}
@keyframes scan-line{0%{top:-5px;}100%{top:105%;}}
@keyframes fade-in{from{opacity:0;}to{opacity:.9;}}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0;}}
</style>
"""

FACE_SVG_HTML = """
<div class="face-scan-box">
  <div class="grid-overlay"></div>
  <svg class="face-svg" viewBox="0 0 100 110">
    <!-- Outer face outline -->
    <path d="M50 6 C28 6 16 24 16 48 C16 72 34 96 50 102 C66 96 84 72 84 48 C84 24 72 6 50 6 Z"/>
    <!-- Eye sockets -->
    <ellipse cx="35" cy="44" rx="9" ry="6"/>
    <ellipse cx="65" cy="44" rx="9" ry="6"/>
    <!-- Pupils -->
    <circle cx="35" cy="44" r="3" fill="rgba(0,255,136,.6)" stroke="none"/>
    <circle cx="65" cy="44" r="3" fill="rgba(0,255,136,.6)" stroke="none"/>
    <!-- Nose bridge -->
    <path d="M50 36 L50 60"/>
    <path d="M44 60 Q50 65 56 60"/>
    <!-- Mouth -->
    <path d="M36 76 Q50 85 64 76"/>
    <!-- Cheek markers -->
    <line x1="16" y1="54" x2="28" y2="54" stroke="#00ff88" stroke-width="1"/>
    <line x1="72" y1="54" x2="84" y2="54" stroke="#00ff88" stroke-width="1"/>
    <!-- Forehead center -->
    <circle cx="50" cy="20" r="1.5" fill="#00aaff" stroke="none"/>
    <line x1="50" y1="22" x2="50" y2="28" stroke="#00aaff" stroke-width="1"/>
    <!-- Eyebrow lines -->
    <path d="M26 37 Q35 33 44 36"/>
    <path d="M56 36 Q65 33 74 37"/>
    <!-- Landmark dots -->
    <circle cx="35" cy="38" r="1.2" fill="#00aaff" stroke="none"/>
    <circle cx="65" cy="38" r="1.2" fill="#00aaff" stroke="none"/>
    <circle cx="50" cy="102" r="1.5" fill="#00ff88" stroke="none"/>
    <circle cx="16" cy="48" r="1.2" fill="#00ff88" stroke="none"/>
    <circle cx="84" cy="48" r="1.2" fill="#00ff88" stroke="none"/>
  </svg>
  <div class="laser-line"></div>
</div>
"""

def waveform_html(n=40):
    bars = ""
    import random
    for i in range(n):
        dur = round(random.uniform(0.3, 1.2), 2)
        bars += f'<div class="wave-bar" style="--dur:{dur}s;height:50%"></div>'
    return f'<div class="wave-row">{bars}</div>'


def log_box_html(lines):
    rows = "".join(f'<div class="log-line">{l}</div>' for l in lines[-7:])
    return f'<div class="log-terminal">{rows}<div class="log-cursor">█</div></div>'


def hud_stat_box(label, val):
    return f'<div class="hud-stat"><div class="hud-stat-label">{label}</div><div class="hud-stat-val">{val}</div></div>'


def meta_chip(label, val):
    return f"""
    <div style="background:rgba(0,0,0,.5);border:1px solid rgba(0,170,255,.2);border-radius:10px;
      padding:14px;text-align:center;">
      <div style="color:#4a6070;font-size:.7rem;letter-spacing:2px;margin-bottom:6px;">{label}</div>
      <div style="color:#e2e8f0;font-family:'Orbitron',sans-serif;font-size:.95rem;font-weight:700;">{val}</div>
    </div>"""


def engine_card_html(name, score, status, summary):
    if score > 65:
        sc = f'<span class="eng-score-bad">{score}/100</span>'
        left = "#ff3333"
    elif score > 40:
        sc = f'<span class="eng-score-warn">{score}/100</span>'
        left = "#ffcc00"
    else:
        sc = f'<span class="eng-score-ok">{score}/100</span>'
        left = "#00ff88"
    return f"""
    <div class="eng-card" style="border-left-color:{left};">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div class="eng-name">{name}</div>{sc}
      </div>
      <div style="color:#a0aec0;font-size:.8rem;margin-top:6px;">{status}</div>
      <div style="color:#4a6070;font-size:.73rem;font-style:italic;margin-top:3px;">{summary}</div>
    </div>"""


# ── AUDIO-SPECIFIC CSS ────────────────────────────────────────────────────────
AUDIO_SCAN_CSS = """
<style>
/* Audio waveform visualizer */
.audio-wave-container{display:flex;align-items:center;gap:3px;height:80px;justify-content:center;
  margin:18px 0;padding:0 12px;background:rgba(0,5,12,.6);border-radius:12px;
  border:1px solid rgba(0,170,255,.2);}
.audio-bar{width:4px;border-radius:4px;background:linear-gradient(180deg,#00aaff,#00ff88);
  animation:audio-pulse var(--dur,.5s) ease-in-out infinite alternate;}
@keyframes audio-pulse{from{height:8%;opacity:.4;}to{height:92%;opacity:1;}}

/* Spectrogram display */
.spectrogram-box{background:rgba(0,5,12,.8);border:1px solid rgba(0,170,255,.25);border-radius:12px;
  padding:16px;overflow:hidden;position:relative;}
.spec-title{color:#00aaff;font-family:'Orbitron',sans-serif;font-size:.75rem;letter-spacing:2px;
  margin-bottom:10px;text-transform:uppercase;}

/* Neural voice scan */
.voice-scan-orb{width:180px;height:180px;border-radius:50%;margin:0 auto 20px;
  background:radial-gradient(circle,rgba(0,170,255,.3),rgba(0,255,136,.1),transparent);
  border:2px solid rgba(0,170,255,.5);position:relative;
  box-shadow:0 0 60px rgba(0,170,255,.3),0 0 100px rgba(0,255,136,.1);
  animation:orb-pulse 2s ease-in-out infinite;}
.voice-scan-orb::before{content:'';position:absolute;inset:12px;border-radius:50%;
  border:1px dashed rgba(0,255,136,.4);animation:spin 8s linear infinite;}
.voice-scan-orb::after{content:'';position:absolute;inset:28px;border-radius:50%;
  border:1px solid rgba(0,170,255,.3);animation:spin 5s linear infinite reverse;}
.orb-center{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-family:'Orbitron',sans-serif;font-size:.75rem;color:#00ff88;letter-spacing:2px;text-align:center;}
@keyframes orb-pulse{0%,100%{box-shadow:0 0 60px rgba(0,170,255,.3),0 0 100px rgba(0,255,136,.1);}
  50%{box-shadow:0 0 100px rgba(0,170,255,.6),0 0 150px rgba(0,255,136,.3);}}
@keyframes spin{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}

/* Frequency pulse rings */
.pulse-ring{position:absolute;top:50%;left:50%;border-radius:50%;border:1px solid rgba(0,170,255,.4);
  transform:translate(-50%,-50%);animation:ring-expand 2.4s ease-out infinite;}
.pulse-ring:nth-child(2){animation-delay:.8s;}
.pulse-ring:nth-child(3){animation-delay:1.6s;}
@keyframes ring-expand{0%{width:60px;height:60px;opacity:.8;}100%{width:300px;height:300px;opacity:0;}}
</style>
"""

def audio_waveform_html(n=60):
    import random
    bars = ""
    for i in range(n):
        dur = round(random.uniform(0.2, 1.0), 2)
        bars += f'<div class="audio-bar" style="--dur:{dur}s;height:30%"></div>'
    return f'<div class="audio-wave-container">{bars}</div>'

def voice_scan_orb_html():
    return """
<div style="position:relative;width:200px;height:200px;margin:0 auto 20px;">
  <div class="voice-scan-orb">
    <div class="orb-center">VOICE<br>SCAN<br>ACTIVE</div>
  </div>
  <div class="pulse-ring"></div>
  <div class="pulse-ring"></div>
  <div class="pulse-ring"></div>
</div>"""

