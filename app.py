"""
🚗 Driver Drowsiness Detection — Streamlit App 
=======================================================
KEY FIX: Uses a thread-safe class to share state between
         the webrtc callback thread and the Streamlit UI thread.

INSTALL:
    pip install streamlit streamlit-webrtc opencv-python-headless mediapipe numpy scipy av

RUN:
    streamlit run app.py
"""

import os, sys, time, threading, urllib.request
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["LIBGL_ALWAYS_SOFTWARE"]    = "1"
os.environ["QT_QPA_PLATFORM"]          = "offscreen"
os.environ["DISPLAY"]                  = ""
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from scipy.spatial import distance as dist
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
.stApp { background: #07090f; color: #dde6ff; }
.title { font-size:2.4rem; font-weight:700; letter-spacing:.1em;
         background: linear-gradient(90deg,#00cfff,#0055ff);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.mono  { font-family:'Share Tech Mono',monospace; color:#3a6090; font-size:.78rem; letter-spacing:.18em; }
.card  { background:#0c1220; border:1px solid #1a2d55; border-radius:12px;
         padding:16px 18px; margin:6px 0; }
.card-label { font-family:'Share Tech Mono',monospace; font-size:.68rem;
              color:#3a5880; letter-spacing:.2em; text-transform:uppercase; }
.card-val   { font-size:2rem; font-weight:700; color:#00cfff; }
.card-val.red   { color:#ff3355; }
.card-val.green { color:#00e676; }
.card-val.amber { color:#ffaa00; }
.alert-on  { background:#1a0010; border:2px solid #ff3355; border-radius:12px;
             padding:18px; text-align:center; }
.alert-off { background:#001510; border:1px solid #00e676; border-radius:12px;
             padding:18px; text-align:center; }
section[data-testid="stSidebar"] { background:#060810; border-right:1px solid #1a2d55; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  MODEL — load once, cache forever
# ══════════════════════════════════════════════════════════
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

@st.cache_resource(show_spinner="Loading face model …")
def load_landmarker():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode)

    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(opts)

landmarker = load_landmarker()

# ══════════════════════════════════════════════════════════
#  THREAD-SAFE STATE CLASS
#  This is the critical fix — st.session_state cannot be
#  accessed from the webrtc callback thread safely.
#  We use a plain Python object with a Lock instead.
# ══════════════════════════════════════════════════════════
class DetectionState:
    def __init__(self):
        self._lock       = threading.Lock()
        self.ear         = 0.30
        self.jaw         = 0.0
        self.eye_state   = "OPEN"      # "OPEN" | "CLOSED"
        self.secs_closed = 0.0
        self.close_time  = 0.0
        self.blinks      = 0
        self.yawns       = 0
        self.yawn_frames = 0
        self._in_yawn    = False
        self.alert       = ""          # "" | "DROWSY" | "YAWN"
        self.face_found  = False
        self.ear_history = []          # last 3 frames
        self.jaw_history = []

    def get(self):
        """Snapshot for the UI — always safe to call."""
        with self._lock:
            return dict(
                ear         = self.ear,
                jaw         = self.jaw,
                eye_state   = self.eye_state,
                secs_closed = self.secs_closed,
                blinks      = self.blinks,
                yawns       = self.yawns,
                yawn_frames = self.yawn_frames,
                alert       = self.alert,
                face_found  = self.face_found,
            )

    def update(self, ear_raw, jaw_raw, face_ok, ear_thr, jaw_thr, drowsy_sec):
        """Called from the webrtc thread — all logic lives here."""
        now = time.time()
        with self._lock:
            self.face_found = face_ok

            if not face_ok:
                self.eye_state   = "OPEN"
                self.secs_closed = 0.0
                self.ear         = 0.30
                self.jaw         = 0.0
                self.alert       = ""
                return

            # ── Smooth ─────────────────────────────────
            self.ear_history.append(ear_raw)
            self.jaw_history.append(jaw_raw)
            if len(self.ear_history) > 3: self.ear_history.pop(0)
            if len(self.jaw_history) > 5: self.jaw_history.pop(0)
            self.ear = float(np.mean(self.ear_history))
            self.jaw = float(np.mean(self.jaw_history))

            # ── Blink state machine (raw ear, not smoothed) ──
            blink_thr = ear_thr  # same threshold for blink close

            if self.eye_state == "OPEN":
                if ear_raw < blink_thr:
                    self.eye_state  = "CLOSED"
                    self.close_time = now
            else:  # CLOSED
                self.secs_closed = now - self.close_time
                if ear_raw >= blink_thr:
                    dur = self.secs_closed
                    self.eye_state   = "OPEN"
                    self.secs_closed = 0.0
                    if 0.05 < dur < 0.5:      # valid blink duration
                        self.blinks += 1

            # ── Yawn (jaw blendshape) ───────────────────
            if self.jaw > jaw_thr:
                self.yawn_frames = min(self.yawn_frames + 1, 60)
            else:
                self.yawn_frames = max(self.yawn_frames - 1, 0)

            if self.yawn_frames >= 20:
                if not self._in_yawn:
                    self.yawns    += 1
                    self._in_yawn  = True
            else:
                self._in_yawn = False

            # ── Alert ───────────────────────────────────
            if self.eye_state == "CLOSED" and self.secs_closed >= drowsy_sec:
                self.alert = "DROWSY"
            elif self.yawn_frames >= 20:
                self.alert = "YAWN"
            else:
                self.alert = ""

    def reset_counters(self):
        with self._lock:
            self.blinks = 0
            self.yawns  = 0

# ── Store one instance in session_state (survives reruns) ──
if "det" not in st.session_state:
    st.session_state.det = DetectionState()

det = st.session_state.det   # shorthand

# ══════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ══════════════════════════════════════════════════════════
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_8   = [61,  40,  37,   0, 267, 270, 291, 321]

def calc_ear(lm, idx, W, H):
    p = np.array([[lm[i].x*W, lm[i].y*H] for i in idx])
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2.0*C + 1e-6)

def draw_region(img, lm, idx, W, H, color):
    pts = np.array([[int(lm[i].x*W), int(lm[i].y*H)] for i in idx])
    cv2.polylines(img, [pts], True, color, 1)
    for p in pts:
        cv2.circle(img, tuple(p), 2, color, -1)

# ══════════════════════════════════════════════════════════
#  VIDEO CALLBACK — runs in webrtc thread
# ══════════════════════════════════════════════════════════
def video_callback(frame):
    img  = frame.to_ndarray(format="bgr24")
    img  = cv2.flip(img, 1)
    H, W = img.shape[:2]

    # Read thresholds from session_state (set by UI thread)
    # These are simple floats — safe to read across threads
    ear_thr    = getattr(st.session_state, "_ear_thr",    0.20)
    jaw_thr    = getattr(st.session_state, "_jaw_thr",    0.30)
    drowsy_sec = getattr(st.session_state, "_drowsy_sec", 1.5)

    # ── MediaPipe detect ──────────────────────────────────
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result  = landmarker.detect(mp_img)

    ear_raw = 0.30
    jaw_raw = 0.0
    face_ok = False

    if result.face_landmarks:
        face_ok = True
        lm = result.face_landmarks[0]

        le = calc_ear(lm, LEFT_EYE,  W, H)
        re = calc_ear(lm, RIGHT_EYE, W, H)
        ear_raw = (le + re) / 2.0

        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            jaw_raw = bs.get("jawOpen", 0.0)

        # Draw landmarks
        draw_region(img, lm, LEFT_EYE,  W, H, (0, 225, 80))
        draw_region(img, lm, RIGHT_EYE, W, H, (0, 225, 80))
        draw_region(img, lm, MOUTH_8,   W, H, (0, 180, 255))

    # ── Update thread-safe state ──────────────────────────
    det.update(ear_raw, jaw_raw, face_ok, ear_thr, jaw_thr, drowsy_sec)
    snap = det.get()

    # ── Draw HUD on frame ─────────────────────────────────
    cv2.rectangle(img, (0, 0), (W, 100), (8, 10, 20), -1)

    # EAR bar
    ear_col = (0, 55, 255) if snap["eye_state"] == "CLOSED" else (0, 200, 80)
    bw = int(min(snap["ear"] / 0.45, 1.0) * 200)
    cv2.rectangle(img, (130, 8),  (330, 28), (25, 35, 65), -1)
    cv2.rectangle(img, (130, 8),  (130+bw, 28), ear_col, -1)
    cv2.putText(img, f"EAR {snap['ear']:.3f}  thr={ear_thr:.2f}",
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_col, 2)

    # JAW bar
    jaw_col = (0, 55, 255) if snap["jaw"] > jaw_thr else (0, 200, 80)
    jw = int(min(snap["jaw"] / 0.8, 1.0) * 200)
    cv2.rectangle(img, (130, 35), (330, 55), (25, 35, 65), -1)
    cv2.rectangle(img, (130, 35), (130+jw, 55), jaw_col, -1)
    cv2.putText(img, f"JAW {snap['jaw']:.3f}  thr={jaw_thr:.2f}",
                (5, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, jaw_col, 2)

    # Eye state
    eye_lbl = f"Eye: {snap['eye_state']}"
    if snap["eye_state"] == "CLOSED":
        eye_lbl += f"  {snap['secs_closed']:.1f}s"
    cv2.putText(img, eye_lbl, (5, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 55, 255) if snap["eye_state"] == "CLOSED" else (0, 200, 80), 2)

    cv2.putText(img, f"Blinks:{snap['blinks']}  Yawns:{snap['yawns']}",
                (W - 195, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 1)

    # Alert banner
    if snap["alert"] == "DROWSY":
        cv2.rectangle(img, (0, H-65), (W, H), (0, 0, 160), -1)
        cv2.putText(img, "  DROWSY! WAKE UP!", (10, H-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    elif snap["alert"] == "YAWN":
        cv2.rectangle(img, (0, H-65), (W, H), (0, 80, 0), -1)
        cv2.putText(img, "  YAWN DETECTED!", (10, H-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    if not face_ok:
        cv2.putText(img, "No face detected — look at camera",
                    (W//2 - 185, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 40, 220), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ══════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════
st.markdown('<div class="title">🚗 Driver Drowsiness Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="mono">// REAL-TIME FATIGUE MONITORING</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar — thresholds
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ear_thr = st.slider("EAR Threshold", 0.10, 0.35, 0.20, 0.01,
        help="Eye closed if EAR drops BELOW this. Increase if blinks not detected.")
    jaw_thr = st.slider("JAW Threshold", 0.10, 0.70, 0.30, 0.01,
        help="Yawn if jawOpen score is ABOVE this. Decrease if yawns not detected.")
    drowsy_sec = st.slider("Drowsy delay (seconds)", 0.5, 4.0, 1.5, 0.25,
        help="How long eyes must be closed before alert.")

    # Write thresholds to session_state so callback thread can read them
    st.session_state._ear_thr    = ear_thr
    st.session_state._jaw_thr    = jaw_thr
    st.session_state._drowsy_sec = drowsy_sec

    st.markdown("---")
    st.markdown("**How to tune:**")
    st.markdown("""
- Watch the **EAR value** in the video when your eyes are fully open.
- Then slowly close them — EAR should drop noticeably.
- Set threshold between those two values.
- Similarly watch **JAW** when you yawn vs normal.
    """)
    if st.button("🔄 Reset counters"):
        det.reset_counters()
        st.rerun()

# Main layout
col_video, col_stats = st.columns([2.2, 1], gap="large")

with col_video:
    st.markdown("**📷 Live Feed** — allow camera access when prompted")
    webrtc_streamer(
        key="drowsiness-v2",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_callback,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

with col_stats:
    snap = det.get()

    # Alert box
    if snap["alert"] == "DROWSY":
        st.markdown("""<div class="alert-on">
            <div style="font-size:2.5rem">😴</div>
            <div style="font-size:1.4rem;font-weight:700;color:#ff3355">DROWSY! WAKE UP!</div>
        </div>""", unsafe_allow_html=True)
    elif snap["alert"] == "YAWN":
        st.markdown("""<div class="alert-on" style="border-color:#ffaa00;background:#1a1000">
            <div style="font-size:2.5rem">🥱</div>
            <div style="font-size:1.4rem;font-weight:700;color:#ffaa00">YAWN DETECTED!</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="alert-off">
            <div style="font-size:2.5rem">✅</div>
            <div style="font-size:1.2rem;font-weight:600;color:#00e676">DRIVER ALERT</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # EAR card
    ear_cls = "red" if snap["ear"] < ear_thr else "green"
    ear_pct = int(min(snap["ear"]/0.45, 1.0)*100)
    st.markdown(f"""<div class="card">
        <div class="card-label">Eye Aspect Ratio (EAR)</div>
        <div class="card-val {ear_cls}">{snap['ear']:.3f}</div>
        <div style="background:#1a2540;border-radius:4px;height:8px;margin:6px 0">
          <div style="background:{'#ff3355' if ear_cls=='red' else '#00e676'};
                      width:{ear_pct}%;height:8px;border-radius:4px"></div>
        </div>
        <div style="font-size:.68rem;color:#3a5080">threshold &lt; {ear_thr:.2f}</div>
    </div>""", unsafe_allow_html=True)

    # JAW card
    jaw_cls = "red" if snap["jaw"] > jaw_thr else "green"
    jaw_pct = int(min(snap["jaw"]/0.8, 1.0)*100)
    st.markdown(f"""<div class="card">
        <div class="card-label">Jaw Open Score</div>
        <div class="card-val {jaw_cls}">{snap['jaw']:.3f}</div>
        <div style="background:#1a2540;border-radius:4px;height:8px;margin:6px 0">
          <div style="background:{'#ff3355' if jaw_cls=='red' else '#00e676'};
                      width:{jaw_pct}%;height:8px;border-radius:4px"></div>
        </div>
        <div style="font-size:.68rem;color:#3a5080">threshold &gt; {jaw_thr:.2f}</div>
    </div>""", unsafe_allow_html=True)

    # Eye state
    eye_cls = "red" if snap["eye_state"] == "CLOSED" else "green"
    eye_val = snap["eye_state"]
    if snap["eye_state"] == "CLOSED":
        eye_val += f"  {snap['secs_closed']:.1f}s"
    st.markdown(f"""<div class="card">
        <div class="card-label">Eye State</div>
        <div class="card-val {eye_cls}">{eye_val}</div>
    </div>""", unsafe_allow_html=True)

    # Blinks + Yawns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="card">
            <div class="card-label">Blinks</div>
            <div class="card-val">{snap['blinks']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        yawn_cls = "amber" if snap["yawns"] > 2 else ""
        st.markdown(f"""<div class="card">
            <div class="card-label">Yawns</div>
            <div class="card-val {yawn_cls}">{snap['yawns']}</div>
        </div>""", unsafe_allow_html=True)


    # ── SOUND: persistent iframe with Web Audio API ───────
    # st.markdown <script> tags get killed on every rerun.
    # components.html() creates an iframe that SURVIVES reruns
    # and keeps playing as long as the alert is active.
    import streamlit.components.v1 as components

    alert_now = snap["alert"]
    if alert_now == "DROWSY":
        freq, repeat_ms = 1200, 1500
    elif alert_now == "YAWN":
        freq, repeat_ms = 850, 2500
    else:
        freq, repeat_ms = 0, 0

    components.html(f"""<!DOCTYPE html><html><body style="margin:0">
    <script>
    const FREQ = {freq};
    const RPT  = {repeat_ms};
    function beep() {{
        if (!FREQ) return;
        const ctx = new (window.AudioContext||window.webkitAudioContext)();
        [0, 0.6].forEach(function(t) {{
            var o = ctx.createOscillator(), g = ctx.createGain();
            o.connect(g); g.connect(ctx.destination);
            o.type = 'sine'; o.frequency.value = FREQ;
            g.gain.setValueAtTime(1, ctx.currentTime + t);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + t + 0.5);
            o.start(ctx.currentTime + t);
            o.stop(ctx.currentTime + t + 0.55);
        }});
    }}
    if (FREQ) {{ beep(); setInterval(beep, RPT); }}
    </script></body></html>""", height=0)

    # Rerun every 900ms to refresh stats — st.rerun() is safe,
    # unlike location.reload() which would kill the audio iframe
    time.sleep(0.9)
    st.rerun()
