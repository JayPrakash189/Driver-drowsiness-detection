"""
Driver Drowsiness Detection - Streamlit App
============================================
FIX 1: env vars set BEFORE cv2 import (fixes libGL error)
FIX 2: RunningMode.IMAGE used (VIDEO mode needs strict timestamps)
FIX 3: Thread-safe DetectionState class
FIX 4: packages.txt uses correct Debian package names
"""

# ============================================================
# CRITICAL: env vars MUST be set before ANY other import
# ============================================================
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["LIBGL_ALWAYS_SOFTWARE"]    = "1"
os.environ["QT_QPA_PLATFORM"]          = "offscreen"
os.environ["DISPLAY"]                  = ""

# ============================================================
# Now safe to import cv2 and others
# ============================================================
import sys
import time
import threading
import urllib.request
from collections import deque

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from scipy.spatial import distance as dist
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="Car",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background: #07090f; color: #dde6ff; }
.title { font-size:2.2rem; font-weight:700; color:#00cfff; }
.card  { background:#0c1220; border:1px solid #1a2d55;
         border-radius:12px; padding:16px 18px; margin:6px 0; }
.card-label { font-size:.68rem; color:#3a5880;
              text-transform:uppercase; letter-spacing:.15em; }
.card-val   { font-size:1.8rem; font-weight:700; color:#00cfff; }
.card-val.red   { color:#ff3355; }
.card-val.green { color:#00e676; }
.card-val.amber { color:#ffaa00; }
.alert-on  { background:#1a0010; border:2px solid #ff3355;
             border-radius:12px; padding:18px; text-align:center; }
.alert-off { background:#001510; border:1px solid #00e676;
             border-radius:12px; padding:18px; text-align:center; }
section[data-testid="stSidebar"] {
    background:#060810; border-right:1px solid #1a2d55;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL DOWNLOAD + LOAD
# ============================================================
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

@st.cache_resource(show_spinner="Loading face model ...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode
    )
    # FIX: Use RunningMode.IMAGE not VIDEO
    # VIDEO mode requires strictly increasing timestamps which
    # streamlit-webrtc does not guarantee across threads
    opts = FaceLandmarkerOptions(
        base_options                  = BaseOptions(model_asset_path=MODEL_PATH),
        running_mode                  = RunningMode.IMAGE,
        num_faces                     = 1,
        min_face_detection_confidence = 0.5,
        min_face_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
        output_face_blendshapes       = True,
    )
    return FaceLandmarker.create_from_options(opts)

try:
    landmarker = load_model()
    model_ok   = True
except Exception as e:
    st.error(f"Model load failed: {e}")
    model_ok = False

# ============================================================
# LANDMARK INDICES
# ============================================================
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_8   = [61,   40,  37,   0, 267, 270, 291, 321]

def calc_ear(lm, idx, W, H):
    p = np.array([[lm[i].x * W, lm[i].y * H] for i in idx])
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2.0 * C + 1e-6)

def draw_landmarks(img, lm, idx, W, H, color):
    pts = np.array([[int(lm[i].x * W), int(lm[i].y * H)] for i in idx])
    cv2.polylines(img, [pts], True, color, 1)
    for p in pts:
        cv2.circle(img, tuple(p), 2, color, -1)

# ============================================================
# THREAD-SAFE STATE
# FIX: st.session_state cannot be accessed from webrtc thread
# Use a plain Python class with threading.Lock instead
# ============================================================
class DetectionState:
    def __init__(self):
        self._lock       = threading.Lock()
        self.ear         = 0.30
        self.jaw         = 0.0
        self.eye_state   = "OPEN"
        self.secs_closed = 0.0
        self.close_time  = 0.0
        self.blinks      = 0
        self.yawns       = 0
        self.yawn_frames = 0
        self._in_yawn    = False
        self.alert       = ""
        self.alert_type  = ""
        self.face_found  = False
        self._ear_hist   = []
        self._jaw_hist   = []

    def snapshot(self):
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
                alert_type  = self.alert_type,
                face_found  = self.face_found,
            )

    def update(self, ear_raw, jaw_raw, face_ok,
               ear_thr, jaw_thr, drowsy_sec):
        now = time.time()
        with self._lock:
            self.face_found = face_ok
            if not face_ok:
                self.eye_state   = "OPEN"
                self.secs_closed = 0.0
                self.alert       = ""
                self.alert_type  = ""
                return

            # Smooth
            self._ear_hist.append(ear_raw)
            self._jaw_hist.append(jaw_raw)
            if len(self._ear_hist) > 3: self._ear_hist.pop(0)
            if len(self._jaw_hist) > 5: self._jaw_hist.pop(0)
            self.ear = float(np.mean(self._ear_hist))
            self.jaw = float(np.mean(self._jaw_hist))

            # Blink state machine (raw EAR)
            if self.eye_state == "OPEN":
                if ear_raw < ear_thr:
                    self.eye_state  = "CLOSED"
                    self.close_time = now
            else:
                self.secs_closed = now - self.close_time
                if ear_raw >= ear_thr:
                    dur = self.secs_closed
                    self.eye_state   = "OPEN"
                    self.secs_closed = 0.0
                    if 0.05 < dur < 0.5:
                        self.blinks += 1

            # Yawn
            if self.jaw > jaw_thr:
                self.yawn_frames = min(self.yawn_frames + 1, 60)
            else:
                self.yawn_frames = max(self.yawn_frames - 1, 0)

            if self.yawn_frames >= 20:
                if not self._in_yawn:
                    self.yawns   += 1
                    self._in_yawn = True
            else:
                self._in_yawn = False

            # Alert
            if self.eye_state == "CLOSED" and self.secs_closed >= drowsy_sec:
                self.alert      = "DROWSY! WAKE UP!"
                self.alert_type = "DROWSY"
            elif self.yawn_frames >= 20:
                self.alert      = "YAWN DETECTED!"
                self.alert_type = "YAWN"
            else:
                self.alert      = ""
                self.alert_type = ""

    def reset_counters(self):
        with self._lock:
            self.blinks = 0
            self.yawns  = 0


# One DetectionState instance per session
if "det" not in st.session_state:
    st.session_state.det = DetectionState()

det = st.session_state.det

# ============================================================
# VIDEO CALLBACK (runs in webrtc background thread)
# ============================================================
def video_callback(frame):
    img  = frame.to_ndarray(format="bgr24")
    img  = cv2.flip(img, 1)
    H, W = img.shape[:2]

    # Read thresholds (simple floats - safe cross-thread read)
    ear_thr    = float(getattr(st.session_state, "_ear_thr",    0.20))
    jaw_thr    = float(getattr(st.session_state, "_jaw_thr",    0.30))
    drowsy_sec = float(getattr(st.session_state, "_drowsy_sec", 1.5))

    ear_raw = 0.30
    jaw_raw = 0.0
    face_ok = False

    try:
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
        # FIX: detect() not detect_for_video()
        result = landmarker.detect(mp_img)

        if result.face_landmarks:
            face_ok = True
            lm      = result.face_landmarks[0]
            le      = calc_ear(lm, LEFT_EYE,  W, H)
            re      = calc_ear(lm, RIGHT_EYE, W, H)
            ear_raw = (le + re) / 2.0

            if result.face_blendshapes:
                bs      = {c.category_name: c.score
                           for c in result.face_blendshapes[0]}
                jaw_raw = bs.get("jawOpen", 0.0)

            draw_landmarks(img, lm, LEFT_EYE,  W, H, (0, 225, 80))
            draw_landmarks(img, lm, RIGHT_EYE, W, H, (0, 225, 80))
            draw_landmarks(img, lm, MOUTH_8,   W, H, (0, 180, 255))

    except Exception:
        pass

    det.update(ear_raw, jaw_raw, face_ok, ear_thr, jaw_thr, drowsy_sec)
    snap = det.snapshot()

    # Draw HUD on frame
    cv2.rectangle(img, (0, 0), (W, 95), (10, 14, 28), -1)

    ear_col = (0, 55, 255) if snap["eye_state"] == "CLOSED" else (0, 210, 80)
    bw = int(min(snap["ear"] / 0.45, 1.0) * 200)
    cv2.rectangle(img, (130, 8),  (330, 28), (30, 40, 70), -1)
    cv2.rectangle(img, (130, 8),  (130 + bw, 28), ear_col, -1)
    cv2.putText(img, f"EAR {snap['ear']:.3f}",
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_col, 2)

    jaw_col = (0, 55, 255) if snap["jaw"] > jaw_thr else (0, 210, 80)
    jw = int(min(snap["jaw"] / 0.8, 1.0) * 200)
    cv2.rectangle(img, (130, 35), (330, 55), (30, 40, 70), -1)
    cv2.rectangle(img, (130, 35), (130 + jw, 55), jaw_col, -1)
    cv2.putText(img, f"JAW {snap['jaw']:.3f}",
                (5, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, jaw_col, 2)

    eye_lbl = f"Eye:{snap['eye_state']}"
    if snap["eye_state"] == "CLOSED":
        eye_lbl += f" {snap['secs_closed']:.1f}s"
    cv2.putText(img, eye_lbl, (5, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 55, 255) if snap["eye_state"] == "CLOSED" else (0, 200, 80), 2)

    if snap["alert"]:
        cv2.rectangle(img, (0, H - 60), (W, H), (0, 0, 160), -1)
        cv2.putText(img, snap["alert"], (12, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if not face_ok:
        cv2.putText(img, "No face - look at camera",
                    (W // 2 - 150, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 40, 220), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================================
# UI LAYOUT
# ============================================================
st.markdown('<div class="title">Driver Drowsiness Detection</div>',
            unsafe_allow_html=True)
st.markdown("Real-time fatigue monitoring using MediaPipe + WebRTC")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    ear_thr = st.slider("EAR Threshold (eye closed)", 0.10, 0.35, 0.20, 0.01)
    jaw_thr = st.slider("JAW Threshold (yawn)", 0.10, 0.70, 0.30, 0.01)
    drowsy_sec = st.slider("Drowsy delay (seconds)", 0.5, 4.0, 1.5, 0.25)

    st.session_state._ear_thr    = ear_thr
    st.session_state._jaw_thr    = jaw_thr
    st.session_state._drowsy_sec = drowsy_sec

    st.markdown("---")
    st.markdown("**Tuning guide:**")
    st.markdown("Watch EAR value when eyes open vs closed. Set threshold between the two values.")

    if st.button("Reset counters"):
        det.reset_counters()
        st.rerun()

# Main columns
col_video, col_stats = st.columns([2.2, 1], gap="large")

with col_video:
    st.markdown("**Live Camera** - click Allow when browser asks for camera")
    if model_ok:
        webrtc_streamer(
            key="drowsiness-det",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=video_callback,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True,
        )
    else:
        st.error("Model not loaded.")

with col_stats:
    snap = det.snapshot()

    # Alert box
    if snap["alert_type"] == "DROWSY":
        st.markdown("""<div class="alert-on">
            <div style="font-size:2rem">!</div>
            <div style="font-size:1.3rem;font-weight:700;color:#ff3355">
            DROWSY! WAKE UP!</div></div>""",
            unsafe_allow_html=True)
    elif snap["alert_type"] == "YAWN":
        st.markdown("""<div class="alert-on"
            style="border-color:#ffaa00;background:#1a1000">
            <div style="font-size:2rem">!</div>
            <div style="font-size:1.3rem;font-weight:700;color:#ffaa00">
            YAWN DETECTED!</div></div>""",
            unsafe_allow_html=True)
    else:
        st.markdown("""<div class="alert-off">
            <div style="font-size:2rem">OK</div>
            <div style="font-size:1.1rem;font-weight:600;color:#00e676">
            DRIVER ALERT</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # EAR
    ear_cls = "red" if snap["ear"] < ear_thr else "green"
    ear_pct = int(min(snap["ear"] / 0.45, 1.0) * 100)
    st.markdown(f"""<div class="card">
        <div class="card-label">Eye Aspect Ratio</div>
        <div class="card-val {ear_cls}">{snap['ear']:.3f}</div>
        <div style="background:#1a2540;border-radius:4px;height:7px;margin:5px 0">
        <div style="background:{'#ff3355' if ear_cls=='red' else '#00e676'};
            width:{ear_pct}%;height:7px;border-radius:4px"></div></div>
        <div style="font-size:.65rem;color:#3a5080">closed if below {ear_thr:.2f}</div>
    </div>""", unsafe_allow_html=True)

    # JAW
    jaw_cls = "red" if snap["jaw"] > jaw_thr else "green"
    jaw_pct = int(min(snap["jaw"] / 0.8, 1.0) * 100)
    st.markdown(f"""<div class="card">
        <div class="card-label">Jaw Open Score</div>
        <div class="card-val {jaw_cls}">{snap['jaw']:.3f}</div>
        <div style="background:#1a2540;border-radius:4px;height:7px;margin:5px 0">
        <div style="background:{'#ff3355' if jaw_cls=='red' else '#00e676'};
            width:{jaw_pct}%;height:7px;border-radius:4px"></div></div>
        <div style="font-size:.65rem;color:#3a5080">yawn if above {jaw_thr:.2f}</div>
    </div>""", unsafe_allow_html=True)

    # Eye state
    eye_cls = "red" if snap["eye_state"] == "CLOSED" else "green"
    eye_val = snap["eye_state"]
    if snap["eye_state"] == "CLOSED":
        eye_val += f" {snap['secs_closed']:.1f}s"
    st.markdown(f"""<div class="card">
        <div class="card-label">Eye State</div>
        <div class="card-val {eye_cls}">{eye_val}</div>
    </div>""", unsafe_allow_html=True)

    # Counts
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="card">
            <div class="card-label">Blinks</div>
            <div class="card-val">{snap['blinks']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        y_cls = "amber" if snap["yawns"] > 2 else ""
        st.markdown(f"""<div class="card">
            <div class="card-label">Yawns</div>
            <div class="card-val {y_cls}">{snap['yawns']}</div>
        </div>""", unsafe_allow_html=True)

    # Browser beep via Web Audio API
    alert_type = snap["alert_type"]
    freq = 1200 if alert_type == "DROWSY" else (850 if alert_type == "YAWN" else 0)
    rpt  = 1500 if alert_type == "DROWSY" else 2500

    components.html(f"""<!DOCTYPE html><html><body style="margin:0">
    <script>
    const FREQ={freq}, RPT={rpt};
    function beep(){{
        if(!FREQ) return;
        const ctx=new(window.AudioContext||window.webkitAudioContext)();
        [0,0.6].forEach(function(t){{
            var o=ctx.createOscillator(),g=ctx.createGain();
            o.connect(g);g.connect(ctx.destination);
            o.type='sine';o.frequency.value=FREQ;
            g.gain.setValueAtTime(1,ctx.currentTime+t);
            g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+t+0.5);
            o.start(ctx.currentTime+t);o.stop(ctx.currentTime+t+0.55);
        }});
    }}
    if(FREQ){{beep();setInterval(beep,RPT);}}
    </script></body></html>""", height=0)

    time.sleep(0.9)
    st.rerun()
