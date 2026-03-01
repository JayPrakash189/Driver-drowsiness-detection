import os, time, threading, urllib.request
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import av
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit.components.v1 as components

st.set_page_config(page_title="Driver Monitor", page_icon="🚗", layout="wide")

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
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

model = load_model()

class State:
    def __init__(self):
        self.lock        = threading.Lock()
        self.ear         = 0.30
        self.jaw         = 0.0
        self.eye_state   = "OPEN"
        self.secs_closed = 0.0
        self.close_time  = 0.0
        self.blinks      = 0
        self.yawns       = 0
        self.yawn_frames = 0
        self.in_yawn     = False
        self.distraction = None
        self.dist_frames = 0
        self.alert       = ""
        self.face_found  = False
        self.ear_hist    = []
        self.jaw_hist    = []
        self.start_time  = time.time()

    def snapshot(self):
        with self.lock:
            return {
                "ear":         self.ear,
                "jaw":         self.jaw,
                "eye_state":   self.eye_state,
                "secs_closed": self.secs_closed,
                "blinks":      self.blinks,
                "yawns":       self.yawns,
                "alert":       self.alert,
                "face_found":  self.face_found,
                "mins":        int((time.time() - self.start_time) / 60),
            }

    def update(self, ear_raw, jaw_raw, face_ok, ear_thr, jaw_thr, drowsy_sec, distraction):
        now = time.time()
        with self.lock:
            self.face_found = face_ok
            if not face_ok:
                self.eye_state = "OPEN"
                self.secs_closed = 0.0
                self.alert = ""
                self.dist_frames = 0
                return
            self.ear_hist.append(ear_raw)
            self.jaw_hist.append(jaw_raw)
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            if len(self.jaw_hist) > 5: self.jaw_hist.pop(0)
            self.ear = float(np.mean(self.ear_hist))
            self.jaw = float(np.mean(self.jaw_hist))
            if self.eye_state == "OPEN":
                if ear_raw < ear_thr:
                    self.eye_state = "CLOSED"
                    self.close_time = now
            else:
                self.secs_closed = now - self.close_time
                if ear_raw >= ear_thr:
                    dur = self.secs_closed
                    self.eye_state = "OPEN"
                    self.secs_closed = 0.0
                    if 0.05 < dur < 0.5:
                        self.blinks += 1
            self.distraction = distraction
            if distraction:
                self.dist_frames = min(self.dist_frames + 1, 60)
            else:
                self.dist_frames = max(self.dist_frames - 1, 0)
            if self.jaw > jaw_thr:
                self.yawn_frames = min(self.yawn_frames + 1, 60)
            else:
                self.yawn_frames = max(self.yawn_frames - 1, 0)
            if self.yawn_frames >= 20:
                if not self.in_yawn:
                    self.yawns += 1
                    self.in_yawn = True
            else:
                self.in_yawn = False
            if self.eye_state == "CLOSED" and self.secs_closed >= drowsy_sec:
                self.alert = "DROWSY"
            elif self.yawn_frames >= 20:
                self.alert = "YAWN"
            elif distraction and self.dist_frames >= 15:
                self.alert = "DISTRACTED"
            else:
                self.alert = ""

    def reset(self):
        with self.lock:
            self.blinks = 0
            self.yawns = 0
            self.start_time = time.time()

if "state" not in st.session_state:
    st.session_state.state = State()
state = st.session_state.state

LEFT_EYE    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE   = [362, 385, 387, 263, 373, 380]
MOUTH_OUTER = [61, 40, 37, 0, 267, 270, 291, 321, 375, 321, 405, 314, 17, 84, 181, 91, 61]
MOUTH_INNER = [78, 82, 87, 13, 317, 312, 308, 402, 317, 14, 87]

def calc_ear(lm, idx, W, H):
    p = np.array([[lm[i].x * W, lm[i].y * H] for i in idx])
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2.0 * C + 1e-6)

def check_distraction(lm):
    cx = (lm[133].x + lm[362].x) / 2
    cy = (lm[152].y + lm[10].y) / 2
    h  = lm[1].x - cx
    v  = lm[1].y - cy
    if abs(h) > 0.07: return "H"
    if v < -0.07:     return "U"
    if v >  0.07:     return "D"
    return None

def draw_pts(img, lm, idx, W, H, color):
    pts = np.array([[int(lm[i].x * W), int(lm[i].y * H)] for i in idx])
    cv2.polylines(img, [pts], True, color, 1)
    for p in pts:
        cv2.circle(img, tuple(p), 2, color, -1)

def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    H, W = img.shape[:2]

    ear_thr    = getattr(st.session_state, "_ear_thr",    0.20)
    jaw_thr    = getattr(st.session_state, "_jaw_thr",    0.30)
    drowsy_sec = getattr(st.session_state, "_drowsy_sec", 1.5)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = model.detect(mp_img)

    ear_raw = 0.30
    jaw_raw = 0.0
    face_ok = False
    distraction = None

    if result.face_landmarks:
        face_ok = True
        lm = result.face_landmarks[0]
        ear_raw = (calc_ear(lm, LEFT_EYE, W, H) + calc_ear(lm, RIGHT_EYE, W, H)) / 2
        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            jaw_raw = bs.get("jawOpen", 0.0)
        distraction = check_distraction(lm)
        draw_pts(img, lm, LEFT_EYE,    W, H, (0, 225, 80))
        draw_pts(img, lm, RIGHT_EYE,   W, H, (0, 225, 80))
        draw_pts(img, lm, MOUTH_OUTER, W, H, (0, 180, 255))
        draw_pts(img, lm, MOUTH_INNER, W, H, (0, 140, 200))

    state.update(ear_raw, jaw_raw, face_ok, ear_thr, jaw_thr, drowsy_sec, distraction)
    snap = state.snapshot()

    cv2.rectangle(img, (0, 0), (W, 85), (5, 8, 18), -1)
    ec = (50, 50, 255) if snap["eye_state"] == "CLOSED" else (0, 210, 90)
    jc = (50, 50, 255) if snap["jaw"] > jaw_thr else (0, 210, 90)

    cv2.rectangle(img, (120, 8),  (300, 24), (20, 30, 55), -1)
    cv2.rectangle(img, (120, 8),  (120 + int(min(snap["ear"] / 0.45, 1) * 180), 24), ec, -1)
    cv2.putText(img, f"EAR {snap['ear']:.3f}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ec, 1)

    cv2.rectangle(img, (120, 32), (300, 48), (20, 30, 55), -1)
    cv2.rectangle(img, (120, 32), (120 + int(min(snap["jaw"] / 0.8, 1) * 180), 48), jc, -1)
    cv2.putText(img, f"JAW {snap['jaw']:.3f}", (5, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, jc, 1)

    eye_txt = f"Eye: {snap['eye_state']}"
    if snap["eye_state"] == "CLOSED":
        eye_txt += f"  {snap['secs_closed']:.1f}s"
    cv2.putText(img, eye_txt, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ec, 1)
    cv2.putText(img, f"Blinks:{snap['blinks']}  Yawns:{snap['yawns']}",
                (W - 185, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 140, 180), 1)

    if snap["alert"] == "DROWSY":
        cv2.rectangle(img, (0, H - 60), (W, H), (0, 0, 140), -1)
        cv2.putText(img, "  DROWSY! WAKE UP!", (8, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 2)
    elif snap["alert"] == "YAWN":
        cv2.rectangle(img, (0, H - 60), (W, H), (0, 80, 0), -1)
        cv2.putText(img, "  YAWN DETECTED!", (8, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 140), 2)
    elif snap["alert"] == "DISTRACTED":
        cv2.rectangle(img, (0, H - 60), (W, H), (120, 50, 0), -1)
        cv2.putText(img, "  DISTRACTED! FOCUS!", (8, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 180, 255), 2)

    if not face_ok:
        cv2.putText(img, "NO FACE DETECTED", (W // 2 - 130, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 220), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Settings")
    ear_thr    = st.slider("EAR Threshold",      0.10, 0.35, 0.20, 0.01)
    jaw_thr    = st.slider("JAW Threshold",      0.10, 0.70, 0.30, 0.01)
    drowsy_sec = st.slider("Drowsy Delay (sec)", 0.5,  4.0,  1.5,  0.25)
    st.session_state._ear_thr    = ear_thr
    st.session_state._jaw_thr    = jaw_thr
    st.session_state._drowsy_sec = drowsy_sec
    st.markdown("---")
    st.markdown("""
**Tuning Guide:**
- Open eyes EAR ≈ 0.28–0.35
- Closed eyes EAR ≈ 0.10–0.18
- Set threshold between them
- Yawn JAW score rises above 0.4
    """)
    if st.button("🔄 Reset Counters"):
        state.reset()
        st.rerun()

# MAIN UI
snap = state.snapshot()

st.title("🚗 Driver Drowsiness Monitor")
st.markdown("---")

# Alert box
if snap["alert"] == "DROWSY":
    st.error("😴 DROWSY! Eyes closed too long — Wake Up!")
elif snap["alert"] == "YAWN":
    st.warning("🥱 YAWN detected — You may be tired, consider a break.")
elif snap["alert"] == "DISTRACTED":
    st.info("👀 DISTRACTED! Look at the road!")
else:
    st.success("✅ Driver is Alert and Focused")

# Stats row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👁 EAR",     f"{snap['ear']:.3f}")
c2.metric("😮 JAW",     f"{snap['jaw']:.3f}")
c3.metric("👁 Blinks",  snap["blinks"])
c4.metric("🥱 Yawns",   snap["yawns"])
c5.metric("⏱ Session", f"{snap['mins']}m")

if snap["yawns"] >= 3:
    st.warning("⚠️ HIGH FATIGUE — 3+ yawns! Please take a break.")

st.markdown("---")

# Video + info
col_video, col_info = st.columns([2.2, 1])

with col_video:
    st.markdown("📷 **Live Camera Feed**")
    webrtc_streamer(
        key="driver-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_callback,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

with col_info:
    face_status = "✅ Face Detected" if snap["face_found"] else "❌ No Face"
    eye_status  = f"🔴 CLOSED {snap['secs_closed']:.1f}s" if snap["eye_state"] == "CLOSED" else "🟢 OPEN"
    st.markdown(f"""
**Face:** {face_status}

**Eye State:** {eye_status}

**Blinks:** {snap['blinks']}

**Yawns:** {snap['yawns']}

**Session:** {snap['mins']} minutes
    """)
    st.markdown("---")
    st.markdown("""
**Alert Guide:**
- 😴 Drowsy = eyes closed too long
- 🥱 Yawn = mouth open > 20 frames
- 👀 Distracted = head turned away
- ⚠️ 3+ yawns = take a break!
    """)

# BEEP
freq, rpt = 0, 0
if snap["alert"] == "DROWSY":       freq, rpt = 1200, 1500
elif snap["alert"] == "YAWN":       freq, rpt = 850,  2500
elif snap["alert"] == "DISTRACTED": freq, rpt = 1000, 2000

components.html(f"""<!DOCTYPE html><html><body style="margin:0">
<script>
const F = {freq}, R = {rpt};
function beep() {{
    if (!F) return;
    try {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        [0, 0.55].forEach(t => {{
            const o = ctx.createOscillator(), g = ctx.createGain();
            o.connect(g); g.connect(ctx.destination);
            o.type = 'sine'; o.frequency.value = F;
            g.gain.setValueAtTime(0.8, ctx.currentTime + t);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + t + 0.5);
            o.start(ctx.currentTime + t);
            o.stop(ctx.currentTime + t + 0.55);
        }});
    }} catch(e) {{}}
}}
if (F) {{ beep(); setInterval(beep, R); }}
</script></body></html>""", height=0)

time.sleep(0.9)
st.rerun()
