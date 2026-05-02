import os, time, urllib.request
import numpy as np
import streamlit as st
import mediapipe as mp
import cv2
from scipy.spatial import distance as dist
from PIL import Image
import streamlit.components.v1 as components

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["LIBGL_ALWAYS_SOFTWARE"]    = "1"
os.environ["QT_QPA_PLATFORM"]          = "offscreen"
os.environ["DISPLAY"]                  = ""

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
    cy = (lm[152].y + lm[10].y)  / 2
    h  = lm[1].x - cx
    v  = lm[1].y - cy
    if abs(h) > 0.15: return "H"
    if v < -0.14:     return "U"
    if v >  0.18:     return "D"
    return None

def draw_pts(img, lm, idx, W, H, color):
    pts = np.array([[int(lm[i].x * W), int(lm[i].y * H)] for i in idx])
    cv2.polylines(img, [pts], True, color, 1)
    for p in pts:
        cv2.circle(img, tuple(p), 2, color, -1)

def analyze_frame(img_bgr, ear_thr, jaw_thr):
    H, W = img_bgr.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = model.detect(mp_img)

    ear = 0.30; jaw = 0.0; face_found = False; distraction = None; alert = ""
    annotated = img_bgr.copy()

    if result.face_landmarks:
        face_found = True
        lm  = result.face_landmarks[0]
        ear = (calc_ear(lm, LEFT_EYE, W, H) + calc_ear(lm, RIGHT_EYE, W, H)) / 2
        if result.face_blendshapes:
            bs  = {c.category_name: c.score for c in result.face_blendshapes[0]}
            jaw = bs.get("jawOpen", 0.0)
        distraction = check_distraction(lm)
        draw_pts(annotated, lm, LEFT_EYE,    W, H, (0, 225, 80))
        draw_pts(annotated, lm, RIGHT_EYE,   W, H, (0, 225, 80))
        draw_pts(annotated, lm, MOUTH_OUTER, W, H, (0, 180, 255))
        draw_pts(annotated, lm, MOUTH_INNER, W, H, (0, 140, 200))

        ec = (50, 50, 255) if ear < ear_thr else (0, 210, 90)
        jc = (50, 50, 255) if jaw > jaw_thr  else (0, 210, 90)
        cv2.rectangle(annotated, (0, 0), (W, 80), (5, 8, 18), -1)
        cv2.putText(annotated, f"EAR: {ear:.3f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ec, 2)
        cv2.putText(annotated, f"JAW: {jaw:.3f}", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, jc, 2)

        if ear < ear_thr:   alert = "DROWSY"
        elif jaw > jaw_thr: alert = "YAWN"
        elif distraction:   alert = "DISTRACTED"

        if alert == "DROWSY":
            cv2.rectangle(annotated, (0, H-60), (W, H), (0, 0, 140), -1)
            cv2.putText(annotated, "DROWSY! WAKE UP!", (8, H-16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 2)
        elif alert == "YAWN":
            cv2.rectangle(annotated, (0, H-60), (W, H), (0, 80, 0), -1)
            cv2.putText(annotated, "YAWN DETECTED!", (8, H-16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 140), 2)
        elif alert == "DISTRACTED":
            cv2.rectangle(annotated, (0, H-60), (W, H), (120, 50, 0), -1)
            cv2.putText(annotated, "DISTRACTED! FOCUS!", (8, H-16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 180, 255), 2)
    else:
        cv2.putText(annotated, "NO FACE DETECTED", (W//2-130, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 220), 2)

    return annotated, ear, jaw, face_found, alert, distraction

# ── SESSION STATE ─────────────────────────────────────────
defaults = {
    "blinks": 0, "yawns": 0, "start_time": time.time(),
    "eye_was_closed": False, "yawn_active": False,
    "last_ear": 0.30, "last_jaw": 0.0, "last_alert": "",
    "last_face": False, "frame_count": 0, "running": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    ear_thr = st.slider("EAR Threshold", 0.10, 0.35, 0.20, 0.01,
                        help="Eyes closed if EAR below this")
    jaw_thr = st.slider("JAW Threshold", 0.10, 0.70, 0.35, 0.01,
                        help="Yawn detected if JAW above this")
    st.markdown("---")
    col_a, col_b = st.columns(2)
    if col_a.button("▶️ Start"):
        st.session_state.running = True
    if col_b.button("⏹ Stop"):
        st.session_state.running = False
    if st.button("🔄 Reset Counters"):
        st.session_state.blinks = 0
        st.session_state.yawns  = 0
        st.session_state.start_time = time.time()
        st.rerun()
    st.markdown("""
**Alert Guide:**
- 😴 Drowsy = EAR low
- 🥱 Yawn = JAW high
- 👀 Distracted = head turned
    """)

# ── MAIN UI ───────────────────────────────────────────────
st.title("🚗 Driver Drowsiness Monitor")
st.markdown("---")

mins  = int((time.time() - st.session_state.start_time) / 60)
alert = st.session_state.last_alert

if alert == "DROWSY":       st.error("😴 DROWSY! Eyes closed — Wake Up!")
elif alert == "YAWN":       st.warning("🥱 YAWN detected — Consider a break.")
elif alert == "DISTRACTED": st.info("👀 DISTRACTED! Focus on the road!")
elif st.session_state.last_face: st.success("✅ Driver is Alert and Focused")
else:                       st.info("📷 Click ▶️ Start in sidebar to begin")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("EAR",     f"{st.session_state.last_ear:.3f}")
c2.metric("JAW",     f"{st.session_state.last_jaw:.3f}")
c3.metric("Blinks",  st.session_state.blinks)
c4.metric("Yawns",   st.session_state.yawns)
c5.metric("Session", f"{mins}m")

if st.session_state.yawns >= 3:
    st.warning("⚠️ HIGH FATIGUE — 3+ yawns! Please take a break.")

st.markdown("---")
col_cam, col_info = st.columns([2, 1])

with col_cam:
    st.markdown("### 📷 Live Camera")
    img_placeholder   = st.empty()
    status_placeholder = st.empty()

    # Camera input — key changes each frame to force a new capture
    cam_key = f"cam_{st.session_state.frame_count}"
    camera_image = st.camera_input("", key=cam_key, label_visibility="collapsed")

    if camera_image is not None and st.session_state.running:
        pil_img = Image.open(camera_image)
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        annotated, ear, jaw, face_found, alert, distraction = analyze_frame(img_bgr, ear_thr, jaw_thr)

        # Update counters
        st.session_state.last_ear   = ear
        st.session_state.last_jaw   = jaw
        st.session_state.last_alert = alert
        st.session_state.last_face  = face_found

        if ear < ear_thr and not st.session_state.eye_was_closed:
            st.session_state.blinks += 1
            st.session_state.eye_was_closed = True
        elif ear >= ear_thr:
            st.session_state.eye_was_closed = False

        if jaw > jaw_thr and not st.session_state.yawn_active:
            st.session_state.yawns += 1
            st.session_state.yawn_active = True
        elif jaw <= jaw_thr:
            st.session_state.yawn_active = False

        # Show annotated frame
        img_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Auto-advance
        st.session_state.frame_count += 1
        time.sleep(0.15)
        st.rerun()

    elif not st.session_state.running:
        status_placeholder.info("⏸ Paused — Click ▶️ Start in sidebar")

with col_info:
    st.markdown("### 📊 Status")
    st.markdown(f"""
| | |
|---|---|
| **Face** | {"✅ Yes" if st.session_state.last_face else "❌ No"} |
| **EAR** | {st.session_state.last_ear:.3f} |
| **JAW** | {st.session_state.last_jaw:.3f} |
| **Blinks** | {st.session_state.blinks} |
| **Yawns** | {st.session_state.yawns} |
| **Session** | {mins} min |
| **Alert** | {st.session_state.last_alert or "✅ None"} |
    """)
    st.markdown("---")
    st.markdown("""
**Thresholds:**
- Normal EAR: 0.28–0.35
- Normal JAW: 0.00–0.15
- Yawn JAW: 0.40–0.70
    """)

# Alert beep
freq = 0
if alert == "DROWSY":       freq = 1200
elif alert == "YAWN":       freq = 850
elif alert == "DISTRACTED": freq = 1000

components.html(f"""<script>
const F={freq};
if(F){{try{{
  const ctx=new(window.AudioContext||window.webkitAudioContext)();
  [0,0.55].forEach(t=>{{
    const o=ctx.createOscillator(),g=ctx.createGain();
    o.connect(g);g.connect(ctx.destination);
    o.type='sine';o.frequency.value=F;
    g.gain.setValueAtTime(0.8,ctx.currentTime+t);
    g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+t+0.5);
    o.start(ctx.currentTime+t);o.stop(ctx.currentTime+t+0.55);
  }});
}}catch(e){{}}}}
</script>""", height=0)
