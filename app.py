import os, time, threading, urllib.request
import numpy as np
import streamlit as st
import mediapipe as mp
import cv2
from scipy.spatial import distance as dist
import streamlit.components.v1 as components
from PIL import Image

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

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
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
    if abs(h) > 0.15: return "Looking Away (H)"
    if v < -0.14:     return "Looking Up"
    if v >  0.18:     return "Looking Down"
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

    ear = 0.30
    jaw = 0.0
    face_found = False
    distraction = None
    alert = ""
    annotated = img_bgr.copy()

    if result.face_landmarks:
        face_found = True
        lm = result.face_landmarks[0]
        ear = (calc_ear(lm, LEFT_EYE, W, H) + calc_ear(lm, RIGHT_EYE, W, H)) / 2
        if result.face_blendshapes:
            bs  = {c.category_name: c.score for c in result.face_blendshapes[0]}
            jaw = bs.get("jawOpen", 0.0)
        distraction = check_distraction(lm)
        draw_pts(annotated, lm, LEFT_EYE,    W, H, (0, 225, 80))
        draw_pts(annotated, lm, RIGHT_EYE,   W, H, (0, 225, 80))
        draw_pts(annotated, lm, MOUTH_OUTER, W, H, (0, 180, 255))
        draw_pts(annotated, lm, MOUTH_INNER, W, H, (0, 140, 200))

        # Determine alert
        if ear < ear_thr:
            alert = "DROWSY"
        elif jaw > jaw_thr:
            alert = "YAWN"
        elif distraction:
            alert = "DISTRACTED"

        # Draw HUD
        ec = (50, 50, 255) if ear < ear_thr else (0, 210, 90)
        jc = (50, 50, 255) if jaw > jaw_thr  else (0, 210, 90)
        cv2.rectangle(annotated, (0, 0), (W, 80), (5, 8, 18), -1)
        cv2.putText(annotated, f"EAR: {ear:.3f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ec, 2)
        cv2.putText(annotated, f"JAW: {jaw:.3f}", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, jc, 2)

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
if "blinks"     not in st.session_state: st.session_state.blinks     = 0
if "yawns"      not in st.session_state: st.session_state.yawns      = 0
if "start_time" not in st.session_state: st.session_state.start_time = time.time()
if "prev_ear"   not in st.session_state: st.session_state.prev_ear   = 0.30
if "eye_was_closed" not in st.session_state: st.session_state.eye_was_closed = False
if "yawn_active"    not in st.session_state: st.session_state.yawn_active    = False

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    ear_thr    = st.slider("EAR Threshold",      0.10, 0.35, 0.20, 0.01,
                           help="Eyes CLOSED if EAR drops below this")
    jaw_thr    = st.slider("JAW Threshold",      0.10, 0.70, 0.35, 0.01,
                           help="Yawn if jawOpen rises above this")
    st.markdown("---")
    st.markdown("""
**How to use:**
1. Allow camera access
2. Click **📸 Capture & Analyze**
3. Results appear instantly

**Alert Guide:**
- 😴 Drowsy = EAR below threshold
- 🥱 Yawn = JAW above threshold  
- 👀 Distracted = head turned
    """)
    if st.button("🔄 Reset Counters"):
        st.session_state.blinks     = 0
        st.session_state.yawns      = 0
        st.session_state.start_time = time.time()
        st.rerun()

# ── MAIN UI ───────────────────────────────────────────────
st.title("🚗 Driver Drowsiness Monitor")
st.markdown("---")

mins = int((time.time() - st.session_state.start_time) / 60)

# Metrics row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("EAR",     f"{st.session_state.get('last_ear', 0.30):.3f}")
c2.metric("JAW",     f"{st.session_state.get('last_jaw', 0.00):.3f}")
c3.metric("Blinks",  st.session_state.blinks)
c4.metric("Yawns",   st.session_state.yawns)
c5.metric("Session", f"{mins}m")

if st.session_state.yawns >= 3:
    st.warning("⚠️ HIGH FATIGUE — 3+ yawns detected! Please take a break.")

st.markdown("---")

col_cam, col_info = st.columns([2, 1])

with col_cam:
    st.markdown("### 📷 Camera Feed")
    
    # Camera input — works natively on Streamlit Cloud
    camera_image = st.camera_input("", key="camera", label_visibility="collapsed")

    if camera_image is not None:
        # Convert to OpenCV format
        pil_img = Image.open(camera_image)
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Analyze
        annotated, ear, jaw, face_found, alert, distraction = analyze_frame(img_bgr, ear_thr, jaw_thr)

        # Save metrics
        st.session_state.last_ear = ear
        st.session_state.last_jaw = jaw

        # Blink detection (eye was open, now closed)
        if ear < ear_thr and not st.session_state.eye_was_closed:
            st.session_state.blinks += 1
            st.session_state.eye_was_closed = True
        elif ear >= ear_thr:
            st.session_state.eye_was_closed = False

        # Yawn detection
        if jaw > jaw_thr and not st.session_state.yawn_active:
            st.session_state.yawns += 1
            st.session_state.yawn_active = True
        elif jaw <= jaw_thr:
            st.session_state.yawn_active = False

        # Show annotated image
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

        # Alert banner
        if alert == "DROWSY":
            st.error("😴 DROWSY! Eyes closed — Wake Up!")
        elif alert == "YAWN":
            st.warning("🥱 YAWN detected — Consider taking a break.")
        elif alert == "DISTRACTED":
            st.info(f"👀 DISTRACTED! {distraction} — Focus on the road!")
        elif face_found:
            st.success("✅ Driver is Alert and Focused")
        else:
            st.warning("🔍 No face detected — Position your face in frame")

with col_info:
    st.markdown("### 📊 Status")
    last_alert = st.session_state.get("last_alert", "")
    face_found = st.session_state.get("last_face", False)

    if camera_image:
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| **Face** | {"✅ Detected" if face_found else "❌ Not found"} |
| **EAR** | {st.session_state.get('last_ear', 0.30):.3f} |
| **JAW** | {st.session_state.get('last_jaw', 0.00):.3f} |
| **Blinks** | {st.session_state.blinks} |
| **Yawns** | {st.session_state.yawns} |
| **Session** | {mins} min |
        """)
    else:
        st.info("👆 Take a photo to start analysis")

    st.markdown("---")
    st.markdown("""
**Thresholds:**
- EAR < threshold → Eyes closed
- JAW > threshold → Mouth open (yawn)
- Normal EAR: 0.28–0.35
- Normal JAW: 0.00–0.15
    """)

# Alert beep
alert_val = st.session_state.get("last_alert_val", "")
freq, rpt = 0, 0
if alert_val == "DROWSY":       freq, rpt = 1200, 1500
elif alert_val == "YAWN":       freq, rpt = 850,  2500
elif alert_val == "DISTRACTED": freq, rpt = 1000, 2000

if camera_image and 'alert' in dir():
    st.session_state.last_alert_val = alert
    st.session_state.last_face = face_found if 'face_found' in dir() else False

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
if (F) {{ beep(); }}
</script></body></html>""", height=0)
