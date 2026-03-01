"""
🚗 Driver Drowsiness Detection — FINAL STABLE VERSION
"""

import os, time, threading, urllib.request
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

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Driver Drowsiness Detection")
st.markdown("### Real-Time Fatigue Monitoring")

# ─────────────────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

@st.cache_resource
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
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(opts)

landmarker = load_landmarker()

# ─────────────────────────────────────────────────────────
# THREAD SAFE STATE
# ─────────────────────────────────────────────────────────
class DetectionState:
    def __init__(self):
        self._lock = threading.Lock()
        self.ear = 0.3
        self.jaw = 0.0
        self.eye_state = "OPEN"
        self.secs_closed = 0.0
        self.close_time = 0
        self.blinks = 0
        self.yawns = 0
        self.yawn_frames = 0
        self.alert = ""
        self.face_found = False

    def get(self):
        with self._lock:
            return dict(
                ear=self.ear,
                jaw=self.jaw,
                eye_state=self.eye_state,
                secs_closed=self.secs_closed,
                blinks=self.blinks,
                yawns=self.yawns,
                alert=self.alert,
                face_found=self.face_found,
            )

    def update(self, ear_raw, jaw_raw, face_ok, ear_thr, jaw_thr, drowsy_sec):
        now = time.time()
        with self._lock:
            self.face_found = face_ok

            if not face_ok:
                self.eye_state = "OPEN"
                self.secs_closed = 0
                self.alert = ""
                return

            self.ear = ear_raw
            self.jaw = jaw_raw

            if self.eye_state == "OPEN":
                if ear_raw < ear_thr:
                    self.eye_state = "CLOSED"
                    self.close_time = now
            else:
                self.secs_closed = now - self.close_time
                if ear_raw >= ear_thr:
                    if 0.05 < self.secs_closed < 0.5:
                        self.blinks += 1
                    self.eye_state = "OPEN"
                    self.secs_closed = 0

            if self.jaw > jaw_thr:
                self.yawn_frames += 1
                if self.yawn_frames == 15:
                    self.yawns += 1
            else:
                self.yawn_frames = 0

            if self.eye_state == "CLOSED" and self.secs_closed > drowsy_sec:
                self.alert = "DROWSY"
            elif self.jaw > jaw_thr:
                self.alert = "YAWN"
            else:
                self.alert = ""

if "det" not in st.session_state:
    st.session_state.det = DetectionState()

det = st.session_state.det

# ─────────────────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────────────────
LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def calc_ear(lm, idx, W, H):
    p = np.array([[lm[i].x*W, lm[i].y*H] for i in idx])
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A+B)/(2*C+1e-6)

# ─────────────────────────────────────────────────────────
# VIDEO CALLBACK
# ─────────────────────────────────────────────────────────
def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    H, W = img.shape[:2]

    ear_thr = st.session_state.get("_ear_thr", 0.2)
    jaw_thr = st.session_state.get("_jaw_thr", 0.3)
    drowsy_sec = st.session_state.get("_drowsy_sec", 1.5)

    ear_raw = 0.3
    jaw_raw = 0.0
    face_ok = False

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    result = landmarker.detect(mp_img)

    if result.face_landmarks:
        face_ok = True
        lm = result.face_landmarks[0]
        le = calc_ear(lm, LEFT_EYE, W, H)
        re = calc_ear(lm, RIGHT_EYE, W, H)
        ear_raw = (le + re) / 2.0

        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            jaw_raw = bs.get("jawOpen", 0.0)

    det.update(ear_raw, jaw_raw, face_ok, ear_thr, jaw_thr, drowsy_sec)
    snap = det.get()

    if snap["alert"] == "DROWSY":
        cv2.putText(img, "DROWSY! WAKE UP!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────
# SIDEBAR SETTINGS
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.session_state._ear_thr = st.slider("EAR Threshold", 0.1, 0.35, 0.2, 0.01)
    st.session_state._jaw_thr = st.slider("Jaw Threshold", 0.1, 0.7, 0.3, 0.01)
    st.session_state._drowsy_sec = st.slider("Drowsy Delay", 0.5, 4.0, 1.5, 0.25)

# ─────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────
col1, col2 = st.columns([2,1])

with col1:
    webrtc_streamer(
        key="driver-drowsy",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_callback,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True,
    )

with col2:
    snap = det.get()
    st.metric("EAR", round(snap["ear"],3))
    st.metric("Jaw", round(snap["jaw"],3))
    st.metric("Eye State", snap["eye_state"])
    st.metric("Blinks", snap["blinks"])
    st.metric("Yawns", snap["yawns"])
    st.metric("Alert", snap["alert"])
