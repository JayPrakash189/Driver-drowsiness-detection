import os, time, urllib.request
import numpy as np
import streamlit as st
import mediapipe as mp
import cv2
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Environment fixes for headless servers
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

st.set_page_config(page_title="Driver Monitor", page_icon="🚗", layout="wide")

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO, # Changed to VIDEO for WebRTC
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(opts)

model = load_model()

# Constants for detection
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.20
JAW_THRESHOLD = 0.35

def calc_ear(lm, idx, W, H):
    p = np.array([[lm[i].x * W, lm[i].y * H] for i in idx])
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2.0 * C + 1e-6)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    H, W = img.shape[:2]
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Using timestamp_ms for video mode
    result = model.detect_for_video(mp_img, int(time.time() * 1000))

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        ear = (calc_ear(lm, LEFT_EYE, W, H) + calc_ear(lm, RIGHT_EYE, W, H)) / 2
        
        jaw = 0.0
        if result.face_blendshapes:
            bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
            jaw = bs.get("jawOpen", 0.0)

        # Draw overlays
        color = (0, 255, 0)
        label = "SAFE"
        if ear < EAR_THRESHOLD:
            color, label = (0, 0, 255), "DROWSY!!"
        elif jaw > JAW_THRESHOLD:
            color, label = (0, 165, 255), "YAWN!!"

        cv2.putText(img, f"{label} EAR: {ear:.2f} JAW: {jaw:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame.from_ndarray(img, format="bgr24")

st.title("🚗 Real-Time Driver Monitor")

# WebRTC configuration with STUN server
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="driver-monitor",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
