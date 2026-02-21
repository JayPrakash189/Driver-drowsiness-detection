"""
🚗 Driver Drowsiness Detection  v4.0
=====================================
Works with MediaPipe 0.10+ (Tasks API)
Auto-downloads the required model file on first run.

INSTALL:
    pip install opencv-python mediapipe numpy scipy sounddevice

RUN:
    python drowsiness_detection.py
"""

import cv2
import numpy as np
import sys
import os
import time
import threading
from collections import deque
from scipy.spatial import distance as dist

# ══════════════════════════════════════════════════════════
#  STEP 1 — Download model file if not present
# ══════════════════════════════════════════════════════════
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading MediaPipe face model (~6 MB) — first run only …")
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded.\n")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"   Download manually from:\n   {MODEL_URL}")
        print(f"   Save as: {MODEL_PATH}  (same folder as this script)")
        sys.exit(1)

# ══════════════════════════════════════════════════════════
#  STEP 2 — Load MediaPipe FaceLandmarker (Tasks API 0.10+)
# ══════════════════════════════════════════════════════════
try:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode
    )

    options = FaceLandmarkerOptions(
        base_options              = BaseOptions(model_asset_path=MODEL_PATH),
        running_mode              = RunningMode.IMAGE,
        num_faces                 = 1,
        min_face_detection_confidence = 0.5,
        min_face_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
        output_face_blendshapes       = True,   # needed for yawn via blendshape
    )
    landmarker = FaceLandmarker.create_from_options(options)
    print("✅ MediaPipe FaceLandmarker ready (Tasks API)")

except Exception as e:
    print(f"❌ MediaPipe init failed: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  STEP 3 — Sound  (tries sounddevice → winsound → bell)
# ══════════════════════════════════════════════════════════
_snd_method = None

def _init_sound():
    global _snd_method
    try:
        import sounddevice as sd
        t   = np.linspace(0, 0.1, 4410, False)
        sd.play((np.sin(2*np.pi*440*t)*0.3).astype(np.float32), 44100, blocking=True)
        _snd_method = "sounddevice"
        return
    except Exception:
        pass
    try:
        import winsound
        winsound.Beep(440, 100)
        _snd_method = "winsound"
        return
    except Exception:
        pass
    try:
        import pygame
        pygame.mixer.init(44100, -16, 1, 512)
        _snd_method = "pygame"
        return
    except Exception:
        pass
    _snd_method = "bell"

def _play(freq, dur):
    try:
        if _snd_method == "sounddevice":
            import sounddevice as sd
            t   = np.linspace(0, dur, int(44100*dur), False)
            wav = (np.sin(2*np.pi*freq*t) * 0.85).astype(np.float32)
            sd.play(wav, 44100, blocking=True)
        elif _snd_method == "winsound":
            import winsound
            winsound.Beep(int(freq), int(dur*1000))
        elif _snd_method == "pygame":
            import pygame
            t   = np.linspace(0, dur, int(44100*dur), False)
            wav = (np.sin(2*np.pi*freq*t) * 32767).astype(np.int16)
            pygame.sndarray.make_sound(wav).play()
            time.sleep(dur)
        else:
            for _ in range(3):
                sys.stdout.write('\a'); sys.stdout.flush(); time.sleep(0.15)
    except Exception:
        pass

def beep(freq=1100, dur=0.6):
    threading.Thread(target=_play, args=(freq, dur), daemon=True).start()

# ══════════════════════════════════════════════════════════
#  STEP 4 — Landmark indices & ratio helpers
#  MediaPipe 478-point mesh (with irises)
#  Standard EAR indices:
#    p0=outer-corner  p1=top-outer  p2=top-inner
#    p3=inner-corner  p4=bot-inner  p5=bot-outer
# ══════════════════════════════════════════════════════════
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_8   = [61,  40,  37,   0, 267, 270, 291, 321]

# Blendshape indices for eye openness and jaw open
# (more reliable than geometric EAR for blink detection)
BS_EYE_BLINK_LEFT  = 9    # eyeBlinkLeft
BS_EYE_BLINK_RIGHT = 10   # eyeBlinkRight
BS_JAW_OPEN        = 25   # jawOpen

def get_blendshapes(result):
    """Return dict name→score from blendshape result."""
    if not result.face_blendshapes:
        return {}
    return {c.category_name: c.score for c in result.face_blendshapes[0]}

def calc_ear(lm, idx, W, H):
    p = np.array([[lm[i].x*W, lm[i].y*H] for i in idx])
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2.0*C + 1e-6)

def calc_mar(lm, idx, W, H):
    p = np.array([[lm[i].x*W, lm[i].y*H] for i in idx])
    A = dist.euclidean(p[1], p[7])
    B = dist.euclidean(p[2], p[6])
    C = dist.euclidean(p[3], p[5])
    D = dist.euclidean(p[0], p[4])
    return (A+B+C) / (3.0*D + 1e-6)

def draw_region(frame, lm, idx, W, H, color):
    pts = np.array([[int(lm[i].x*W), int(lm[i].y*H)] for i in idx])
    cv2.polylines(frame, [pts], True, color, 1)
    for p in pts: cv2.circle(frame, tuple(p), 2, color, -1)

# ══════════════════════════════════════════════════════════
#  STEP 5 — Calibration
# ══════════════════════════════════════════════════════════
def calibrate(cap):
    print("\n📷 CALIBRATION: Eyes OPEN, mouth CLOSED for 4 s …")
    ear_samples, jaw_samples = [], []
    t0 = time.time()

    while time.time() - t0 < 4.0:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        left  = int(4.0 - (time.time()-t0)) + 1

        dark = frame.copy(); dark[:]=0
        cv2.addWeighted(dark, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "CALIBRATION", (W//2-130, H//2-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,220,255), 3)
        cv2.putText(frame, "Eyes OPEN  Mouth CLOSED", (W//2-165, H//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(frame, f"{left}s", (W//2-20, H//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,100), 3)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_img)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            le = calc_ear(lm, LEFT_EYE,  W, H)
            re = calc_ear(lm, RIGHT_EYE, W, H)
            ear_samples.append((le+re)/2.0)
            bs = get_blendshapes(result)
            jaw_samples.append(bs.get("jawOpen", 0.0))

        cv2.imshow("Driver Drowsiness Detection", frame)
        cv2.waitKey(1)

    if len(ear_samples) < 5:
        print("⚠️  Not enough data — using defaults")
        return 0.20, 0.25, 0.30   # ear_thresh, blink_thresh, jaw_thresh

    avg_ear = float(np.percentile(ear_samples, 20))
    avg_jaw = float(np.mean(jaw_samples))

    ear_thr   = round(max(0.13, min(avg_ear * 0.72, 0.28)), 3)
    blink_thr = round(max(0.20, min(avg_ear * 0.55, 0.24)), 3)
    jaw_thr   = round(max(0.20, min(avg_jaw * 3.5,  0.55)), 3)

    print(f"✅ EAR open={np.mean(ear_samples):.3f}  → close_thr={ear_thr}  blink_thr={blink_thr}")
    print(f"✅ JAW rest={avg_jaw:.3f}  → yawn_thr={jaw_thr}\n")
    return ear_thr, blink_thr, jaw_thr

# ══════════════════════════════════════════════════════════
#  STEP 6 — Blink state machine
# ══════════════════════════════════════════════════════════
class BlinkDetector:
    def __init__(self, close_thr):
        self.close_thr  = close_thr
        self.state      = "OPEN"
        self.close_time = 0.0
        self.count      = 0

    def update(self, ear, now):
        """Returns (just_blinked, state, secs_closed)"""
        just_blinked = False
        secs_closed  = 0.0

        if self.state == "OPEN":
            if ear < self.close_thr:
                self.state      = "CLOSED"
                self.close_time = now
        else:  # CLOSED
            secs_closed = now - self.close_time
            if ear >= self.close_thr:
                self.state = "OPEN"
                if secs_closed < 0.5:      # short → blink
                    self.count  += 1
                    just_blinked = True
                # long close → drowsiness handled in main loop

        return just_blinked, self.state, secs_closed

# ══════════════════════════════════════════════════════════
#  STEP 7 — HUD
# ══════════════════════════════════════════════════════════
def draw_hud(frame, ear, jaw, ear_thr, jaw_thr,
             eye_state, secs_closed, blinks, yawn_f, alert):
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (W,128), (12,12,12), -1)
    cv2.rectangle(frame, (0,0), (W,128), (45,45,45),  1)

    # EAR bar
    ec = (0,55,255) if eye_state=="CLOSED" else (0,200,80)
    bw = int(min(ear/0.45, 1.0) * 210)
    cv2.rectangle(frame, (140,10), (350,32), (50,50,50), -1)
    cv2.rectangle(frame, (140,10), (140+bw, 32), ec, -1)
    cv2.putText(frame, f"EAR  {ear:.3f}", (8,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ec, 2)
    cv2.putText(frame, f"<{ear_thr}", (355,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110,110,110), 1)

    # JAW bar
    jc = (0,55,255) if jaw > jaw_thr else (0,200,80)
    jw = int(min(jaw/0.8, 1.0) * 210)
    cv2.rectangle(frame, (140,40), (350,62), (50,50,50), -1)
    cv2.rectangle(frame, (140,40), (140+jw, 62), jc, -1)
    cv2.putText(frame, f"JAW  {jaw:.3f}", (8,58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, jc, 2)
    cv2.putText(frame, f">{jaw_thr}", (355,58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110,110,110), 1)

    # Eye state
    eye_lbl = f"Eye: {eye_state}"
    if eye_state == "CLOSED": eye_lbl += f"  {secs_closed:.1f}s"
    cv2.putText(frame, eye_lbl, (8,88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,55,255) if eye_state=="CLOSED" else (0,200,80), 2)

    cv2.putText(frame, f"Blinks: {blinks}   Yawn-score: {yawn_f}/15", (8,114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (155,155,155), 1)

    if alert:
        cv2.rectangle(frame, (0,H-70), (W,H), (0,0,175), -1)
        cv2.putText(frame, alert, (12,H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    else:
        cv2.putText(frame, "Q = quit", (W-85,H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,80,80), 1)

# ══════════════════════════════════════════════════════════
#  STEP 8 — MAIN
# ══════════════════════════════════════════════════════════
import mediapipe as mp   # needed for mp.Image

def main():
    _init_sound()
    print(f"🔊 Sound: {_snd_method}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ear_thr, blink_thr, jaw_thr = calibrate(cap)

    blink_det = BlinkDetector(blink_thr)
    ear_buf   = deque(maxlen=3)
    jaw_buf   = deque(maxlen=5)
    yawn_f    = 0
    last_beep = 0.0
    alert     = ""
    alert_end = 0.0
    DROWSY_SEC = 1.5

    print("🎥 Running … press Q to quit\n")
    beep(800, 0.25)   # startup ping

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        now   = time.time()

        ear_raw = 0.30
        jaw_raw = 0.0
        face_ok = False

        # ── Detect ────────────────────────────────────────
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_img)

        if result.face_landmarks:
            face_ok = True
            lm = result.face_landmarks[0]
            le  = calc_ear(lm, LEFT_EYE,  W, H)
            re  = calc_ear(lm, RIGHT_EYE, W, H)
            ear_raw = (le + re) / 2.0

            bs      = get_blendshapes(result)
            jaw_raw = bs.get("jawOpen", 0.0)

            draw_region(frame, lm, LEFT_EYE,  W, H, (0,225,80))
            draw_region(frame, lm, RIGHT_EYE, W, H, (0,225,80))
            draw_region(frame, lm, MOUTH_8,   W, H, (0,180,255))

        # ── Smooth ────────────────────────────────────────
        ear_buf.append(ear_raw)
        jaw_buf.append(jaw_raw)
        ear_s = float(np.mean(ear_buf))
        jaw_s = float(np.mean(jaw_buf))

        # ── Blink (use RAW ear — don't smooth it) ─────────
        blinked, eye_state, secs_closed = False, "OPEN", 0.0
        if face_ok:
            blinked, eye_state, secs_closed = blink_det.update(ear_raw, now)
            if blinked:
                print(f"  👁  Blink #{blink_det.count}  (EAR={ear_raw:.3f})")

        # ── Yawn (blendshape jawOpen) ─────────────────────
        if face_ok and jaw_s > jaw_thr:
            yawn_f = min(yawn_f + 1, 30)
        else:
            yawn_f = max(yawn_f - 1, 0)

        # ── Alerts ────────────────────────────────────────
        if face_ok and eye_state == "CLOSED" and secs_closed >= DROWSY_SEC:
            alert     = "  DROWSY!  Wake Up!"
            alert_end = now + 2.0
            if now - last_beep > 1.2:
                beep(1200, 0.8); last_beep = now

        elif face_ok and yawn_f >= 15:
            alert     = "  YAWN!  Stay Alert!"
            alert_end = now + 2.0
            if now - last_beep > 3.0:
                beep(800, 0.5); last_beep = now

        if now > alert_end:
            alert = ""

        if not face_ok:
            blink_det.state = "OPEN"
            cv2.putText(frame, "No face — look at camera",
                        (W//2-165, H//2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,40,255), 2)

        # ── HUD ───────────────────────────────────────────
        draw_hud(frame, ear_s, jaw_s, ear_thr, jaw_thr,
                 eye_state, secs_closed, blink_det.count, yawn_f, alert)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\n👋 Session ended — Total blinks: {blink_det.count}")

if __name__ == "__main__":
    main()