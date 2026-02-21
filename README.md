## Real-Time Driver Drowsiness Detection System

AI-based driver monitoring system using MediaPipe face landmarks, EAR,JAW analysis and Streamlit interface to detect fatigue and trigger real-time alerts.

## Overview

Driver drowsiness is a major cause of road accidents.
This project uses real-time webcam input to monitor eye closure and yawning patterns and alerts the driver using an audio warning.

## Features

* Real-time face landmark detection (MediaPipe Face Mesh)
* Eye Aspect Ratio (EAR) based eye-closure detection
* Mouth Aspect Ratio (MAR/JAW) yawning detection
* Audio alert when drowsiness detected
* Streamlit web interface
* Lightweight & real-time performance
  
#Pipeline:

Webcam → Face Mesh → EAR/MAR → Drowsiness Logic → Alert + UI


### Technical Details

## Eye Aspect Ratio (EAR)
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

If EAR < threshold → Eye closed → Possible drowsiness

## Mouth Aspect Ratio (MAR)

MAR = vertical mouth distance / horizontal mouth distance

If MAR > threshold → Yawning detected

## 🚀 Future Improvements

* Head pose tracking
* CNN/LSTM fatigue prediction
* Mobile deployment
* Edge device optimization

