#!/usr/bin/env python3
"""
Hand Gesture Tracker with Overlay (Robust Version)
-------------------------------------------------
- Works with RGB or RGBA overlay images
- Uses MediaPipe Tasks API (HandLandmarker)
- Detects index finger above wrist
- Displays overlay when gesture is detected
- Fully compatible with Python 3.12+ and Fedora
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# -------------------------------
# 1️⃣ Model setup
# -------------------------------
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[INFO] Model downloaded!")

# -------------------------------
# 2️⃣ Initialize HandLandmarker
# -------------------------------
base_options = BaseOptions(model_asset_path=MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(hand_options)

# -------------------------------
# 3️⃣ Load overlay PNG and handle RGB/RGBA
# -------------------------------
overlay = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED)
if overlay is None:
    raise FileNotFoundError("overlay.png not found in project folder.")

# If image has only 3 channels (BGR), convert to BGRA for alpha support
if overlay.shape[2] == 3:
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

def overlay_image(background, overlay_img, x, y):
    """
    Overlay a BGRA image onto a BGR background at (x, y).
    Works for images with or without transparency.
    """
    h, w, _ = overlay_img.shape
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    alpha = overlay_img[:, :, 3] / 255.0  # alpha channel
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay_img[:, :, c] + (1 - alpha) * background[y:y+h, x:x+w, c]
        )
    return background

# -------------------------------
# 4️⃣ Initialize webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

frame_count = 0
fps_estimate = 30  # approximate fps for timestamps

# -------------------------------
# 5️⃣ Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to read frame. Exiting...")
        break

    frame_count += 1
    timestamp_ms = frame_count * int(1000 / fps_estimate)

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks
    results = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

    # -------------------------------
    # Gesture detection: index finger above wrist
    # -------------------------------
    gesture_detected = False
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]  # first detected hand
        wrist = hand[0]
        index_tip = hand[8]

        # Coordinates are normalized: (0,0) top-left, (1,1) bottom-right
        if index_tip.y < wrist.y:
            gesture_detected = True

    # Overlay image and label if gesture detected
    if gesture_detected:
        frame = overlay_image(frame, overlay, 50, 50)
        cv2.putText(frame, "Gesture Detected!", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam
    cv2.imshow("Hand Gesture Tracker", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# 6️⃣ Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
