#!/usr/bin/env python3
"""
Hand Gesture Tracker - Photo Booth Edition
-------------------------------------------
Poses:
  😶 Idle          → no hands (1.5s)           → overlays/idle.png
  ☝️  Point         → index only up             → overlays/point.png
  ✌️  Peace         → index + middle up         → overlays/peace.png
  🤔 Think         → fist low-center           → overlays/think.png
  😱 Rodrick       → both fists up             → overlays/rodrick.png
  👋 Hello         → all fingers + thumb open  → overlays/hello.png
  👍 Thumbs up     → thumb up, fingers curled  → overlays/thumbsup.png
  🖕 Middle finger → middle only up            → overlays/middle.png
  🤘 Rock on       → pinky + index up          → overlays/rockon.png

Camera polish:
  - Auto brightness/contrast boost
  - Soft vignette to draw focus to you
  - Slight warmth tone
  - Clean pill-style HUD label with fade-in on pose change

Drop PNGs into overlays/ folder. Press ESC to quit.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time
import urllib.request

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
        MODEL_PATH,
    )
    print("[INFO] Done!")

# ── Landmark indices ──────────────────────────────────────────────────────────

WRIST                  = 0
THUMB_TIP, THUMB_MCP   = 4, 2
INDEX_TIP,  INDEX_PIP  = 8,  6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP,   RING_PIP   = 16, 14
PINKY_TIP,  PINKY_PIP  = 20, 18

FINGER_TIPS = [INDEX_TIP,  MIDDLE_TIP, RING_TIP,  PINKY_TIP]
FINGER_PIPS = [INDEX_PIP,  MIDDLE_PIP, RING_PIP,  PINKY_PIP]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Overlay helpers ───────────────────────────────────────────────────────────

def load_overlay(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def blend(bg, fg, x: int, y: int, scale: float = 1.0, alpha: float = 1.0):
    """Alpha-blend a BGRA overlay onto a BGR background, with optional global alpha."""
    if fg is None:
        return bg
    h0, w0 = fg.shape[:2]
    h, w = int(h0 * scale), int(w0 * scale)
    if h < 1 or w < 1:
        return bg
    fg = cv2.resize(fg, (w, h))

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg.shape[1]), min(y + h, bg.shape[0])
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return bg

    roi = bg[y1:y2, x1:x2].astype(np.float32)
    fgr = fg[fy1:fy2, fx1:fx2]
    a   = (fgr[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    src = fgr[:, :, :3].astype(np.float32)
    bg[y1:y2, x1:x2] = (a * src + (1 - a) * roi).astype(np.uint8)
    return bg

# ── Camera polish ─────────────────────────────────────────────────────────────

def build_vignette(h: int, w: int, strength: float = 0.55) -> np.ndarray:
    """Pre-build a vignette mask (call once, reuse every frame)."""
    ky = np.linspace(-1, 1, h)
    kx = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(kx, ky)
    dist = np.sqrt(X**2 + Y**2)
    dist = dist / dist.max()
    mask = 1.0 - np.clip(dist * strength * 1.4, 0, strength)
    return mask.astype(np.float32)


def apply_vignette(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32)
    out *= mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_enhance(frame: np.ndarray) -> np.ndarray:
    """Subtle brightness/contrast boost + slight warmth."""
    # Convert to LAB and boost L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.08 + 6, 0, 255)
    frame = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    # Tiny warmth: nudge red up, blue down slightly
    frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.03, 0, 255)  # R
    frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.97, 0, 255)  # B
    return frame.astype(np.uint8)

# ── Gesture logic ─────────────────────────────────────────────────────────────

def fingers_up(hand) -> list:
    return [hand[tip].y < hand[pip].y for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]


def thumb_up_check(hand) -> bool:
    """Thumb tip clearly above its MCP joint, and all fingers curled."""
    return hand[THUMB_TIP].y < hand[THUMB_MCP].y - 0.04


def is_fist(hand) -> bool:
    return not any(fingers_up(hand))


def is_open_hand(hand) -> bool:
    up = fingers_up(hand)
    thumb_open = hand[THUMB_TIP].x < hand[THUMB_MCP].x - 0.02  # thumb extended sideways
    return all(up) and thumb_open


def hand_center(hand):
    xs = [lm.x for lm in hand]
    ys = [lm.y for lm in hand]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def classify_pose(hands: list, fw: int, fh: int) -> str:
    n = len(hands)

    # ── Two hands ─────────────────────────────────────────────────────────────
    if n == 2:
        h0, h1 = hands[0], hands[1]
        # Rodrick: both fists raised into upper frame
        if is_fist(h0) and is_fist(h1):
            cy0 = hand_center(h0)[1]
            cy1 = hand_center(h1)[1]
            if cy0 < 0.65 and cy1 < 0.65:
                return "rodrick"

    # ── One hand ──────────────────────────────────────────────────────────────
    if n >= 1:
        hand = hands[0]
        up   = fingers_up(hand)
        cx, cy = hand_center(hand)

        # Hello: all four fingers up + thumb spread open
        if is_open_hand(hand):
            return "hello"

        # Peace: index + middle up only
        if up == [True, True, False, False]:
            return "peace"

        # Rock on: index + pinky up, middle + ring curled
        if up == [True, False, False, True]:
            return "rockon"

        # Middle finger: only middle up
        if up == [False, True, False, False]:
            return "middle"

        # Point: only index up
        if up == [True, False, False, False]:
            return "point"

        # Thumbs up: thumb clearly up, all fingers curled
        if is_fist(hand) and thumb_up_check(hand):
            return "thumbsup"

        # Thinking: plain fist held low and centered
        if is_fist(hand) and cy > 0.50 and 0.20 < cx < 0.80:
            return "think"

    return ""

# ── Landmark drawing ──────────────────────────────────────────────────────────

def draw_landmarks(frame, hand, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 4 if i in (4, 8, 12, 16, 20) else 2
        cv2.circle(frame, (x, y), r + 1, (0, 0, 0), -1)
        cv2.circle(frame, (x, y), r, color, -1)

# ── HUD pill label ────────────────────────────────────────────────────────────

LABELS = {
    "idle":     "type shih",
    "point":    "Akshually",
    "peace":    "peace",
    "think":    "thinking",
    "rodrick":  "RODRICK!!",
    "hello":    "hi",
    "thumbsup": "thumbs up",
    "middle":   "fuck u too then",
    "rockon":   "rock on",
}

# Pose -> BGR accent colour for the pill
COLORS = {
    "idle":     (120, 120, 120),
    "point":    (255, 200,  60),
    "peace":    ( 80, 220, 120),
    "think":    (180, 130, 255),
    "rodrick":  ( 60, 100, 255),
    "hello":    ( 60, 210, 255),
    "thumbsup": ( 60, 210, 255),
    "middle":   ( 60,  60, 220),
    "rockon":   ( 60,  60, 200),
}


def draw_hud(frame, pose: str, fade: float = 1.0):
    """Draw a clean pill-style label at the bottom-center with fade alpha."""
    if not pose or fade <= 0:
        return
    label = LABELS.get(pose, pose).upper()
    color = COLORS.get(pose, (200, 200, 200))

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    fh, fw = frame.shape[:2]
    pad_x, pad_y = 20, 10
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2
    x = (fw - pill_w) // 2
    y = fh - pill_h - 24

    # Draw pill background with alpha blending
    overlay_layer = frame.copy()
    cv2.rectangle(overlay_layer, (x, y), (x + pill_w, y + pill_h), (15, 15, 15), -1)
    cv2.rectangle(overlay_layer, (x, y), (x + pill_w, y + pill_h), color, 2)
    cv2.addWeighted(overlay_layer, fade * 0.88, frame, 1 - fade * 0.88, 0, frame)

    # Draw text
    text_color = tuple(int(c * fade) for c in color)
    cv2.putText(frame, label, (x + pad_x, y + pad_y + th),
                font, scale, text_color, thick, cv2.LINE_AA)

# ── Load overlays ─────────────────────────────────────────────────────────────

POSE_FILES = {
    "idle":     "overlays/idle.png",
    "point":    "overlays/point.png",
    "peace":    "overlays/peace.png",
    "think":    "overlays/think.png",
    "rodrick":  "overlays/rodrick.png",
    "hello":    "overlays/hello.png",
    "thumbsup": "overlays/thumbsup.png",
    "middle":   "overlays/middle.png",
    "rockon":   "overlays/rockon.png",
}

overlays = {}
for pose, path in POSE_FILES.items():
    img = load_overlay(path)
    if img is not None:
        overlays[pose] = img
        print(f"[INFO] Loaded '{pose}' -> {path}")
    else:
        print(f"[WARN] Missing '{pose}' -> {path}")

# ── MediaPipe ─────────────────────────────────────────────────────────────────

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
)

# ── Main loop ─────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# Try to request a higher resolution for a nicer image
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

frame_count  = 0
FPS          = 30
IDLE_DELAY   = 1.5
last_hand_ts = time.time()

# Pose fade state
current_pose  = ""
pose_start_ts = 0.0
FADE_IN_SECS  = 0.25   # how long the label takes to fade in

# Vignette mask — built once after first frame
vignette_mask = None

print("\n[INFO] Running — press ESC to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    ts_ms = frame_count * (1000 // FPS)
    now   = time.time()

    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    # Build vignette once we know frame size
    if vignette_mask is None:
        vignette_mask = build_vignette(fh, fw, strength=0.5)

    # ── Camera polish ──────────────────────────────────────────────────────
    frame = auto_enhance(frame)
    frame = apply_vignette(frame, vignette_mask)

    # ── Hand detection ─────────────────────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms=ts_ms)
    hands  = result.hand_landmarks

    if hands:
        last_hand_ts = now
        for hand in hands:
            draw_landmarks(frame, hand)

    # ── Pose classification ────────────────────────────────────────────────
    if hands:
        new_pose = classify_pose(hands, fw, fh)
    else:
        new_pose = "idle" if (now - last_hand_ts) >= IDLE_DELAY else ""

    if new_pose != current_pose:
        current_pose  = new_pose
        pose_start_ts = now

    # ── Overlay ────────────────────────────────────────────────────────────
    if current_pose and current_pose in overlays:
        ov     = overlays[current_pose]
        oh, ow = ov.shape[:2]
        scale  = min(fw * 0.28 / ow, fh * 0.28 / oh)
        frame  = blend(frame, ov, 28, 28, scale)

    # ── HUD with fade-in ───────────────────────────────────────────────────
    if current_pose:
        elapsed = now - pose_start_ts
        fade    = min(elapsed / FADE_IN_SECS, 1.0)
        draw_hud(frame, current_pose, fade)

    cv2.imshow("Pose Cam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()