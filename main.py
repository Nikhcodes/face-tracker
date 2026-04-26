#!/usr/bin/env python3
"""
pose-cam — real-time hand gesture recognition
----------------------------------------------
Gestures:
  idle       no hands detected (1.5s delay)
  point      index finger only
  peace      index + middle extended
  think      fist at chin level
  rodrick    both fists raised above shoulders
  hello      open hand, all fingers spread
  thumbsup   fist with thumb pointing up
  rockon     index + pinky extended

Controls: ESC to quit
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


def blend(bg, fg, x: int, y: int, scale: float = 1.0):
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
    a   = fgr[:, :, 3:4].astype(np.float32) / 255.0
    src = fgr[:, :, :3].astype(np.float32)
    bg[y1:y2, x1:x2] = (a * src + (1 - a) * roi).astype(np.uint8)
    return bg

# ── Camera polish ─────────────────────────────────────────────────────────────

def build_vignette(h: int, w: int, strength: float = 0.45) -> np.ndarray:
    ky = np.linspace(-1, 1, h)
    kx = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(kx, ky)
    dist = np.sqrt(X**2 + Y**2) / np.sqrt(2)
    mask = 1.0 - np.clip(dist * strength * 1.6, 0, strength)
    return mask.astype(np.float32)


def apply_vignette(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32) * mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_enhance(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.08 + 5, 0, 255)
    frame = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.03, 0, 255)
    frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.97, 0, 255)
    return frame.astype(np.uint8)

# ── Gesture logic ─────────────────────────────────────────────────────────────

def fingers_up(hand) -> list:
    return [hand[tip].y < hand[pip].y for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]


def thumb_up_check(hand) -> bool:
    return hand[THUMB_TIP].y < hand[THUMB_MCP].y - 0.04


def is_fist(hand) -> bool:
    return not any(fingers_up(hand))


def is_open_hand(hand) -> bool:
    up = fingers_up(hand)
    thumb_open = hand[THUMB_TIP].x < hand[THUMB_MCP].x - 0.02
    return all(up) and thumb_open


def hand_center(hand):
    xs = [lm.x for lm in hand]
    ys = [lm.y for lm in hand]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def classify_pose(hands: list, fw: int, fh: int) -> str:
    n = len(hands)

    if n == 2:
        h0, h1 = hands[0], hands[1]
        if is_fist(h0) and is_fist(h1):
            if hand_center(h0)[1] < 0.65 and hand_center(h1)[1] < 0.65:
                return "rodrick"

    if n >= 1:
        hand = hands[0]
        up   = fingers_up(hand)
        cx, cy = hand_center(hand)

        if is_open_hand(hand):
            return "hello"
        if up == [True, True, False, False]:
            return "peace"
        if up == [True, False, False, True]:
            return "rockon"
        if up == [True, False, False, False]:
            return "point"
        if is_fist(hand) and thumb_up_check(hand):
            return "thumbsup"
        if is_fist(hand) and cy > 0.50 and 0.20 < cx < 0.80:
            return "think"

    return ""

# ── Landmark drawing ──────────────────────────────────────────────────────────

def draw_landmarks(frame, hand):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 4 if i in (4, 8, 12, 16, 20) else 2
        cv2.circle(frame, (x, y), r + 1, (0, 0, 0), -1)
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1)

# ── Glass cheat sheet (top-right corner) ─────────────────────────────────────

CHEAT_ROWS = [
    ("idle",     "stare at camera"),
    ("point",    "index finger"),
    ("peace",    "index + middle"),
    ("think",    "fist at chin"),
    ("rodrick",  "both fists up"),
    ("hello",    "open hand"),
    ("thumbsup", "thumb up"),
    ("rockon",   "index + pinky"),
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
_ROW_H     = 18
_PAD       = 10
_FONT_S    = 0.36
_THICK     = 1


def draw_cheatsheet(frame, active_pose: str):
    fh, fw = frame.shape[:2]

    # measure width from longest row
    max_w = 0
    for name, hint in CHEAT_ROWS:
        row_text = f"{name}  {hint}"
        (tw, _), _ = cv2.getTextSize(row_text, FONT, _FONT_S, _THICK)
        max_w = max(max_w, tw)

    box_w = max_w + _PAD * 2
    box_h = len(CHEAT_ROWS) * _ROW_H + _PAD * 2
    margin = 12
    x0 = fw - box_w - margin
    y0 = margin

    # frosted glass — very subtle dark tint, low opacity
    glass = frame.copy()
    cv2.rectangle(glass, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(glass, 0.35, frame, 0.65, 0, frame)

    # thin border
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
    # make border very faint
    border_layer = frame.copy()
    cv2.rectangle(border_layer, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
    cv2.addWeighted(border_layer, 0.18, frame, 0.82, 0, frame)

    # rows
    for i, (name, hint) in enumerate(CHEAT_ROWS):
        row_y = y0 + _PAD + i * _ROW_H + _ROW_H - 4
        is_active = (name == active_pose)

        if is_active:
            hi = frame.copy()
            cv2.rectangle(hi,
                          (x0 + 2, row_y - _ROW_H + 4),
                          (x0 + box_w - 2, row_y + 4),
                          (255, 255, 255), -1)
            cv2.addWeighted(hi, 0.12, frame, 0.88, 0, frame)

        name_col = (255, 255, 255) if is_active else (180, 180, 180)
        hint_col = (160, 160, 160) if is_active else (100, 100, 100)

        cv2.putText(frame, name, (x0 + _PAD, row_y),
                    FONT, _FONT_S, name_col, _THICK, cv2.LINE_AA)

        (nw, _), _ = cv2.getTextSize(name, FONT, _FONT_S, _THICK)
        cv2.putText(frame, f"  {hint}", (x0 + _PAD + nw, row_y),
                    FONT, _FONT_S, hint_col, _THICK, cv2.LINE_AA)

# ── HUD pill label ────────────────────────────────────────────────────────────

LABELS = {
    "idle":     "idle",
    "point":    "point",
    "peace":    "peace",
    "think":    "thinking",
    "rodrick":  "RODRICK!!",
    "hello":    "hello!",
    "thumbsup": "thumbs up",
    "rockon":   "rock on",
}


def draw_hud(frame, pose: str, fade: float = 1.0):
    if not pose or fade <= 0:
        return
    label = LABELS.get(pose, pose).upper()
    fh, fw = frame.shape[:2]
    font_scale, thick = 0.72, 2
    (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, thick)
    pad_x, pad_y = 18, 9
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2
    x = (fw - pill_w) // 2
    y = fh - pill_h - 20

    layer = frame.copy()
    cv2.rectangle(layer, (x, y), (x + pill_w, y + pill_h), (20, 20, 20), -1)
    cv2.rectangle(layer, (x, y), (x + pill_w, y + pill_h), (255, 255, 255), 1)
    cv2.addWeighted(layer, fade * 0.80, frame, 1 - fade * 0.80, 0, frame)

    alpha_val = int(255 * fade)
    text_color = (alpha_val, alpha_val, alpha_val)
    cv2.putText(frame, label, (x + pad_x, y + pad_y + th),
                FONT, font_scale, text_color, thick, cv2.LINE_AA)

# ── Load overlays ─────────────────────────────────────────────────────────────

POSE_FILES = {
    "idle":     "overlays/idle.png",
    "point":    "overlays/point.png",
    "peace":    "overlays/peace.png",
    "think":    "overlays/think.png",
    "rodrick":  "overlays/rodrick.png",
    "hello":    "overlays/hello.png",
    "thumbsup": "overlays/thumbsup.png",
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

frame_count   = 0
FPS           = 30
IDLE_DELAY    = 1.5
last_hand_ts  = time.time()
current_pose  = ""
pose_start_ts = 0.0
FADE_IN_SECS  = 0.20
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

    if vignette_mask is None:
        vignette_mask = build_vignette(fh, fw)

    frame = auto_enhance(frame)
    frame = apply_vignette(frame, vignette_mask)

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms=ts_ms)
    hands  = result.hand_landmarks

    if hands:
        last_hand_ts = now
        for hand in hands:
            draw_landmarks(frame, hand)

    if hands:
        new_pose = classify_pose(hands, fw, fh)
    else:
        new_pose = "idle" if (now - last_hand_ts) >= IDLE_DELAY else ""

    if new_pose != current_pose:
        current_pose  = new_pose
        pose_start_ts = now

    if current_pose and current_pose in overlays:
        ov     = overlays[current_pose]
        oh, ow = ov.shape[:2]
        scale  = min(fw * 0.26 / ow, fh * 0.26 / oh)
        frame  = blend(frame, ov, 24, 24, scale)

    draw_cheatsheet(frame, current_pose)

    if current_pose:
        elapsed = now - pose_start_ts
        fade    = min(elapsed / FADE_IN_SECS, 1.0)
        draw_hud(frame, current_pose, fade)

    cv2.imshow("pose-cam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()