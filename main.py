#!/usr/bin/env python3
"""
Pose Cam — Real-Time Hand Gesture Tracker
------------------------------------------
Detects hand poses via MediaPipe and overlays
a custom PNG image for each recognised gesture.

Poses:
  idle       no hands detected (after 1.5s)
  point      index finger only
  peace      index + middle up
  think      fist held low and centered
  rodrick    both fists raised near shoulders
  hello      open hand, all fingers spread
  thumbsup   fist with thumb pointing up
  rockon     index + pinky up

Controls:  ESC to quit
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time
import urllib.request

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ── Catppuccin Mocha palette (BGR) ────────────────────────────────────────────
MOCHA = {
    "base":    (  30,  30,  46),   # #1e1e2e
    "surface": (  49,  50,  69),   # #313244
    "overlay": (  62,  60,  88),   # #45475a (approx)
    "text":    ( 205, 214, 244),   # #cdd6f4
    "subtext": ( 166, 173, 200),   # #a6adc8
    "lavender":( 235, 190, 180),   # #b4befe
    "blue":    ( 243, 189, 137),   # #89b4fa
    "green":   ( 166, 227, 161),   # #a6e3a1
    "yellow":  ( 148, 226, 213),   # #d5c4a1 approx / using teal
    "peach":   ( 122, 162, 250),   # #fab387
    "mauve":   ( 203, 166, 247),   # #cba4f7
    "red":     ( 114, 135, 243),   # #f38ba8
    "sky":     ( 231, 218, 137),   # #89dceb
    "sapphire":( 220, 185, 116),   # #74c7ec
    "pink":    ( 193, 148, 245),   # #f5c2e7
    "teal":    ( 180, 214, 148),   # #94e2d5
}

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


def blend(bg, fg, x: int, y: int, scale: float = 1.0, alpha: float = 1.0):
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
def build_vignette(h: int, w: int, strength: float = 0.5) -> np.ndarray:
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
    line_color = MOCHA["overlay"]
    dot_color  = MOCHA["lavender"]
    tip_color  = MOCHA["blue"]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        is_tip = i in (4, 8, 12, 16, 20)
        r = 5 if is_tip else 3
        cv2.circle(frame, (x, y), r + 1, MOCHA["base"], -1)
        cv2.circle(frame, (x, y), r, tip_color if is_tip else dot_color, -1)

# ── Pose panel (right sidebar) ────────────────────────────────────────────────
POSE_TABLE = [
    ("idle",     "stare at camera"),
    ("point",    "index finger only"),
    ("peace",    "index + middle up"),
    ("think",    "fist at chin level"),
    ("rodrick",  "both fists up high"),
    ("hello",    "open hand, spread"),
    ("thumbsup", "fist, thumb up"),
    ("rockon",   "index + pinky up"),
]

POSE_COLORS = {
    "idle":     MOCHA["subtext"],
    "point":    MOCHA["yellow"],
    "peace":    MOCHA["green"],
    "think":    MOCHA["mauve"],
    "rodrick":  MOCHA["red"],
    "hello":    MOCHA["sky"],
    "thumbsup": MOCHA["peach"],
    "rockon":   MOCHA["pink"],
}

PANEL_W      = 220
PANEL_PAD    = 14
ROW_H        = 36
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL   = 0.42
FONT_LABEL   = 0.52
THICK        = 1


def draw_panel(canvas: np.ndarray, active_pose: str):
    """Draw the Catppuccin-styled pose reference panel on the right."""
    fh, fw = canvas.shape[:2]
    px = fw - PANEL_W
    panel_h = fh

    # Panel background
    overlay_layer = canvas.copy()
    cv2.rectangle(overlay_layer, (px, 0), (fw, panel_h), MOCHA["base"], -1)
    cv2.addWeighted(overlay_layer, 0.82, canvas, 0.18, 0, canvas)

    # Thin separator line
    cv2.line(canvas, (px, 0), (px, panel_h), MOCHA["surface"], 1)

    # Header
    header_y = 28
    cv2.putText(canvas, "POSES", (px + PANEL_PAD, header_y),
                FONT, FONT_LABEL, MOCHA["text"], THICK, cv2.LINE_AA)
    cv2.line(canvas, (px + PANEL_PAD, header_y + 6),
             (fw - PANEL_PAD, header_y + 6), MOCHA["surface"], 1)

    # Rows
    for i, (pose, hint) in enumerate(POSE_TABLE):
        row_y = header_y + 20 + i * ROW_H
        color = POSE_COLORS.get(pose, MOCHA["subtext"])
        is_active = (pose == active_pose)

        # Active row highlight
        if is_active:
            hi = canvas.copy()
            cv2.rectangle(hi,
                          (px + 4, row_y - 2),
                          (fw - 4, row_y + ROW_H - 8),
                          MOCHA["surface"], -1)
            cv2.addWeighted(hi, 0.7, canvas, 0.3, 0, canvas)

        # Accent dot
        dot_x = px + PANEL_PAD + 5
        dot_y = row_y + 10
        cv2.circle(canvas, (dot_x, dot_y), 4, color, -1)

        # Pose name
        name_color = MOCHA["text"] if is_active else color
        cv2.putText(canvas, pose, (dot_x + 12, dot_y + 4),
                    FONT, FONT_LABEL, name_color, THICK, cv2.LINE_AA)

        # Hint text
        cv2.putText(canvas, hint, (dot_x + 12, dot_y + 18),
                    FONT, FONT_SMALL, MOCHA["subtext"], 1, cv2.LINE_AA)

    # Footer
    footer_text = "ESC to quit"
    (fw2, _), _ = cv2.getTextSize(footer_text, FONT, FONT_SMALL, 1)
    cv2.putText(canvas, footer_text,
                (px + (PANEL_W - fw2) // 2, panel_h - 14),
                FONT, FONT_SMALL, MOCHA["overlay"], 1, cv2.LINE_AA)

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


def draw_hud(frame, pose: str, fade: float = 1.0, cam_w: int = 0):
    if not pose or fade <= 0:
        return
    label = LABELS.get(pose, pose).upper()
    color = POSE_COLORS.get(pose, MOCHA["text"])

    scale, thick = 0.72, 2
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, thick)
    fh = frame.shape[0]
    if cam_w == 0:
        cam_w = frame.shape[1]

    pad_x, pad_y = 18, 9
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2
    x = (cam_w - pill_w) // 2
    y = fh - pill_h - 20

    # Pill background
    layer = frame.copy()
    cv2.rectangle(layer, (x, y), (x + pill_w, y + pill_h), MOCHA["base"], -1)
    cv2.rectangle(layer, (x, y), (x + pill_w, y + pill_h), color, 2)
    cv2.addWeighted(layer, fade * 0.90, frame, 1 - fade * 0.90, 0, frame)

    text_color = tuple(int(c * fade) for c in color)
    cv2.putText(frame, label, (x + pad_x, y + pad_y + th),
                FONT, scale, text_color, thick, cv2.LINE_AA)

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

    # Camera polish
    frame = auto_enhance(frame)
    frame = apply_vignette(frame, vignette_mask)

    # Hand detection
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms=ts_ms)
    hands  = result.hand_landmarks

    if hands:
        last_hand_ts = now
        for hand in hands:
            draw_landmarks(frame, hand)

    # Pose classification
    if hands:
        new_pose = classify_pose(hands, fw, fh)
    else:
        new_pose = "idle" if (now - last_hand_ts) >= IDLE_DELAY else ""

    if new_pose != current_pose:
        current_pose  = new_pose
        pose_start_ts = now

    # Overlay image (keep within camera area, left of panel)
    cam_display_w = fw - PANEL_W
    if current_pose and current_pose in overlays:
        ov     = overlays[current_pose]
        oh, ow = ov.shape[:2]
        scale  = min(cam_display_w * 0.26 / ow, fh * 0.26 / oh)
        frame  = blend(frame, ov, 24, 24, scale)

    # Build final canvas with panel
    canvas = np.zeros((fh, fw + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :fw] = frame
    draw_panel(canvas, current_pose)

    # HUD label centered over camera area only
    if current_pose:
        elapsed = now - pose_start_ts
        fade    = min(elapsed / FADE_IN_SECS, 1.0)
        draw_hud(canvas, current_pose, fade, cam_w=fw)

    cv2.imshow("Pose Cam", canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()