<div align="center">

<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/MediaPipe-Tasks_API-FF6F00?style=for-the-badge&logo=google&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-4-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />

<br />
<br />

# pose cam

### real-time hand gesture detection — clean, fast, and actually fun

<br />

**[repository](https://github.com/Nikhcodes/face-tracker)**

<br />

</div>

---

## a different kind of webcam tool

most gesture trackers are demos
this one is for actually using

strike a pose — see it react instantly

---

## poses

```
idle          stare at the camera (triggers after 1.5s)
point         index finger only
peace         index + middle up
thinking      fist held low and centered (chin level)
rodrick       both fists raised near your shoulders
hello         open hand, all fingers spread
thumbs up     fist with thumb pointing up
middle        middle finger only
rock on       index + pinky up, middle + ring curled
```

<sub>each pose maps to its own overlay image — swap them out for whatever you want</sub>

---

## camera polish

- auto brightness and contrast boost via LAB color space
- subtle vignette to draw focus to you
- slight warmth tone
- pill-style label at the bottom that fades in on pose change
- requests 1280x720 from your webcam automatically

<sub>built to look good for photos, not just for demos</sub>

---

## stack

```
python 3.12+  .  mediapipe tasks api  .  opencv  .  numpy
```

<sub>no servers, no accounts, runs entirely on your machine</sub>

---

## setup

```bash
git clone https://github.com/Nikhcodes/face-tracker.git
cd face-tracker
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install mediapipe opencv-python numpy
python main.py
```

the model (`hand_landmarker.task`) downloads automatically on first run.

---

## overlays

create an `overlays/` folder next to `main.py` and drop in your PNGs:

```
overlays/
  idle.png
  point.png
  peace.png
  think.png
  rodrick.png
  hello.png
  thumbsup.png
  middle.png
  rockon.png
```

missing files are skipped gracefully — you can start with just a few.
supports both transparent (RGBA) and regular (RGB) PNGs.

---

## controls

```
ESC          quit
Ctrl+C       force quit from terminal
```

---

## structure

```
face-tracker/
  main.py               entry point and main loop
  hand_landmarker.task  mediapipe model (auto-downloaded)
  overlays/             your PNG images, one per pose
```

---

## how it works

mediapipe tracks 21 landmarks on each hand in real time.
the script reads which finger tips are above their knuckle joints
to classify the current pose on every frame.
the result triggers an overlay image and a label — no ml training needed,
just geometry.

---

<div align="center">

built by nikh &nbsp; <sub>with intention</sub>

</div>
