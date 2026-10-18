# Architecture

POSE CAM is a static single-page app: Vite + React + TypeScript, with no server. Everything (camera, hand tracking, effects, photos, the pose editor) runs in the visitor's browser.

## Data flow

```
getUserMedia ─► <video> ─┬─► createImageBitmap (≤640 px) ─► Web Worker: MediaPipe HandLandmarker
                         │                                        │
                         │                       21 image + 21 world landmarks per hand
                         │                                        ▼
                         │                         classifyFrame()  (gestures/classify.ts)
                         │                                        ▼
                         │                         PoseStabilizer   (gestures/stabilizer.ts)
                         │                                        ▼
                         │                     enter / exit / held events
                         │                                        ▼
                         └──────────────► StudioRenderer + ReactionDirector ─► <canvas> ─► photo
```

`StudioEngine` (src/engine) owns the camera, the detector and the `requestAnimationFrame` loop. React only subscribes to a small status object (camera state, model state, current pose, countdown), so per-frame work never causes React renders.

## Key decisions

| Decision | Why |
| --- | --- |
| **Vite static build, not Next.js** | Everything is client-side. A server framework adds hydration and SSR edge cases around `window`, camera and canvas, for nothing. |
| **Self-hosted MediaPipe** | `scripts/copy-mediapipe.mjs` copies the WASM runtime from the exact npm version pinned in `package-lock.json` (`@mediapipe/tasks-vision@0.10.35`) into `public/mediapipe/<version>/` at build time. The model (`public/models/hand_landmarker.task`) is committed. Visitors make no third-party requests. |
| **Detection in a module Web Worker** | Frames are transferred as `ImageBitmap`s, one in flight at a time. Rendering stays smooth even when inference is slow. If the worker path fails (no `OffscreenCanvas`, worker quirks), detection runs on the main thread. GPU is tried first, then CPU. |
| **Lazy loading** | The vision library (~136 kB) and the pose editor (~35 kB) are separate chunks, loaded only when used. The model loads once and survives camera restarts. |
| **One canvas** | Camera, colour look, vignette, grain, landmarks, stickers, particles and captions are all drawn onto one canvas. A photo is a copy of it plus a small wordmark. Stickers are same-origin or `data:` images, so the canvas is never tainted. |
| **Blend-mode colour looks** | `soft-light` / `multiply` / `saturation` fills are GPU-cheap and identical in preview and photo; `ctx.filter` isn't available everywhere. |
| **History-API router** | Five routes don't need a routing dependency. `vercel.json` rewrites extension-less paths to `index.html`. |

## Gesture recognition

**Features** (`gestures/features.ts`) are computed on MediaPipe's metric *world* landmarks:

- **Finger state:** total bend across MCP + PIP + DIP joints (knuckle bend measured with sideways spread removed), combined with the ratio tip→wrist / knuckle→wrist. Each digit is `extended`, `folded` or `uncertain`.
- **Thumb state:** its bend, the distance from its tip to the fingers/palm, and its reach away from the pinky side.
- **Thumb direction:** the vertical component of MCP→tip, plus whether the tip is the highest or lowest point of the hand.
- **Position:** hand centre and size in image space, corrected for the frame's aspect ratio.

None of the finger features depend on rotation, so poses work tilted, sideways or upside down, with either hand.

**Classifier** (`gestures/classify.ts`) is pure: landmarks in, a verdict out. The verdict is `pose`, `unknown` (with a reason: `unrecognized`, `uncertain` or `ambiguous`) or `none`. Custom editor poses are checked first, then two-hand built-ins, then one-hand built-ins. Switched-off poses are skipped.

**Stabilizer** (`gestures/stabilizer.ts`):
- a pose must win for 160 ms and 3 frames to show (320 ms for experimental poses);
- it survives 300 ms drop-outs;
- 1.5 s without hands means idle;
- holding a pose for 1.2 s fires `held`;
- a per-pose cooldown stops repeated sounds.

**Mirroring** is purely visual. Classification always uses the raw camera frame, so the mirrored preview can't flip results.

## The pose registry

`config/registry.ts` merges three layers into the live list of poses:

1. the **local pack**, edited in this browser and stored in IndexedDB;
2. the **site pack** at `/packs/default.json`, shipped with the deployment;
3. the **built-in catalog** in `config/poses.ts`.

Every consumer (classifier options, reactions, pose guide, poses page, sticker preloading) reads from the registry, so edits apply live. See [custom-poses.md](custom-poses.md) for the pack format.

## Synthetic hand model

`gestures/synthetic.ts` is a small kinematic hand that outputs MediaPipe-shaped landmarks for any finger flexion, rotation, hand, size, aspect ratio and noise level. It drives the unit tests and every hand illustration in the UI, so the drawings match the logic exactly. It is an approximation, not a replacement for real-hand testing.

## Privacy guarantees (and how they're enforced)

- No network requests except the site's own static files. The e2e suite asserts zero third-party requests during a full camera session.
- The camera starts only from a click. Leaving the studio or pressing camera-off stops every track (unit and e2e tested).
- No analytics or cookies. `localStorage` holds settings and found poses; IndexedDB holds editor changes.
