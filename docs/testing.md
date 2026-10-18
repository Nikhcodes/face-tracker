# Testing

## Automated

```bash
npm test             # unit tests (Vitest + jsdom)
npm run build && npm run test:e2e   # browser tests (Playwright + Chromium fake camera)
```

E2E needs Chromium: run `npx playwright install chromium` once, or set `PW_CHROMIUM_PATH=/path/to/chrome`.

| Suite | Covers |
| --- | --- |
| `gestures/classify.test.ts` | every single-hand pose across 180 synthetic orientations (roll/yaw/pitch, both hands, 2 aspect ratios, sizes, ±2 mm noise); thumbs sweeps; position and two-hand poses; heart geometry; uncertain, ambiguous and empty frames; mirroring |
| `gestures/stabilizer.test.ts` | enter window, no flicker, confidence threshold, hysteresis, switching, experimental delay, idle, held, cooldown |
| `gestures/customRules.test.ts` | editor rule validation, matching in any orientation, thumb direction, zone, two hands, copy-from-camera, custom-first and disabled-pose classification |
| `config/registry.test.ts`, `config/initPacks.test.ts` | pack validation (unsafe images, unknown ids, limits), JSON round trip, override merging, conflict detection, site pack loading |
| `editor/editPack.test.ts` | editor patches only store real changes; enable/disable; pictures; create/duplicate/delete |
| `engine/StudioEngine.test.ts` | camera only after `start()`, model loaded once, full release on `stop()`/`dispose()`, model failure keeps the camera usable, pose unlocks |
| `camera/camera.test.ts`, `render/capture.test.ts`, `state/settings.test.ts` | error mapping, constraint retry, photo composition and encoding, sharing fallbacks, settings sanitising |
| `e2e/app.spec.ts` | no camera on load; navigation, refresh and 404; denied and unsupported camera; full pipeline (camera → local inference → photo → release, zero third-party requests); pose editor create, reload, export, reset, import |

The synthetic hand proves the logic is orientation-independent and stable. It is **not** a substitute for real hands, so use the checklist below.

## Manual webcam checklist

Run on the deployed HTTPS URL in desktop Chrome/Edge, Safari (macOS + iOS), Firefox and Android Chrome.

**Camera**
- [ ] Landing loads with no permission prompt. **Enter the camera** prompts once; **Allow** shows the mirrored preview.
- [ ] Block permission → clear error screen → re-allow in site settings → **Try again** works.
- [ ] Camera-off and **Exit** turn the camera light off. Switching tabs and back resumes cleanly.
- [ ] Switch camera (phone front/back). Mirror off/on keeps poses correct.
- [ ] Smooth / Balanced / Sharp modes. Smooth stays fluid on a mid-range phone.

**Poses**
- [ ] Each live pose shows its sticker and caption within about ¼ s and stays steady while held. Try them tilted and sideways.
- [ ] No cross-talk: thumbs up vs down, point vs finger gun, peace vs three.
- [ ] Two open hands → confetti. Two thumbs up → double thumbs.
- [ ] Experimental poses (finger gun, thinking, rodrick, heart): note how reliable each is.
- [ ] Hands down → idle cat after about 1.5 s.

**Photos and sound**
- [ ] Three fingers → 3-2-1 → photo dialog. The photo matches the preview. PNG and JPEG download.
- [ ] Share photo opens the share sheet on mobile.
- [ ] Sound is off by default. Toggling it plays cues; music and volume work.

**Pose editor**
- [ ] New pose → **Copy my hand shape** in the tester → **Detected** shows when you repeat the pose.
- [ ] Upload a picture → it appears in the preview and in the camera.
- [ ] Switch a built-in off → it's no longer detected and leaves the guide.
- [ ] Export → put it at `public/packs/default.json` → deploy → a fresh browser sees the poses.

**Other**
- [ ] OS "reduce motion" → no shake, calmer animations.
- [ ] Keyboard only: Tab through controls, Space shoots, Esc closes dialogs.
