/**
 * Owns everything that runs per frame: camera stream, detector, classifier,
 * stabilizer, renderer, reactions, audio cues and the photo timer.
 *
 * React only sees a small, slowly-changing status object (see `status`), so
 * per-frame work never triggers React renders.
 */
import { CameraError, listCameras, mapCameraError, openCamera, stopStream, type CameraErrorCode } from '../camera/camera';
import { getPose, livePoseIds, registryStore } from '../config/registry';
import { sound } from '../audio/sound';
import { classifyFrame, type FrameResult } from '../gestures/classify';
import type { PoseId } from '../gestures/poseIds';
import { Cooldown, PoseStabilizer, type DisplayState, type StabilizerEvent } from '../gestures/stabilizer';
import type { HandObservation } from '../gestures/types';
import { createStore, type Store } from '../lib/store';
import { canvasToBlob, composePhoto } from '../render/capture';
import { ReactionDirector, type ReactionFrame } from '../render/reactions';
import { StudioRenderer } from '../render/renderer';
import { StickerLibrary } from '../render/stickers';
import { hasAllLivePoses, unlockPose, unlockedStore } from '../state/progress';
import { prefersReducedMotion, settingsStore, type PhotoFormat, type Settings } from '../state/settings';
import { createDetector, type DetectorBackend, type HandDetector } from '../vision/detector';

export interface StudioStatus {
  camera: 'off' | 'starting' | 'on' | 'error';
  cameraError: CameraErrorCode | null;
  model: 'idle' | 'loading' | 'ready' | 'error';
  modelError: string | null;
  backend: DetectorBackend | null;
  devices: { id: string; label: string }[];
  deviceId: string | null;
  display: DisplayState;
  /** Whole seconds left on the photo timer, or null. */
  countdown: number | null;
  stats: { fps: number; detectFps: number; inferenceMs: number; verdict: string };
}

export interface Toast {
  id: number;
  title: string;
  body?: string;
  tone?: 'default' | 'accent' | 'gold';
}

export interface CapturedPhoto {
  blob: Blob;
  url: string;
  format: PhotoFormat;
  width: number;
  height: number;
}

const INITIAL_STATUS: StudioStatus = {
  camera: 'off',
  cameraError: null,
  model: 'idle',
  modelError: null,
  backend: null,
  devices: [],
  deviceId: null,
  display: { kind: 'none' },
  countdown: null,
  stats: { fps: 0, detectFps: 0, inferenceMs: 0, verdict: '' },
};

/** Minimum time between detections, per quality mode. */
const DETECT_INTERVAL: Record<Settings['quality'], number> = { performance: 66, balanced: 0, high: 0 };
const HANDS_STALE_MS = 450;
const TIMER_SECONDS = 3;

function sameDisplay(a: DisplayState, b: DisplayState): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'pose' && b.kind === 'pose') return a.pose === b.pose;
  if (a.kind === 'unknown' && b.kind === 'unknown') return a.reason === b.reason;
  return true;
}

export class StudioEngine {
  readonly status: Store<StudioStatus> = createStore(INITIAL_STATUS);
  onToast: (t: Toast) => void = () => {};
  onTimerCapture: (photo: CapturedPhoto) => void = () => {};

  private video: HTMLVideoElement | null = null;
  private renderer: StudioRenderer | null = null;
  private readonly stickers = new StickerLibrary();
  private readonly director = new ReactionDirector(this.stickers);
  private readonly stabilizer = new PoseStabilizer();
  private readonly soundCooldown = new Cooldown(1200);
  private stream: MediaStream | null = null;
  private detector: HandDetector | null = null;
  private detectorPromise: Promise<HandDetector | null> | null = null;
  private raf = 0;
  private generation = 0;
  private inFlight = false;
  private lastDetectAt = 0;
  private lastVideoTime = -1;
  private lastFrameAt = 0;
  private hands: HandObservation[] = [];
  private handsAt = 0;
  private aspect = 16 / 9;
  private countdown: { endsAt: number; seconds: number; lastTick: number } | null = null;
  private timerArmed = true;
  private threeSince: number | null = null;
  private toastId = 1;
  private fps = { frames: 0, detections: 0, inference: 0, since: 0 };
  private lastVerdict = '';
  private unsubs: Array<() => void> = [];
  private disposed = false;

  attach(video: HTMLVideoElement, canvas: HTMLCanvasElement): void {
    this.video = video;
    this.renderer = new StudioRenderer(canvas, this.director);
    void this.stickers.load();
    this.unsubs.push(registryStore.subscribe(() => void this.stickers.load()));
    // canvas text needs the display font loaded before the first caption is drawn
    void document.fonts?.load(`800 48px 'Bricolage Grotesque Variable'`).catch(() => undefined);
    void document.fonts?.load(`700 24px 'Inter Variable'`).catch(() => undefined);
    const onVisibility = () => {
      if (!document.hidden) {
        this.stabilizer.reset(performance.now());
        this.lastFrameAt = performance.now();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    const onDeviceChange = () => void this.refreshDevices();
    navigator.mediaDevices?.addEventListener?.('devicechange', onDeviceChange);
    let prevQuality = settingsStore.get().quality;
    const unsubSettings = settingsStore.subscribe(() => {
      const q = settingsStore.get().quality;
      if (q !== prevQuality) {
        prevQuality = q;
        if (this.status.get().camera === 'on') void this.restartCamera();
      }
    });
    this.unsubs.push(
      () => document.removeEventListener('visibilitychange', onVisibility),
      () => navigator.mediaDevices?.removeEventListener?.('devicechange', onDeviceChange),
      unsubSettings,
    );
  }

  private patch(p: Partial<StudioStatus>): void {
    this.status.set((s) => ({ ...s, ...p }));
  }

  private toast(title: string, body?: string, tone: Toast['tone'] = 'default'): void {
    this.onToast({ id: this.toastId++, title, body, tone });
  }

  /** Starts the camera (must be triggered by a user action) and the detector. */
  async start(): Promise<void> {
    if (this.disposed || !this.video) return;
    const st = this.status.get();
    if (st.camera === 'on' || st.camera === 'starting') return;
    this.patch({ camera: 'starting', cameraError: null });
    void this.ensureDetector();
    const gen = ++this.generation;
    try {
      const stream = await openCamera({ deviceId: st.deviceId, quality: settingsStore.get().quality });
      if (gen !== this.generation || this.disposed) {
        stopStream(stream);
        return;
      }
      await this.useStream(stream);
      this.patch({ camera: 'on' });
      void this.refreshDevices();
      this.startLoop();
    } catch (err) {
      if (gen !== this.generation) return;
      const e = err instanceof CameraError ? err : mapCameraError(err);
      this.patch({ camera: 'error', cameraError: e.code });
    }
  }

  private async useStream(stream: MediaStream): Promise<void> {
    const video = this.video!;
    this.stream = stream;
    const track = stream.getVideoTracks()[0];
    track?.addEventListener('ended', () => {
      if (this.stream === stream) {
        this.stop();
        this.patch({ camera: 'error', cameraError: 'not-found' });
      }
    });
    const settingsDevice = track?.getSettings?.().deviceId ?? null;
    if (settingsDevice) this.patch({ deviceId: settingsDevice });
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play().catch(() => undefined);
    if (video.readyState < 2) {
      await new Promise<void>((resolve) => {
        const done = () => resolve();
        video.addEventListener('loadeddata', done, { once: true });
        setTimeout(done, 3000);
      });
    }
  }

  /** Stops the camera and per-frame work. The model stays loaded for a quick restart. */
  stop(): void {
    this.generation++;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    stopStream(this.stream);
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    this.stabilizer.reset();
    this.director.reset();
    this.hands = [];
    this.countdown = null;
    this.threeSince = null;
    this.timerArmed = true;
    this.patch({ camera: 'off', display: { kind: 'none' }, countdown: null });
  }

  private async restartCamera(): Promise<void> {
    const deviceId = this.status.get().deviceId;
    this.stop();
    this.patch({ deviceId });
    await this.start();
  }

  async switchCamera(): Promise<void> {
    const { devices, deviceId } = this.status.get();
    if (devices.length < 2) return;
    const idx = devices.findIndex((d) => d.id === deviceId);
    const next = devices[(idx + 1) % devices.length];
    this.stop();
    this.patch({ deviceId: next.id });
    await this.start();
  }

  async selectCamera(id: string): Promise<void> {
    if (id === this.status.get().deviceId) return;
    this.stop();
    this.patch({ deviceId: id });
    await this.start();
  }

  private async refreshDevices(): Promise<void> {
    const cams = await listCameras();
    this.patch({ devices: cams.map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i + 1}` })).filter((d) => d.id) });
  }

  private ensureDetector(): Promise<HandDetector | null> {
    if (this.detector) return Promise.resolve(this.detector);
    if (this.detectorPromise) return this.detectorPromise;
    this.patch({ model: 'loading', modelError: null });
    this.detectorPromise = createDetector()
      .then((d) => {
        if (this.disposed) {
          d.close();
          return null;
        }
        this.detector = d;
        this.patch({ model: 'ready', backend: d.backend });
        return d;
      })
      .catch((err: unknown) => {
        console.error('[pose-cam] Hand detector failed to load', err);
        this.patch({ model: 'error', modelError: err instanceof Error ? err.message : String(err) });
        return null;
      })
      .finally(() => {
        this.detectorPromise = null;
      });
    return this.detectorPromise;
  }

  retryModel(): void {
    if (this.status.get().model === 'error') void this.ensureDetector();
  }

  private startLoop(): void {
    cancelAnimationFrame(this.raf);
    this.lastFrameAt = performance.now();
    this.fps = { frames: 0, detections: 0, inference: 0, since: this.lastFrameAt };
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      this.frame();
    };
    this.raf = requestAnimationFrame(tick);
  }

  private reactionFrame(now: number, s: Settings): ReactionFrame {
    const r = this.renderer!;
    return {
      now,
      hands: this.hands,
      view: { w: r.canvas.width, h: r.canvas.height, mirror: s.mirror },
      reducedMotion: prefersReducedMotion(s),
      chaos: s.chaos,
      stickers: s.stickers,
      reactions: s.reactions,
    };
  }

  private frame(): void {
    const video = this.video;
    const renderer = this.renderer;
    if (!video || !renderer || !this.stream) return;
    const now = performance.now();
    const dt = now - this.lastFrameAt;
    this.lastFrameAt = now;
    const s = settingsStore.get();

    this.maybeDetect(now, s);
    if (this.hands.length && now - this.handsAt > HANDS_STALE_MS) this.hands = [];
    this.updateTimer(now, s);

    renderer.render({
      now,
      dt,
      video,
      settings: s,
      hands: this.hands,
      aspect: this.aspect,
      reducedMotion: prefersReducedMotion(s),
      countdown: this.countdown,
      goldFrame: s.goldFrame && hasAllLivePoses(),
    });

    this.fps.frames++;
    if (now - this.fps.since >= 1000) {
      if (s.debug) {
        const secs = (now - this.fps.since) / 1000;
        this.patch({
          stats: {
            fps: Math.round(this.fps.frames / secs),
            detectFps: Math.round(this.fps.detections / secs),
            inferenceMs: this.fps.detections ? Math.round((this.fps.inference / this.fps.detections) * 10) / 10 : 0,
            verdict: this.lastVerdict,
          },
        });
      }
      this.fps = { frames: 0, detections: 0, inference: 0, since: now };
    }
  }

  private maybeDetect(now: number, s: Settings): void {
    const detector = this.detector;
    const video = this.video!;
    if (!detector || this.inFlight || video.readyState < 2) return;
    if (now - this.lastDetectAt < DETECT_INTERVAL[s.quality]) return;
    if (video.currentTime === this.lastVideoTime) return; // no new camera frame yet
    this.lastVideoTime = video.currentTime;
    this.lastDetectAt = now;
    this.inFlight = true;
    const gen = this.generation;
    detector
      .detect(video, now)
      .then((res) => {
        if (!res || gen !== this.generation) return;
        this.fps.detections++;
        this.fps.inference += res.inferenceMs;
        this.aspect = res.width / Math.max(1, res.height);
        this.acceptHands(res.hands, performance.now());
        const result = classifyFrame({ hands: res.hands, aspect: this.aspect }, this.classifyOptions());
        this.lastVerdict = describeResult(result);
        this.applyResult(result, performance.now());
      })
      .catch((err) => console.warn('[pose-cam] detection error', err))
      .finally(() => {
        this.inFlight = false;
      });
  }

  private classifyOptions() {
    const reg = registryStore.get();
    return { custom: reg.custom, disabled: reg.disabled };
  }

  /** Light smoothing for drawing; keeps hand order stable between frames. */
  private acceptHands(next: HandObservation[], now: number): void {
    const prev = this.hands;
    let ordered = next;
    if (prev.length === 2 && next.length === 2) {
      const d = (a: HandObservation, b: HandObservation) => Math.hypot(a.image[0].x - b.image[0].x, a.image[0].y - b.image[0].y);
      if (d(prev[0], next[1]) + d(prev[1], next[0]) < d(prev[0], next[0]) + d(prev[1], next[1])) ordered = [next[1], next[0]];
    }
    if (prev.length === ordered.length && now - this.handsAt < 200) {
      const k = 0.6;
      ordered = ordered.map((h, i) => ({
        ...h,
        image: h.image.map((p, j) => {
          const q = prev[i].image[j];
          return { x: q.x + (p.x - q.x) * k, y: q.y + (p.y - q.y) * k, z: p.z };
        }),
      }));
    }
    this.hands = ordered;
    this.handsAt = now;
  }

  private applyResult(result: FrameResult, now: number): void {
    const { state, events } = this.stabilizer.update(result, now);
    for (const e of events) this.handleEvent(e, now);
    if (!sameDisplay(state, this.status.get().display)) this.patch({ display: state });
    if (state.kind === 'pose' && state.pose === 'three') {
      if (this.threeSince === null) this.threeSince = now;
    } else {
      this.threeSince = null;
      this.timerArmed = true;
    }
  }

  private handleEvent(e: StabilizerEvent, now: number): void {
    const s = settingsStore.get();
    const frame = this.reactionFrame(now, s);
    if (e.type === 'enter') {
      this.director.enter(e.pose, frame);
      if (e.pose !== 'idle' && this.soundCooldown.take(e.pose, now)) sound.play(getPose(e.pose).sound);
      if (e.pose !== 'idle') this.maybeUnlock(e.pose);
    } else if (e.type === 'exit') {
      this.director.exit(e.pose, frame);
    } else {
      this.director.heldFor(e.pose, frame);
    }
  }

  private maybeUnlock(id: PoseId): void {
    const hadAll = hasAllLivePoses();
    if (!unlockPose(id)) return;
    const live = livePoseIds();
    const count = [...unlockedStore.get()].filter((p) => live.includes(p)).length;
    const def = getPose(id);
    const extra =
      def.status === 'experimental'
        ? 'Experimental pose — nice work.'
        : def.status === 'custom'
          ? 'One of your custom poses.'
          : `${count} of ${live.length} live poses found`;
    this.toast(`Pose unlocked: ${def.name}`, extra, 'accent');
    setTimeout(() => sound.play('unlock'), 250);
    if (!hadAll && hasAllLivePoses()) {
      setTimeout(() => {
        this.toast('CERTIFIED LEGEND', 'Every live pose found. Gold frame unlocked — toggle it in Effects.', 'gold');
        sound.play('unlock');
      }, 1600);
    }
  }

  private updateTimer(now: number, s: Settings): void {
    if (!this.countdown && s.photoTimer && this.timerArmed && this.threeSince !== null && now - this.threeSince > 500) {
      this.timerArmed = false;
      this.startCountdown();
    }
    const cd = this.countdown;
    if (!cd) return;
    const remaining = cd.endsAt - now;
    const secs = Math.ceil(remaining / 1000);
    if (secs !== cd.lastTick && secs > 0) {
      cd.lastTick = secs;
      sound.play(secs === 1 ? 'tickFinal' : 'tick');
      this.patch({ countdown: secs });
    }
    if (remaining <= 0) {
      this.countdown = null;
      this.patch({ countdown: null });
      void this.capture().then((photo) => photo && this.onTimerCapture(photo));
    }
  }

  /** Starts a photo countdown (button or three-finger gesture). */
  startCountdown(seconds = TIMER_SECONDS): void {
    if (this.status.get().camera !== 'on' || this.countdown) return;
    this.countdown = { endsAt: performance.now() + seconds * 1000, seconds, lastTick: -1 };
  }

  cancelCountdown(): void {
    this.countdown = null;
    this.patch({ countdown: null });
  }

  /** Captures exactly what the preview shows (camera + look + stickers + reactions). */
  async capture(): Promise<CapturedPhoto | null> {
    const r = this.renderer;
    if (!r || this.status.get().camera !== 'on') return null;
    const format = settingsStore.get().format;
    const composed = composePhoto(r.canvas);
    this.director.flash(performance.now());
    sound.play('shutter');
    try {
      const blob = await canvasToBlob(composed, format);
      return { blob, url: URL.createObjectURL(blob), format, width: composed.width, height: composed.height };
    } catch (err) {
      console.error('[pose-cam] capture failed', err);
      this.toast('Couldn’t save that photo', 'Please try again.');
      return null;
    }
  }

  /**
   * Feeds landmarks through the normal classify → stabilize → react pipeline.
   * Used by the dev-only test hook (never exposed in production builds).
   */
  injectHands(hands: HandObservation[], aspect = 16 / 9): void {
    const now = performance.now();
    this.aspect = aspect;
    this.acceptHands(hands, now);
    const result = classifyFrame({ hands, aspect }, this.classifyOptions());
    this.lastVerdict = describeResult(result);
    this.applyResult(result, now);
  }

  dispose(): void {
    if (this.disposed) return;
    this.stop();
    this.disposed = true;
    this.unsubs.forEach((u) => u());
    this.unsubs = [];
    this.detector?.close();
    this.detector = null;
  }
}

function describeResult(r: FrameResult): string {
  if (r.kind === 'none') return 'no hands';
  if (r.kind === 'unknown') return `${r.hands.length} hand(s) · ${r.reason}`;
  return `${r.pose} ${(r.confidence * 100).toFixed(0)}%`;
}
