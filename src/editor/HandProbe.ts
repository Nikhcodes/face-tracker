/**
 * A lightweight camera + detector loop for the pose editor's tester.
 * Same local pipeline as the studio (camera → worker detector → landmarks),
 * without stickers or reactions.
 */
import { CameraError, mapCameraError, openCamera, stopStream, type CameraErrorCode } from '../camera/camera';
import { HAND_CONNECTIONS, type HandObservation } from '../gestures/types';
import { COLORS } from '../render/paint';
import { createDetector, type HandDetector } from '../vision/detector';

export interface ProbeState {
  status: 'off' | 'starting' | 'loading-model' | 'on' | 'error';
  error: CameraErrorCode | 'model' | null;
  hands: HandObservation[];
  aspect: number;
}

export class HandProbe {
  onUpdate: (s: ProbeState) => void = () => {};
  private state: ProbeState = { status: 'off', error: null, hands: [], aspect: 4 / 3 };
  private stream: MediaStream | null = null;
  private detector: HandDetector | null = null;
  private raf = 0;
  private busy = false;
  private lastEmit = 0;
  private lastVideoTime = -1;
  private gen = 0;

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly canvas: HTMLCanvasElement,
  ) {}

  private set(p: Partial<ProbeState>) {
    this.state = { ...this.state, ...p };
    this.onUpdate(this.state);
  }

  async start(): Promise<void> {
    if (this.state.status !== 'off' && this.state.status !== 'error') return;
    const gen = ++this.gen;
    this.set({ status: 'starting', error: null });
    try {
      const stream = await openCamera({ quality: 'performance' });
      if (gen !== this.gen) return stopStream(stream);
      this.stream = stream;
      this.video.srcObject = stream;
      this.video.muted = true;
      this.video.playsInline = true;
      await this.video.play().catch(() => undefined);
    } catch (err) {
      const e = err instanceof CameraError ? err : mapCameraError(err);
      this.set({ status: 'error', error: e.code });
      return;
    }
    this.set({ status: 'loading-model' });
    this.loop();
    try {
      this.detector ??= await createDetector();
      if (gen !== this.gen) return;
      this.set({ status: 'on' });
    } catch {
      if (gen === this.gen) this.set({ status: 'error', error: 'model' });
    }
  }

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    const v = this.video;
    if (v.readyState < 2) return;
    const w = 480;
    const h = Math.round((w * v.videoHeight) / Math.max(1, v.videoWidth));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0, w, h);
    ctx.restore();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(246,243,236,0.9)';
    for (const hand of this.state.hands) {
      const pts = hand.image.map((p) => ({ x: (1 - p.x) * w, y: p.y * h }));
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
      }
      ctx.stroke();
      ctx.fillStyle = COLORS.accent;
      for (const i of [4, 8, 12, 16, 20]) {
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (this.detector && !this.busy && v.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = v.currentTime;
      this.busy = true;
      const gen = this.gen;
      this.detector
        .detect(v, performance.now())
        .then((res) => {
          if (!res || gen !== this.gen) return;
          const now = performance.now();
          const aspect = res.width / Math.max(1, res.height);
          this.state = { ...this.state, hands: res.hands, aspect };
          // React only needs a few updates per second
          if (now - this.lastEmit > 120) {
            this.lastEmit = now;
            this.onUpdate(this.state);
          }
        })
        .finally(() => {
          this.busy = false;
        });
    }
  };

  stop(): void {
    this.gen++;
    cancelAnimationFrame(this.raf);
    stopStream(this.stream);
    this.stream = null;
    this.video.srcObject = null;
    this.set({ status: 'off', hands: [], error: null });
  }

  dispose(): void {
    this.stop();
    this.detector?.close();
    this.detector = null;
  }
}
