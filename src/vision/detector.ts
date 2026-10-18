/**
 * Hand detector facade.
 *
 * Preferred: a module Web Worker (detection never blocks rendering or input).
 * Fallback: the main thread, for browsers where the worker path fails
 * (no OffscreenCanvas, module-worker quirks, etc.).
 */
import type { HandObservation } from '../gestures/types';
import { assetUrls } from './assets';
import { createLandmarker, toObservations, type Delegate, type LandmarkerHandle } from './landmarker';
import type { FromWorker, ToWorker } from './protocol';

export interface DetectionResult {
  hands: HandObservation[];
  width: number;
  height: number;
  inferenceMs: number;
}

export type DetectorBackend = `${'worker' | 'main'}-${Lowercase<Delegate>}`;

export interface HandDetector {
  readonly backend: DetectorBackend;
  /** Detects hands in the current video frame. Resolves null if the frame was skipped. */
  detect(video: HTMLVideoElement, timestamp: number): Promise<DetectionResult | null>;
  close(): void;
}

/** Longest side of the frame handed to the detector. MediaPipe downsamples internally anyway. */
const DETECT_MAX_SIDE = 640;
const INIT_TIMEOUT_MS = 25_000;

function frameSize(video: HTMLVideoElement) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const s = Math.min(1, DETECT_MAX_SIDE / Math.max(w, h));
  return { width: w, height: h, scaledW: Math.round(w * s), scaledH: Math.round(h * s) };
}

class WorkerDetector implements HandDetector {
  backend: DetectorBackend = 'worker-gpu';
  private nextId = 1;
  private pending = new Map<number, { resolve: (r: DetectionResult | null) => void; width: number; height: number }>();
  private closed = false;

  constructor(private readonly worker: Worker) {
    worker.addEventListener('message', (e: MessageEvent<FromWorker>) => {
      const msg = e.data;
      if (msg.type === 'result' || msg.type === 'detect-error') {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        p.resolve(msg.type === 'result' ? { hands: msg.hands, width: p.width, height: p.height, inferenceMs: msg.inferenceMs } : null);
      }
    });
  }

  static async create(preferGpu: boolean): Promise<WorkerDetector> {
    if (typeof Worker === 'undefined' || typeof createImageBitmap === 'undefined' || typeof OffscreenCanvas === 'undefined') {
      throw new Error('Worker detection unsupported');
    }
    const worker = new Worker(new URL('./detector.worker.ts', import.meta.url), { type: 'module', name: 'pose-cam-detector' });
    const detector = new WorkerDetector(worker);
    const delegate = await new Promise<Delegate>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Detector start-up timed out')), INIT_TIMEOUT_MS);
      const onMessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.type === 'ready') {
          clearTimeout(timer);
          worker.removeEventListener('message', onMessage);
          resolve(e.data.delegate);
        } else if (e.data.type === 'init-error') {
          clearTimeout(timer);
          worker.removeEventListener('message', onMessage);
          reject(new Error(e.data.message));
        }
      };
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', (e) => {
        clearTimeout(timer);
        reject(new Error(e.message || 'Detector worker failed to load'));
      });
      const init: ToWorker = { type: 'init', ...assetUrls(), preferGpu };
      worker.postMessage(init);
    }).catch((err) => {
      worker.terminate();
      throw err;
    });
    detector.backend = delegate === 'GPU' ? 'worker-gpu' : 'worker-cpu';
    return detector;
  }

  async detect(video: HTMLVideoElement, timestamp: number): Promise<DetectionResult | null> {
    if (this.closed || video.readyState < 2 || !video.videoWidth) return null;
    const { width, height, scaledW, scaledH } = frameSize(video);
    let frame: ImageBitmap;
    try {
      frame = await createImageBitmap(video, { resizeWidth: scaledW, resizeHeight: scaledH, resizeQuality: 'low' });
    } catch {
      return null;
    }
    if (this.closed) {
      frame.close();
      return null;
    }
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, { resolve, width, height });
      const msg: ToWorker = { type: 'detect', id, frame, timestamp };
      this.worker.postMessage(msg, [frame]);
    });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const p of this.pending.values()) p.resolve(null);
    this.pending.clear();
    try {
      this.worker.postMessage({ type: 'close' } satisfies ToWorker);
    } catch {
      /* already gone */
    }
    // give the worker a moment to free GPU resources, then make sure it's gone
    setTimeout(() => this.worker.terminate(), 500);
  }
}

class MainThreadDetector implements HandDetector {
  readonly backend: DetectorBackend;
  private lastTimestamp = -1;
  private closed = false;

  private constructor(private readonly handle: LandmarkerHandle) {
    this.backend = handle.delegate === 'GPU' ? 'main-gpu' : 'main-cpu';
  }

  static async create(preferGpu: boolean): Promise<MainThreadDetector> {
    const handle = await createLandmarker({ ...assetUrls(window.location.origin), useModule: false, preferGpu });
    return new MainThreadDetector(handle);
  }

  async detect(video: HTMLVideoElement, timestamp: number): Promise<DetectionResult | null> {
    if (this.closed || video.readyState < 2 || !video.videoWidth) return null;
    const ts = Math.max(timestamp, this.lastTimestamp + 1);
    this.lastTimestamp = ts;
    const t0 = performance.now();
    const result = this.handle.landmarker.detectForVideo(video, ts);
    return {
      hands: toObservations(result),
      width: video.videoWidth,
      height: video.videoHeight,
      inferenceMs: performance.now() - t0,
    };
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.handle.landmarker.close();
  }
}

export interface CreateDetectorOptions {
  preferGpu?: boolean;
  /** Skip the worker (used by tests / troubleshooting). */
  forceMainThread?: boolean;
}

export async function createDetector(opts: CreateDetectorOptions = {}): Promise<HandDetector> {
  const preferGpu = opts.preferGpu ?? true;
  if (!opts.forceMainThread) {
    try {
      return await WorkerDetector.create(preferGpu);
    } catch (err) {
      console.info('[pose-cam] Worker detector unavailable, using main thread:', err instanceof Error ? err.message : err);
    }
  }
  return MainThreadDetector.create(preferGpu);
}
