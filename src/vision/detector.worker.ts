/// <reference lib="webworker" />
/**
 * Runs MediaPipe hand landmark detection off the main thread.
 * Frames arrive as transferred ImageBitmaps; results go back as plain objects.
 */
import { createLandmarker, toObservations, type LandmarkerHandle } from './landmarker';
import type { FromWorker, ToWorker } from './protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let handle: LandmarkerHandle | null = null;
let lastTimestamp = -1;

const post = (msg: FromWorker) => ctx.postMessage(msg);

ctx.onmessage = async (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    try {
      handle = await createLandmarker({ ...msg, useModule: true });
      post({ type: 'ready', delegate: handle.delegate });
    } catch (err) {
      post({ type: 'init-error', message: err instanceof Error ? err.message : String(err) });
    }
    return;
  }
  if (msg.type === 'detect') {
    const { frame, id } = msg;
    try {
      if (!handle) throw new Error('Detector not ready');
      // VIDEO mode requires strictly increasing timestamps.
      const ts = Math.max(msg.timestamp, lastTimestamp + 1);
      lastTimestamp = ts;
      const t0 = performance.now();
      const result = handle.landmarker.detectForVideo(frame, ts);
      post({ type: 'result', id, hands: toObservations(result), inferenceMs: performance.now() - t0 });
    } catch (err) {
      post({ type: 'detect-error', id, message: err instanceof Error ? err.message : String(err) });
    } finally {
      frame.close();
    }
    return;
  }
  if (msg.type === 'close') {
    handle?.landmarker.close();
    handle = null;
    ctx.close();
  }
};
