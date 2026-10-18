import type { FilesetResolver as FR, HandLandmarker as HL, HandLandmarkerResult } from '@mediapipe/tasks-vision';
import type { HandObservation, Handedness } from '../gestures/types';

export type Delegate = 'GPU' | 'CPU';

export interface LandmarkerHandle {
  landmarker: HL;
  delegate: Delegate;
}

/** Detector thresholds. Lower presence/tracking thresholds feel more responsive but flicker more. */
export const LANDMARKER_OPTIONS = {
  numHands: 2,
  minHandDetectionConfidence: 0.6,
  minHandPresenceConfidence: 0.6,
  minTrackingConfidence: 0.5,
} as const;

/**
 * Creates a HandLandmarker, trying the GPU delegate first and falling back to CPU.
 * The library is imported lazily so it is only downloaded when the camera opens.
 */
export async function createLandmarker(opts: {
  wasmBase: string;
  modelUrl: string;
  useModule: boolean;
  preferGpu: boolean;
}): Promise<LandmarkerHandle> {
  const vision: { FilesetResolver: typeof FR; HandLandmarker: typeof HL } = await import('@mediapipe/tasks-vision');
  const fileset = await vision.FilesetResolver.forVisionTasks(opts.wasmBase, opts.useModule);
  const order: Delegate[] = opts.preferGpu ? ['GPU', 'CPU'] : ['CPU'];
  let lastError: unknown = null;
  for (const delegate of order) {
    try {
      const landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: opts.modelUrl, delegate },
        runningMode: 'VIDEO',
        ...LANDMARKER_OPTIONS,
      });
      return { landmarker, delegate };
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Converts a MediaPipe result into plain, structured-clone-friendly observations. */
export function toObservations(result: HandLandmarkerResult): HandObservation[] {
  const hands: HandObservation[] = [];
  for (let i = 0; i < result.landmarks.length; i++) {
    const image = result.landmarks[i];
    const world = result.worldLandmarks[i];
    if (!image || !world || image.length < 21 || world.length < 21) continue;
    const cat = result.handedness?.[i]?.[0];
    const name = cat?.categoryName;
    const handedness: Handedness = name === 'Left' || name === 'Right' ? name : 'Unknown';
    hands.push({
      image: image.map((p) => ({ x: p.x, y: p.y, z: p.z })),
      world: world.map((p) => ({ x: p.x, y: p.y, z: p.z })),
      handedness,
      score: cat?.score ?? 1,
    });
  }
  return hands;
}
