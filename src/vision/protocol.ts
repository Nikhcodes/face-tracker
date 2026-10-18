import type { HandObservation } from '../gestures/types';
import type { Delegate } from './landmarker';

export type ToWorker =
  | { type: 'init'; wasmBase: string; modelUrl: string; preferGpu: boolean }
  | { type: 'detect'; id: number; frame: ImageBitmap; timestamp: number }
  | { type: 'close' };

export type FromWorker =
  | { type: 'ready'; delegate: Delegate }
  | { type: 'init-error'; message: string }
  | { type: 'result'; id: number; hands: HandObservation[]; inferenceMs: number }
  | { type: 'detect-error'; id: number; message: string };
