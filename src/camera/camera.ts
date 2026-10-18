import type { Quality } from '../state/settings';

export type CameraErrorCode = 'unsupported' | 'insecure' | 'denied' | 'not-found' | 'in-use' | 'overconstrained' | 'unknown';

export class CameraError extends Error {
  constructor(
    readonly code: CameraErrorCode,
    message: string,
  ) {
    super(message);
    this.name = 'CameraError';
  }
}

export const CAMERA_ERROR_COPY: Record<CameraErrorCode, { title: string; body: string }> = {
  unsupported: {
    title: 'This browser can’t open a camera',
    body: 'Try a recent version of Chrome, Edge, Firefox or Safari.',
  },
  insecure: {
    title: 'Camera needs a secure connection',
    body: 'Browsers only allow cameras on https:// pages (or localhost). Open the secure version of this site.',
  },
  denied: {
    title: 'Camera permission was blocked',
    body: 'Allow camera access for this site in your browser’s address bar or site settings, then try again.',
  },
  'not-found': {
    title: 'No camera found',
    body: 'Connect a camera, or check that your device’s camera is enabled, then try again.',
  },
  'in-use': {
    title: 'Your camera is busy',
    body: 'Another app or tab may be using it. Close video calls or other camera apps, then try again.',
  },
  overconstrained: {
    title: 'That camera couldn’t start',
    body: 'Try another camera, or switch the performance mode in settings.',
  },
  unknown: {
    title: 'The camera couldn’t start',
    body: 'Something unexpected happened. Reload the page and try again.',
  },
};

export function checkCameraSupport(): CameraError | null {
  if (typeof window !== 'undefined' && window.isSecureContext === false) {
    return new CameraError('insecure', CAMERA_ERROR_COPY.insecure.title);
  }
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    return new CameraError('unsupported', CAMERA_ERROR_COPY.unsupported.title);
  }
  return null;
}

export function mapCameraError(err: unknown): CameraError {
  if (err instanceof CameraError) return err;
  const name = (err as { name?: string } | null)?.name ?? '';
  const code: CameraErrorCode =
    name === 'NotAllowedError' || name === 'PermissionDeniedError' || name === 'SecurityError'
      ? 'denied'
      : name === 'NotFoundError' || name === 'DevicesNotFoundError'
        ? 'not-found'
        : name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError'
          ? 'in-use'
          : name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError'
            ? 'overconstrained'
            : 'unknown';
  return new CameraError(code, CAMERA_ERROR_COPY[code].title);
}

export const QUALITY_PRESETS: Record<Quality, { width: number; height: number; frameRate: number }> = {
  performance: { width: 640, height: 480, frameRate: 30 },
  balanced: { width: 1280, height: 720, frameRate: 30 },
  high: { width: 1920, height: 1080, frameRate: 30 },
};

export interface OpenCameraOptions {
  deviceId?: string | null;
  quality: Quality;
}

export async function openCamera({ deviceId, quality }: OpenCameraOptions): Promise<MediaStream> {
  const unsupported = checkCameraSupport();
  if (unsupported) throw unsupported;
  const q = QUALITY_PRESETS[quality];
  const video: MediaTrackConstraints = {
    width: { ideal: q.width },
    height: { ideal: q.height },
    frameRate: { ideal: q.frameRate, max: 60 },
    ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' }),
  };
  try {
    return await navigator.mediaDevices.getUserMedia({ video, audio: false });
  } catch (err) {
    const mapped = mapCameraError(err);
    // A chosen device that rejects the size hints: retry with the device alone.
    if (mapped.code === 'overconstrained' || (mapped.code === 'not-found' && deviceId)) {
      try {
        return await navigator.mediaDevices.getUserMedia({ video: deviceId ? { deviceId } : true, audio: false });
      } catch (retryErr) {
        throw mapCameraError(retryErr);
      }
    }
    throw mapped;
  }
}

export function stopStream(stream: MediaStream | null | undefined): void {
  stream?.getTracks().forEach((t) => {
    t.stop();
  });
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'videoinput');
  } catch {
    return [];
  }
}
