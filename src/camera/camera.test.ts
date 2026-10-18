import { afterEach, describe, expect, it, vi } from 'vitest';
import { checkCameraSupport, mapCameraError, openCamera, stopStream } from './camera';

const domErr = (name: string) => Object.assign(new Error(name), { name });

describe('camera errors', () => {
  it.each([
    ['NotAllowedError', 'denied'],
    ['SecurityError', 'denied'],
    ['NotFoundError', 'not-found'],
    ['NotReadableError', 'in-use'],
    ['OverconstrainedError', 'overconstrained'],
    ['WeirdError', 'unknown'],
  ])('%s → %s', (name, code) => {
    expect(mapCameraError(domErr(name)).code).toBe(code);
  });
});

describe('camera lifecycle', () => {
  const original = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');
  afterEach(() => {
    if (original) Object.defineProperty(navigator, 'mediaDevices', original);
    else delete (navigator as { mediaDevices?: unknown }).mediaDevices;
    vi.restoreAllMocks();
  });

  it('reports unsupported browsers instead of throwing a TypeError', async () => {
    Object.defineProperty(navigator, 'mediaDevices', { value: undefined, configurable: true });
    expect(checkCameraSupport()?.code).toBe('unsupported');
    await expect(openCamera({ quality: 'balanced' })).rejects.toMatchObject({ code: 'unsupported' });
  });

  it('maps a denied permission prompt', async () => {
    const getUserMedia = vi.fn().mockRejectedValue(domErr('NotAllowedError'));
    Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia }, configurable: true });
    await expect(openCamera({ quality: 'balanced' })).rejects.toMatchObject({ code: 'denied' });
    expect(getUserMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
  });

  it('retries without size hints when constraints cannot be met', async () => {
    const stream = { getTracks: () => [] } as unknown as MediaStream;
    const getUserMedia = vi.fn().mockRejectedValueOnce(domErr('OverconstrainedError')).mockResolvedValueOnce(stream);
    Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia }, configurable: true });
    await expect(openCamera({ quality: 'high', deviceId: 'cam-2' })).resolves.toBe(stream);
    expect(getUserMedia).toHaveBeenLastCalledWith({ video: { deviceId: 'cam-2' }, audio: false });
  });

  it('stopStream stops every track', () => {
    const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
    stopStream({ getTracks: () => tracks } as unknown as MediaStream);
    tracks.forEach((t) => expect(t.stop).toHaveBeenCalledOnce());
    expect(() => stopStream(null)).not.toThrow();
  });
});
