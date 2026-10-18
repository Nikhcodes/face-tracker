import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SPECS, synthHand } from '../gestures/synthetic';

// ── Mocks: camera, detector and canvas rendering ───────────────────────────
const track = { stop: vi.fn(), addEventListener: vi.fn(), getSettings: () => ({ deviceId: 'cam-1' }) };
const stream = { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream;
const openCamera = vi.fn();
const detector = { backend: 'worker-cpu', detect: vi.fn().mockResolvedValue(null), close: vi.fn() };
const createDetector = vi.fn();

vi.mock('../camera/camera', async (orig) => {
  const actual = await orig<typeof import('../camera/camera')>();
  return {
    ...actual,
    openCamera: (...a: unknown[]) => openCamera(...a),
    listCameras: vi.fn().mockResolvedValue([{ deviceId: 'cam-1', label: 'Front', kind: 'videoinput' }]),
  };
});
vi.mock('../vision/detector', () => ({ createDetector: () => createDetector() }));
vi.mock('../render/renderer', () => ({
  StudioRenderer: class {
    canvas = { width: 1280, height: 720 };
    render = vi.fn();
  },
}));

const { StudioEngine } = await import('./StudioEngine');
const { unlockedStore } = await import('../state/progress');

function makeEngine() {
  const video = document.createElement('video');
  Object.defineProperty(video, 'readyState', { value: 4 });
  video.play = vi.fn().mockResolvedValue(undefined);
  const canvas = document.createElement('canvas');
  const engine = new StudioEngine();
  engine.attach(video, canvas);
  return { engine, video };
}

describe('StudioEngine lifecycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    openCamera.mockResolvedValue(stream);
    createDetector.mockResolvedValue(detector);
    unlockedStore.set(new Set());
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1));
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
  });
  afterEach(() => vi.unstubAllGlobals());

  it('does not touch the camera until start() is called', () => {
    makeEngine();
    expect(openCamera).not.toHaveBeenCalled();
    expect(createDetector).not.toHaveBeenCalled();
  });

  it('start() opens the camera and loads the detector once', async () => {
    const { engine, video } = makeEngine();
    await engine.start();
    await vi.waitFor(() => expect(engine.status.get().model).toBe('ready'));
    expect(engine.status.get().camera).toBe('on');
    expect(video.srcObject).toBe(stream);
    expect(createDetector).toHaveBeenCalledTimes(1);
    // restarting the camera reuses the loaded model
    engine.stop();
    await engine.start();
    expect(createDetector).toHaveBeenCalledTimes(1);
  });

  it('stop() stops every track, clears the video and cancels the frame loop', async () => {
    const { engine, video } = makeEngine();
    await engine.start();
    engine.stop();
    expect(track.stop).toHaveBeenCalled();
    expect(video.srcObject).toBeNull();
    expect(cancelAnimationFrame).toHaveBeenCalled();
    expect(engine.status.get().camera).toBe('off');
  });

  it('dispose() releases the camera and closes the detector', async () => {
    const { engine } = makeEngine();
    await engine.start();
    await vi.waitFor(() => expect(engine.status.get().model).toBe('ready'));
    engine.dispose();
    expect(track.stop).toHaveBeenCalled();
    expect(detector.close).toHaveBeenCalled();
  });

  it('reports a denied permission as a camera error', async () => {
    const { CameraError } = await import('../camera/camera');
    openCamera.mockRejectedValue(new CameraError('denied', 'blocked'));
    const { engine } = makeEngine();
    await engine.start();
    expect(engine.status.get()).toMatchObject({ camera: 'error', cameraError: 'denied' });
  });

  it('keeps the camera usable when the model fails to load', async () => {
    createDetector.mockRejectedValue(new Error('no wasm'));
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const { engine } = makeEngine();
    await engine.start();
    await vi.waitFor(() => expect(engine.status.get().model).toBe('error'));
    expect(engine.status.get().camera).toBe('on');
  });

  it('a held pose becomes the displayed state and unlocks it once', async () => {
    const { engine } = makeEngine();
    const toasts: string[] = [];
    engine.onToast = (t) => toasts.push(t.title);
    await engine.start();
    const hand = synthHand(SPECS.peace, { cy: 0.35 });
    const t0 = performance.now();
    let now = t0;
    vi.spyOn(performance, 'now').mockImplementation(() => now);
    for (let i = 0; i < 12; i++) {
      now = t0 + i * 33;
      engine.injectHands([hand]);
    }
    expect(engine.status.get().display).toMatchObject({ kind: 'pose', pose: 'peace' });
    expect(toasts).toEqual(['Pose unlocked: Peace']);
    expect(unlockedStore.get().has('peace')).toBe(true);
  });
});
