import { afterEach, describe, expect, it, vi } from 'vitest';
import { canvasToBlob, composePhoto, photoFilename, shareLink, sharePhoto } from './capture';
import { canvasSizeFor } from './renderer';

describe('photo helpers', () => {
  it('builds sortable filenames with the right extension', () => {
    const d = new Date(2026, 9, 18, 16, 5, 9);
    expect(photoFilename('png', d)).toBe('pose-cam-20261018-160509.png');
    expect(photoFilename('jpeg', d)).toBe('pose-cam-20261018-160509.jpg');
  });

  it('keeps the camera aspect ratio when sizing the canvas', () => {
    expect(canvasSizeFor(1920, 1080, 'balanced')).toEqual({ width: 1280, height: 720 });
    expect(canvasSizeFor(720, 1280, 'performance')).toEqual({ width: 540, height: 960 });
    expect(canvasSizeFor(640, 480, 'high')).toEqual({ width: 640, height: 480 });
  });
});

describe('composePhoto / canvasToBlob', () => {
  afterEach(() => vi.restoreAllMocks());

  it('copies the composed preview (not the raw video) and adds a wordmark', () => {
    const calls: string[] = [];
    const fakeCtx = {
      drawImage: vi.fn(() => calls.push('drawImage')),
      fillText: vi.fn((t: string) => calls.push(`text:${t}`)),
      measureText: () => ({ width: 100 }),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
    };
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(fakeCtx as unknown as CanvasRenderingContext2D);
    const source = document.createElement('canvas');
    source.width = 1280;
    source.height = 720;
    const out = composePhoto(source);
    expect(out.width).toBe(1280);
    expect(out.height).toBe(720);
    expect(fakeCtx.drawImage).toHaveBeenCalledWith(source, 0, 0);
    expect(calls[0]).toBe('drawImage');
    expect(calls).toContain('text:POSE CAM');
  });

  it('encodes with the requested type', async () => {
    const canvas = document.createElement('canvas');
    const toBlob = vi.fn((cb: BlobCallback, type?: string) => cb(new Blob(['x'], { type })));
    canvas.toBlob = toBlob as unknown as HTMLCanvasElement['toBlob'];
    const blob = await canvasToBlob(canvas, 'jpeg');
    expect(blob.type).toBe('image/jpeg');
    expect(toBlob).toHaveBeenCalledWith(expect.any(Function), 'image/jpeg', 0.92);
  });

  it('rejects if the browser cannot encode (e.g. tainted canvas)', async () => {
    const canvas = document.createElement('canvas');
    canvas.toBlob = ((cb: BlobCallback) => cb(null)) as unknown as HTMLCanvasElement['toBlob'];
    await expect(canvasToBlob(canvas, 'png')).rejects.toThrow();
  });
});

describe('sharing', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete (navigator as { share?: unknown }).share;
    delete (navigator as { canShare?: unknown }).canShare;
  });

  it('reports unavailable when the Web Share API cannot share files', async () => {
    expect(await sharePhoto(new Blob(['x'], { type: 'image/png' }), 'a.png', 'https://example.com')).toBe('unavailable');
  });

  it('falls back to copying the link when navigator.share is missing', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    expect(await shareLink('https://pose.cam')).toBe('copied');
    expect(writeText).toHaveBeenCalledWith('https://pose.cam');
  });

  it('uses the native share sheet when present and treats dismissal as cancelled', async () => {
    Object.defineProperty(navigator, 'share', {
      value: vi.fn().mockRejectedValue(Object.assign(new Error('x'), { name: 'AbortError' })),
      configurable: true,
    });
    expect(await shareLink('https://pose.cam')).toBe('cancelled');
  });
});
