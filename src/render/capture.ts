import type { PhotoFormat } from '../state/settings';
import { COLORS, FONTS } from './paint';

export const MIME: Record<PhotoFormat, string> = { png: 'image/png', jpeg: 'image/jpeg' };

/** e.g. pose-cam-20261018-161530.png */
export function photoFilename(format: PhotoFormat, date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  const stamp = `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
  return `pose-cam-${stamp}.${format === 'jpeg' ? 'jpg' : 'png'}`;
}

/** Copies the composed camera canvas and adds a small wordmark. */
export function composePhoto(source: HTMLCanvasElement, watermark = true): HTMLCanvasElement {
  const out = document.createElement('canvas');
  out.width = source.width;
  out.height = source.height;
  const ctx = out.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D is not available');
  ctx.drawImage(source, 0, 0);
  if (watermark) {
    const unit = Math.min(out.width, out.height);
    const size = Math.max(12, Math.round(unit * 0.03));
    ctx.font = `800 ${size}px ${FONTS.display}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    const x = out.width - unit * 0.035;
    const y = out.height - unit * 0.03;
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillText('POSE CAM', x + 1, y + 1);
    ctx.fillStyle = COLORS.paper;
    ctx.fillText('POSE CAM', x, y);
    const tw = ctx.measureText('POSE CAM').width;
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath();
    ctx.arc(x - tw - size * 0.45, y - size * 0.38, size * 0.2, 0, Math.PI * 2);
    ctx.fill();
  }
  return out;
}

export function canvasToBlob(canvas: HTMLCanvasElement, format: PhotoFormat): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the photo'))),
      MIME[format],
      format === 'jpeg' ? 0.92 : undefined,
    );
  });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export type ShareOutcome = 'shared' | 'copied' | 'cancelled' | 'unavailable';

/** Shares the photo with the native share sheet if the browser can share files. */
export async function sharePhoto(blob: Blob, filename: string, url: string): Promise<ShareOutcome> {
  const file = new File([blob], filename, { type: blob.type });
  const data: ShareData = { files: [file], title: 'POSE CAM', text: `Made with POSE CAM — ${url}` };
  if (typeof navigator.share !== 'function' || typeof navigator.canShare !== 'function' || !navigator.canShare(data)) {
    return 'unavailable';
  }
  try {
    await navigator.share(data);
    return 'shared';
  } catch (err) {
    return (err as { name?: string })?.name === 'AbortError' ? 'cancelled' : 'unavailable';
  }
}

/** Shares a link to the app: native share sheet, else clipboard. */
export async function shareLink(url: string): Promise<ShareOutcome> {
  const data: ShareData = { title: 'POSE CAM', text: 'Strike a pose — your camera reacts.', url };
  if (typeof navigator.share === 'function' && (typeof navigator.canShare !== 'function' || navigator.canShare(data))) {
    try {
      await navigator.share(data);
      return 'shared';
    } catch (err) {
      if ((err as { name?: string })?.name === 'AbortError') return 'cancelled';
    }
  }
  return (await copyText(url)) ? 'copied' : 'unavailable';
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}
