import type { StickerImage } from '../config/pack';

export const ACCEPTED_IMAGES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
const MAX_INPUT_BYTES = 15 * 1024 * 1024;
const MAX_SIDE = 720;

/**
 * Turns an uploaded file into a compact sticker (max 720 px, WebP when the
 * browser can encode it). Animated GIFs become their first frame.
 */
export async function fileToSticker(file: File): Promise<StickerImage> {
  if (!ACCEPTED_IMAGES.includes(file.type)) throw new Error('Use a PNG, JPEG, WebP or GIF image.');
  if (file.size > MAX_INPUT_BYTES) throw new Error('That image is over 15 MB. Try a smaller one.');
  const bitmap = await decode(file);
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('This browser can’t process images.');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, width, height);
  if ('close' in bitmap) bitmap.close();
  let src = canvas.toDataURL('image/webp', 0.86);
  if (!src.startsWith('data:image/webp')) {
    // no WebP encoder (older Safari): PNG keeps transparency, JPEG is smaller for photos
    const png = canvas.toDataURL('image/png');
    const jpg = canvas.toDataURL('image/jpeg', 0.88);
    src = file.type === 'image/png' && png.length < jpg.length * 2.5 ? png : jpg;
  }
  return { src, width, height };
}

async function decode(file: File): Promise<ImageBitmap | HTMLImageElement> {
  if (typeof createImageBitmap === 'function') {
    try {
      return await createImageBitmap(file);
    } catch {
      /* fall through */
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  } catch {
    throw new Error('That file couldn’t be read as an image.');
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
