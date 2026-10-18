import type { PoseDefinition, StickerRef } from '../config/poses';
import { registryStore } from '../config/registry';
import { glyphGeometry, paintGlyph } from './glyph';
import { COLORS, FONTS, roundRect } from './paint';

/**
 * Loads every sticker image from this site's own origin. Same-origin images
 * never taint the canvas, so photos can always be exported.
 */
export class StickerLibrary {
  private images = new Map<string, HTMLImageElement>();
  private pending = new Map<string, Promise<void>>();

  /** Loads one image (cached). Missing images resolve quietly and just aren't drawn. */
  private loadOne(src: string): Promise<void> {
    const existing = this.pending.get(src);
    if (existing) return existing;
    const p = new Promise<void>((resolve) => {
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => {
        this.images.set(src, img);
        resolve();
      };
      img.onerror = () => resolve();
      img.src = src;
    });
    this.pending.set(src, p);
    return p;
  }

  /** Preloads every sticker currently in the registry (built-in, overridden and custom). */
  load(): Promise<void> {
    const srcs = new Set<string>();
    for (const p of registryStore.get().all) for (const s of p.stickers) if (s.kind !== 'stamp') srcs.add(s.src);
    return Promise.all([...srcs].map((s) => this.loadOne(s))).then(() => undefined);
  }

  get(src: string): HTMLImageElement | undefined {
    const img = this.images.get(src);
    if (!img) void this.loadOne(src);
    return img;
  }
}

/** Aspect ratio (w/h) a sticker is drawn at. */
export function stickerAspect(ref: StickerRef): number {
  if (ref.kind === 'image') return ref.width / ref.height;
  if (ref.kind === 'sprite') return ref.frameWidth / ref.frameHeight;
  return 1;
}

/**
 * Draws a sticker card centred at (cx, cy) with width w. `t` is time in ms
 * since the sticker appeared (for sprite animation).
 */
export function drawStickerCard(
  ctx: CanvasRenderingContext2D,
  lib: StickerLibrary,
  pose: PoseDefinition,
  ref: StickerRef,
  cx: number,
  cy: number,
  w: number,
  t: number,
): void {
  const aspect = stickerAspect(ref);
  const h = w / aspect;
  const border = Math.max(3, w * 0.035);
  const radius = w * 0.07;
  ctx.save();
  ctx.translate(cx, cy);
  // card shadow
  ctx.shadowColor = 'rgba(0,0,0,0.45)';
  ctx.shadowBlur = w * 0.08;
  ctx.shadowOffsetY = w * 0.03;
  ctx.fillStyle = ref.kind === 'stamp' ? COLORS.ink : COLORS.paper;
  roundRect(ctx, -w / 2 - border, -h / 2 - border, w + border * 2, h + border * 2, radius + border);
  ctx.fill();
  ctx.shadowColor = 'transparent';

  if (ref.kind === 'stamp') {
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = border * 0.7;
    roundRect(ctx, -w / 2 - border * 0.4, -h / 2 - border * 0.4, w + border * 0.8, h + border * 0.8, radius + border * 0.6);
    ctx.stroke();
    const geo = glyphGeometry(pose.glyph);
    const g = w * 0.78;
    paintGlyph(ctx, geo, -g / 2, -h / 2 + w * 0.02, g, COLORS.accent, COLORS.ink);
    ctx.fillStyle = COLORS.paper;
    ctx.font = `800 ${Math.round(w * 0.19)}px ${FONTS.display}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(ref.label, 0, h / 2 - w * 0.06);
  } else {
    const img = lib.get(ref.src);
    ctx.save();
    roundRect(ctx, -w / 2, -h / 2, w, h, radius);
    ctx.clip();
    if (img) {
      if (ref.kind === 'image') {
        ctx.drawImage(img, -w / 2, -h / 2, w, h);
      } else {
        const frame = Math.floor(t / ref.frameMs) % ref.frames;
        const sx = (frame % ref.columns) * ref.frameWidth;
        const sy = Math.floor(frame / ref.columns) * ref.frameHeight;
        ctx.drawImage(img, sx, sy, ref.frameWidth, ref.frameHeight, -w / 2, -h / 2, w, h);
      }
    } else {
      ctx.fillStyle = '#ddd';
      ctx.fillRect(-w / 2, -h / 2, w, h);
    }
    ctx.restore();
  }
  ctx.restore();
}
