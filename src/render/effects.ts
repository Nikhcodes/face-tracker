/**
 * Camera looks built from canvas blend modes (no per-pixel JS work), so the
 * live preview and the captured photo are produced by the exact same code.
 * Blend modes are supported in all current browsers, unlike `ctx.filter`.
 */
import type { FilterId } from '../state/settings';

type Layer = { op: GlobalCompositeOperation; color: string; alpha: number };

export const FILTER_LAYERS: Record<FilterId, Layer[]> = {
  original: [],
  cinematic: [
    { op: 'soft-light', color: '#0f5e66', alpha: 0.42 },
    { op: 'soft-light', color: '#ff9a4d', alpha: 0.22 },
    { op: 'multiply', color: '#e9e2d6', alpha: 0.5 },
  ],
  mono: [
    { op: 'saturation', color: '#808080', alpha: 1 },
    { op: 'soft-light', color: '#2a2a2a', alpha: 0.25 },
  ],
  warm: [
    { op: 'soft-light', color: '#ff8a3d', alpha: 0.38 },
    { op: 'multiply', color: '#fff1dc', alpha: 0.6 },
  ],
  cool: [
    { op: 'soft-light', color: '#3d8bff', alpha: 0.36 },
    { op: 'multiply', color: '#e4f0ff', alpha: 0.55 },
  ],
};

export const FILTER_LABELS: Record<FilterId, string> = {
  original: 'Original',
  cinematic: 'Cinematic',
  mono: 'Mono',
  warm: 'Warm',
  cool: 'Cool',
};

export function applyFilter(ctx: CanvasRenderingContext2D, w: number, h: number, filter: FilterId): void {
  const layers = FILTER_LAYERS[filter];
  if (!layers.length) return;
  ctx.save();
  for (const l of layers) {
    ctx.globalCompositeOperation = l.op;
    ctx.globalAlpha = l.alpha;
    ctx.fillStyle = l.color;
    ctx.fillRect(0, 0, w, h);
  }
  ctx.restore();
}

/** Desaturates by `amount` (0..1); used by reactions. */
export function applyDesaturate(ctx: CanvasRenderingContext2D, w: number, h: number, amount: number): void {
  if (amount <= 0.01) return;
  ctx.save();
  ctx.globalCompositeOperation = 'saturation';
  ctx.globalAlpha = Math.min(1, amount);
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}

export class Vignette {
  private key = '';
  private gradient: CanvasGradient | null = null;

  draw(ctx: CanvasRenderingContext2D, w: number, h: number, strength: number): void {
    const key = `${w}x${h}x${strength}`;
    if (key !== this.key || !this.gradient) {
      const r = Math.hypot(w, h) / 2;
      const g = ctx.createRadialGradient(w / 2, h / 2, r * 0.45, w / 2, h / 2, r);
      g.addColorStop(0, 'rgba(0,0,0,0)');
      g.addColorStop(0.6, `rgba(0,0,0,${0.22 * strength})`);
      g.addColorStop(1, `rgba(0,0,0,${0.62 * strength})`);
      this.gradient = g;
      this.key = key;
    }
    ctx.save();
    ctx.fillStyle = this.gradient;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
  }
}

export class Grain {
  private tiles: HTMLCanvasElement[] = [];
  private patterns: (CanvasPattern | null)[] = [];
  private frame = 0;

  private ensure(ctx: CanvasRenderingContext2D) {
    if (this.tiles.length) return;
    const size = 160;
    for (let t = 0; t < 4; t++) {
      const c = document.createElement('canvas');
      c.width = size;
      c.height = size;
      const tctx = c.getContext('2d');
      if (!tctx) continue;
      const data = tctx.createImageData(size, size);
      for (let i = 0; i < data.data.length; i += 4) {
        const v = 128 + (Math.random() - 0.5) * 255;
        data.data[i] = v;
        data.data[i + 1] = v;
        data.data[i + 2] = v;
        data.data[i + 3] = 255;
      }
      tctx.putImageData(data, 0, 0);
      this.tiles.push(c);
      this.patterns.push(ctx.createPattern(c, 'repeat'));
    }
  }

  /** `animate` = false keeps the grain static (reduced motion). */
  draw(ctx: CanvasRenderingContext2D, w: number, h: number, animate: boolean): void {
    this.ensure(ctx);
    if (animate) this.frame++;
    const pattern = this.patterns[Math.floor(this.frame / 2) % Math.max(1, this.patterns.length)];
    if (!pattern) return;
    ctx.save();
    ctx.globalCompositeOperation = 'overlay';
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = pattern;
    // scale grain with resolution so it reads the same on any canvas size
    const s = Math.max(1, Math.round(Math.min(w, h) / 540));
    ctx.scale(s, s);
    ctx.fillRect(0, 0, w / s, h / s);
    ctx.restore();
  }
}
