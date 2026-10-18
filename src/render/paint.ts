/** Shared canvas drawing primitives and palette (mirrors the CSS design tokens). */
export const COLORS = {
  accent: '#c6ff3d',
  accentInk: '#0d1400',
  paper: '#f6f3ec',
  ink: '#0c0c0d',
  gold: '#e8c35a',
  heart: '#ff5c8a',
  confetti: ['#c6ff3d', '#f6f3ec', '#ff5c8a', '#5cc8ff', '#ffcf3d', '#a57bff'],
} as const;

export const FONTS = {
  display: `'Bricolage Grotesque Variable', 'Bricolage Grotesque', ui-sans-serif, system-ui, sans-serif`,
  ui: `'Inter Variable', Inter, ui-sans-serif, system-ui, sans-serif`,
} as const;

export function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

export function starPath(ctx: CanvasRenderingContext2D, points: number, outer: number, inner: number): void {
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const r = i % 2 === 0 ? outer : inner;
    const a = (i * Math.PI) / points - Math.PI / 2;
    if (i === 0) ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r);
    else ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
  }
  ctx.closePath();
}

export function heartPath(ctx: CanvasRenderingContext2D, s: number): void {
  ctx.beginPath();
  ctx.moveTo(0, s * 0.3);
  ctx.bezierCurveTo(-s * 0.05, s * 0.25, -s * 0.5, s * 0.0, -s * 0.5, -s * 0.22);
  ctx.bezierCurveTo(-s * 0.5, -s * 0.45, -s * 0.2, -s * 0.55, 0, -s * 0.3);
  ctx.bezierCurveTo(s * 0.2, -s * 0.55, s * 0.5, -s * 0.45, s * 0.5, -s * 0.22);
  ctx.bezierCurveTo(s * 0.5, s * 0.0, s * 0.05, s * 0.25, 0, s * 0.3);
  ctx.closePath();
}

/** Jagged comic-book burst. Deterministic for a given seed. */
export function burstPath(ctx: CanvasRenderingContext2D, rx: number, ry: number, spikes: number, seed: number): void {
  ctx.beginPath();
  for (let i = 0; i < spikes * 2; i++) {
    const jitter = 0.82 + 0.18 * Math.abs(Math.sin(seed * 12.9898 + i * 78.233));
    const outer = i % 2 === 0;
    const k = outer ? jitter : 0.72;
    const a = (i * Math.PI) / spikes;
    const x = Math.cos(a) * rx * k;
    const y = Math.sin(a) * ry * k;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

export const easeOutBack = (t: number) => {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  const x = Math.min(1, Math.max(0, t));
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
};
export const easeOutCubic = (t: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);

/** The lime caption "tape" used for reactions. Centred at (0,0) of the current transform. */
export function drawTape(ctx: CanvasRenderingContext2D, text: string, size: number, maxWidth: number): void {
  ctx.font = `800 ${size}px ${FONTS.display}`;
  let tw = ctx.measureText(text).width;
  if (tw > maxWidth) {
    size *= maxWidth / tw;
    ctx.font = `800 ${size}px ${FONTS.display}`;
    tw = ctx.measureText(text).width;
  }
  const padX = size * 0.45;
  const padY = size * 0.28;
  ctx.fillStyle = COLORS.accent;
  roundRect(ctx, -tw / 2 - padX, -size * 0.62 - padY, tw + padX * 2, size * 1.24 + padY * 2, size * 0.16);
  ctx.fill();
  ctx.fillStyle = COLORS.accentInk;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 0, size * 0.04);
}
